#include "classifier.cuh"
#include <algorithm>
#include <assert.h>
#include <float.h>
#include <functional>
#include <iostream>
#include <math.h>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

// Row major index
__device__ inline unsigned int RM_Index(unsigned int row, unsigned int col,
                                        unsigned int width) {
  return (row * width + col);
}

// ************ Gausssian **************

GaussianNB::GaussianNB() {}

GaussianNB::~GaussianNB() {}

__global__ void GaussianNBSumKernel(const float *d_data, const int *d_labels,
                                    float *feature_means_, int *class_count_,
                                    unsigned int n_samples_,
                                    unsigned int n_classes_,
                                    unsigned int n_features_) {

  // Each thread will take care of one feature for all training samples
  unsigned int tidx = threadIdx.x;
  unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, row = 0;

  if (feat_col < n_features_) { /* End condition check */

    for (i = 0; i < n_samples_; ++i) { /* For each training sample */
      row = d_labels[i];

      // No race condition since each thread deals with one feature only
      feature_means_[RM_Index(row, feat_col, n_features_)] +=
          d_data[RM_Index(i, feat_col, n_features_)];

      // WARNING: thread divergence :/
      if (feat_col == 0) {
        class_count_[row] += 1;
      }
    }
  }
  return;
}

__global__ void GaussianNBMeanKernel(float *feature_means_, int *class_count_,
                                     float *class_priors_,
                                     unsigned int n_samples_,
                                     unsigned int n_classes_,
                                     unsigned int n_features_) {

  // Each thread will take care of one feature for all training samples
  unsigned int tidx = threadIdx.x;
  unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0;

  if (feat_col < n_features_) { /* End condition check */

    /* Calculate Means */
    for (i = 0; i < n_classes_; ++i) { /* For each class */
      feature_means_[RM_Index(i, feat_col, n_features_)] /= class_count_[i];

      // WARNING: thread divergence
      // Calculating Class priors
      if (feat_col == 0) {
        class_priors_[i] = (float)class_count_[i] / n_samples_;
      }
    }
  }
}

__global__ void GaussianNBVarKernel(const float *d_data, const int *d_labels,
                                    const float *feature_means_,
                                    float *feature_vars_,
                                    const int *class_count_,
                                    const unsigned int n_samples_,
                                    const unsigned int n_classes_,
                                    const unsigned int n_features_) {

  // Each thread will take care of one feature for all training samples
  unsigned int tidx = threadIdx.x;
  unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, row = 0;

  // Calculate variances
  if (feat_col < n_features_) {        /* End condition check */
    for (i = 0; i < n_samples_; ++i) { /* For each sample */
      row = d_labels[i];
      feature_vars_[RM_Index(row, feat_col, n_features_)] +=
          pow(d_data[RM_Index(i, feat_col, n_features_)] -
                  feature_means_[RM_Index(row, feat_col, n_features_)],
              2);
    }

    // Calculate coefficients
    for (i = 0; i < n_classes_; ++i) { /* For each class */
      feature_vars_[RM_Index(i, feat_col, n_features_)] /= class_count_[i];
    }
  }
}

void GaussianNB::train(vector<float> data, vector<int> labels) {
  unsigned int train_size = labels.size();
  n_features_ = data.size() / train_size;

  // Move data and labels to GPU memory
  // NOTE: Memory Operation, put checks later
  float *d_data;
  cudaMallocManaged(&d_data, (n_features_ * train_size) * sizeof(float));
  cudaMemcpy(d_data, &data[0], (n_features_ * train_size) * sizeof(float),
             cudaMemcpyHostToDevice);

  // NOTE: Memory Operation, put checks later
  int *d_labels;
  cudaMallocManaged(&d_labels, train_size * sizeof(int));
  cudaMemcpy(d_labels, &labels[0], train_size * sizeof(int),
             cudaMemcpyHostToDevice);

  // Use thrust to find unique labels -- declare a thrust label list
  thrust::device_vector<int> thr_label_list(d_labels, d_labels + train_size);
  thrust::sort(thr_label_list.begin(), thr_label_list.end());
  auto new_last = thrust::unique(thr_label_list.begin(), thr_label_list.end());
  thr_label_list.erase(new_last, thr_label_list.end());
  n_classes_ = new_last - thr_label_list.begin(); // number of unique classes

  int *d_label_list;
  cudaMallocManaged(&d_label_list, n_classes_ * sizeof(int));
  cudaMemcpy(d_label_list, thrust::raw_pointer_cast(thr_label_list.data()),
             n_classes_ * sizeof(int), cudaMemcpyHostToDevice);

  /* Other initializations */
  cudaMallocManaged(&feature_means_,
                    (n_classes_ * n_features_) * sizeof(float));
  cudaMallocManaged(&feature_vars_,
                    (n_classes_ * n_features_) * sizeof(float));
  cudaMallocManaged(&class_priors_, (n_classes_) * sizeof(float));
  cudaMallocManaged(&class_count_, n_classes_ * sizeof(int));

  /* Calculate the mean for each feature for each class
  Individual thread for each feature. Threads_per_block=1024 */
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(n_features_) / float(threads_per_block.x)));

  GaussianNBSumKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_means_, class_count_, train_size, n_classes_,
      n_features_);

  GaussianNBMeanKernel<<<blocks_per_grid, threads_per_block>>>(
      feature_means_, class_count_, class_priors_, train_size, n_classes_,
      n_features_);

  GaussianNBVarKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_means_, feature_vars_, class_count_, train_size,
      n_classes_, n_features_);

  cudaDeviceSynchronize();
  return;
}

__global__ void GaussianNBTestKernel(const float *d_data, const int *d_labels,
                                     const float *feature_means_,
                                     const float *feature_vars_,
                                     const float *class_priors_, int test_size,
                                     int n_classes_, int n_features_,
                                     int *score) {
  /* Each thread will take one term */
  unsigned int tidx = threadIdx.x;
  unsigned int sample_num = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, j = 0;
  float prob_class = 0.0;
  float max = 0;
  int result = 0;
  float coefficient = 0.0;

  if (sample_num < test_size) {        /* End condition check */
    for (i = 0; i < n_classes_; ++i) { /* For each class */
      prob_class = class_priors_[i];

      for (j = 0; j < n_features_; ++j) { /* For each feature */
        coefficient =
            1.0 / sqrt(2 * M_PI * feature_vars_[RM_Index(i, j, n_features_)]);
        prob_class *= coefficient *
                      exp(-pow(d_data[RM_Index(sample_num, j, n_features_)] -
                                   feature_means_[RM_Index(i, j, n_features_)],
                               2) /
                          (2 * feature_vars_[RM_Index(i, j, n_features_)]));
      }

      if (max < prob_class) {
        max = prob_class;
        result = i;
      }
    }

    if (result == d_labels[sample_num]) {
      score[sample_num] = 1;
    } else {
      score[sample_num] = 0;
    }
  }
}

int GaussianNB::predict(vector<float> data, vector<int> labels) {
  std::vector<int>::size_type test_size = labels.size();
  int total_score = 0;

  /* Moving test data to the device */
  float *d_data;
  cudaMallocManaged(&d_data, (n_features_ * test_size) * sizeof(float));
  cudaMemcpy(d_data, &data[0], (n_features_ * test_size) * sizeof(float),
             cudaMemcpyHostToDevice);
  int *d_labels;
  cudaMallocManaged(&d_labels, test_size * sizeof(int));
  cudaMemcpy(d_labels, &labels[0], test_size * sizeof(int),
             cudaMemcpyHostToDevice);

  /* Score keeper : 0 or 1 corresponding to each test sample */
  int *score;
  cudaMallocManaged(&score, test_size * sizeof(int));

  // One thread for each test sample
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(test_size) / float(threads_per_block.x)));

  GaussianNBTestKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_means_, feature_vars_, class_priors_, test_size,
      n_classes_, n_features_, score);

  cudaDeviceSynchronize();

  // Reduce score to a total score using thrust reduction
  thrust::device_vector<int> temp_vec(score, score + test_size);
  total_score =
      thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());

  return total_score;
}

//**********Multionomial ******************

MultinomialNB::MultinomialNB() {}

MultinomialNB::~MultinomialNB() {}

__global__ void
MultinomialNBCalcKernel(const float *d_data, const int *d_labels,
                        float *feature_probs, float *class_priors,
                        unsigned int n_samples_, unsigned int n_classes_,
                        unsigned int n_features_) {

  // Each thread will take care of one term for all docs
  unsigned int tidx = threadIdx.x;
  unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, row = 0;

  if (feat_col < n_features_) { /* End condition check */

    /* For each document / sample */
    for (i = 0; i < n_samples_; ++i) {
      row = d_labels[i];

      // No race condition since each thread deals with one feature only
      feature_probs[RM_Index(row, feat_col, n_features_)] +=
          d_data[RM_Index(i, feat_col, n_features_)];

      // WARNING: thread divergence :(
      if (feat_col == 0) {
        class_priors[row] += 1;
      }
    }
  }
  return;
}

/* Kernel divides each row by a number and takes log */
__global__ void
MultinomialNBLearnKernel(float *feature_probs, float *class_priors,
                         const float *d_row_sums, unsigned int n_samples_,
                         unsigned int n_classes_, unsigned int n_features_) {

  /* Each thread will take one term */
  unsigned int tidx = threadIdx.x;
  unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0;

  if (feat_col < n_features_) { /* End condition check */
    /* For each label */
    for (i = 0; i < n_classes_; ++i) {
      // TODO: Add Laplace Smoothing
      feature_probs[RM_Index(i, feat_col, n_features_)] = log(
          feature_probs[RM_Index(i, feat_col, n_features_)] / d_row_sums[i]);

      if (feat_col == 0) {
        class_priors[i] = log(class_priors[i] / (float)n_samples_);
      }
    }
  }
}

__global__ void MultinomialNBTestKernel(const float *d_data,
                                        const int *d_labels,
                                        const float *feature_probs,
                                        const float *class_priors,
                                        int test_size, int n_classes_,
                                        int n_features_, int *score) {
  /* Each thread will take one term */
  unsigned int tidx = threadIdx.x;
  unsigned int sample_num = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, j = 0;
  float prob_class = 0;
  float max = 0;
  int result = 0;

  if (sample_num < test_size) {
    for (i = 0; i < n_classes_; ++i) { /* For each class */
      prob_class = class_priors[i];

      for (j = 0; j < n_features_; ++j) { /* For each feature */
        prob_class += (d_data[RM_Index(sample_num, j, n_features_)] *
                       feature_probs[RM_Index(i, j, n_features_)]);
      }

      if (max < prob_class) {
        max = prob_class;
        result = i;
      }
    }

    if (result == d_labels[sample_num]) {
      score[sample_num] = 1;
    } else {
      score[sample_num] = 0;
    }
  }
  return;
}

void MultinomialNB::train(vector<float> data, vector<int> labels) {
  unsigned int train_size = labels.size();
  n_features_ = data.size() / train_size;

  // Move data and labels to GPU memory
  // NOTE: Memory Operation, put checks later
  float *d_data;
  cudaMallocManaged(&d_data, (n_features_ * train_size) * sizeof(float));
  cudaMemcpy(d_data, &data[0], (n_features_ * train_size) * sizeof(float),
             cudaMemcpyHostToDevice);

  // NOTE: Memory Operation, put checks later
  int *d_labels;
  cudaMallocManaged(&d_labels, train_size * sizeof(int));
  cudaMemcpy(d_labels, &labels[0], train_size * sizeof(int),
             cudaMemcpyHostToDevice);

  // Use thrust to find unique labels -- declare a thrust label list
  // WARNING: is it really worth it to sort in thrust?
  thrust::device_vector<int> thr_label_list(d_labels, d_labels + train_size);
  thrust::sort(thr_label_list.begin(), thr_label_list.end());
  auto new_last = thrust::unique(thr_label_list.begin(), thr_label_list.end());
  thr_label_list.erase(new_last, thr_label_list.end());
  n_classes_ = new_last - thr_label_list.begin();

  int *d_label_list;
  cudaMallocManaged(&d_label_list, n_classes_ * sizeof(int));
  cudaMemcpy(d_label_list, thrust::raw_pointer_cast(thr_label_list.data()),
             n_classes_ * sizeof(int), cudaMemcpyHostToDevice);

  /* Other initializations */
  cudaMallocManaged(&feature_probs,
                    (n_classes_ * n_features_) * sizeof(float));
  cudaMallocManaged(&class_priors, n_classes_ * sizeof(float));
  // Is the memset below required?
  // cudaMemset(feature_probs, 0, (n_classes_ * n_features_) * sizeof(float));

  /* Calculate frequency of occurence of each term : CalcKernel
  Individual thread for each term. Threads_per_block=1024 */
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(n_features_) / float(threads_per_block.x)));
  MultinomialNBCalcKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_probs, class_priors, train_size, n_classes_,
      n_features_);
  cudaDeviceSynchronize();

  /* Learning Phase: Calculate conditional probabilities */
  float *d_row_sums;
  cudaMallocManaged(&d_row_sums, n_classes_ * sizeof(float));

  /* Find total number of terms in each class */
  for (unsigned int i = 0; i < n_classes_; ++i) {
    thrust::device_vector<float> temp_vec(feature_probs + (n_features_ * i),
                                           feature_probs +
                                               (n_features_ * (i + 1)));
    d_row_sums[i] =
        thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());
  }

  thrust::device_vector<float> temp_vec(d_row_sums, (d_row_sums + n_classes_));

  MultinomialNBLearnKernel<<<blocks_per_grid, threads_per_block>>>(
      feature_probs, class_priors, d_row_sums, train_size, n_classes_,
      n_features_);
  cudaDeviceSynchronize();

  return;
}

int MultinomialNB::predict(vector<float> data, vector<int> labels) {
  std::vector<int>::size_type test_size = labels.size();
  int total_score = 0;

  /* Moving test data to the device */
  float *d_data;
  cudaMallocManaged(&d_data, (n_features_ * test_size) * sizeof(float));
  cudaMemcpy(d_data, &data[0], (n_features_ * test_size) * sizeof(float),
             cudaMemcpyHostToDevice);
  int *d_labels;
  cudaMallocManaged(&d_labels, test_size * sizeof(int));
  cudaMemcpy(d_labels, &labels[0], test_size * sizeof(int),
             cudaMemcpyHostToDevice);

  /* NOTE: The class priors and conditional probabilities should already
  be on the device after train */

  /* Score keeper : 0 or 1 corresponding to each test sample */
  int *score;
  cudaMallocManaged(&score, test_size * sizeof(int));

  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(test_size) / float(threads_per_block.x)));
  MultinomialNBTestKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_probs, class_priors, test_size, n_classes_,
      n_features_, score);

  cudaDeviceSynchronize();

  // Reduce score to a total score using thrust reduction
  thrust::device_vector<int> temp_vec(score, score + test_size);
  total_score =
      thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());

  return total_score;
}

// ***************************************************

__global__ void BernoulliNBCalcKernel(const float *d_data, const int *d_labels,
                                      float *feature_probs,
                                      float *class_count_,
                                      unsigned int n_samples_,
                                      unsigned int n_classes_,
                                      unsigned int n_features_) {

  // Each thread will take care of one term for all docs
  unsigned int tidx = threadIdx.x;
  unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, row = 0;

  if (feat_col < n_features_) { // End condition check

    // For each document / sample
    for (i = 0; i < n_samples_; ++i) {
      row = d_labels[i];

      // No race condition since each thread deals with one feature only
      feature_probs[RM_Index(row, feat_col, n_features_)] +=
          d_data[RM_Index(i, feat_col, n_features_)];

      // WARNING: thread divergence :(
      if (feat_col == 0) {
        class_count_[row] += 1;
      }
    }
  }
  return;
}

// Kernel divides each row by a number and takes log
__global__ void
BernoulliNBLearnKernel(float *feature_probs, float *class_count_,
                       const float *d_row_sums, unsigned int n_samples_,
                       unsigned int n_classes_, unsigned int n_features_) {

  // Each thread will take one term
  unsigned int tidx = threadIdx.x;
  unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0;

  if (feat_col < n_features_) { // End condition check
    // For each label
    for (i = 0; i < n_classes_; ++i) {
      feature_probs[RM_Index(i, feat_col, n_features_)] /=
          class_count_[i]; // d_row_sums[i];

      if (feat_col == 0) {
        class_count_[i] = class_count_[i] / (float)n_samples_;
        printf("prior at %u = %lf\n", i, class_count_[i]);
      }
    }
  }
}

__global__ void BernoulliNBTestKernel(const float *d_data, const int *d_labels,
                                      const float *feature_probs,
                                      const float *class_count_, int test_size,
                                      int n_classes_, int n_features_,
                                      int *score) {
  // Each thread will take one term
  unsigned int tidx = threadIdx.x;
  unsigned int sample_num = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, j = 0;
  float prob_class = 0;
  float min = 0;
  int result = 0;

  if (sample_num < test_size) {
    for (i = 0; i < n_classes_; ++i) { // For each class
      prob_class = log(class_count_[i]);

      for (j = 0; j < n_features_; ++j) { // For each feature
        prob_class += log(feature_probs[RM_Index(i, j, n_features_)])*d_data[RM_Index(sample_num, j, n_features_)];
        prob_class += log((1 - feature_probs[RM_Index(i, j, n_features_)]))*(1 - d_data[RM_Index(sample_num, j, n_features_)]);
      }

      if (min > prob_class) {
        min = prob_class;
        result = i;
      }
    }

    if (result == d_labels[sample_num]) {
      score[sample_num] = 1;

    } else {
      score[sample_num] = 0;
    }
  }

  return;
}

BernoulliNB::BernoulliNB() {}

BernoulliNB::~BernoulliNB() {}

void BernoulliNB::train(vector<float> data, vector<int> labels) {
  unsigned int train_size = labels.size();
  n_features_ = data.size() / train_size;

  // Move data and labels to GPU memory
  // NOTE: Memory Operation, put checks later
  float *d_data;
  cudaMallocManaged(&d_data, (n_features_ * train_size) * sizeof(float));
  cudaMemcpy(d_data, &data[0], (n_features_ * train_size) * sizeof(float),
             cudaMemcpyHostToDevice);

  // NOTE: Memory Operation, put checks later
  int *d_labels;
  cudaMallocManaged(&d_labels, train_size * sizeof(int));
  cudaMemcpy(d_labels, &labels[0], train_size * sizeof(int),
             cudaMemcpyHostToDevice);

  // Use thrust to find unique labels -- declare a thrust label list
  // WARNING: is it really worth it to sort in thrust?
  thrust::device_vector<int> thr_label_list(d_labels, d_labels + train_size);
  thrust::sort(thr_label_list.begin(), thr_label_list.end());
  auto new_last = thrust::unique(thr_label_list.begin(), thr_label_list.end());
  thr_label_list.erase(new_last, thr_label_list.end());
  n_classes_ = new_last - thr_label_list.begin();

  int *d_label_list;
  cudaMallocManaged(&d_label_list, n_classes_ * sizeof(int));
  cudaMemcpy(d_label_list, thrust::raw_pointer_cast(thr_label_list.data()),
             n_classes_ * sizeof(int), cudaMemcpyHostToDevice);

  /* Other initializations */
  cudaMallocManaged(&feature_probs,
                    (n_classes_ * n_features_) * sizeof(float));
  cudaMallocManaged(&class_count_, n_classes_ * sizeof(float));
  // Is the memset below required?
  // cudaMemset(feature_probs, 0, (n_classes_ * n_features_) * sizeof(float));

  /* Calculate frequency of occurence of each term : CalcKernel
  Individual thread for each term. Threads_per_block=1024 */
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(n_features_) / float(threads_per_block.x)));
  BernoulliNBCalcKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_probs, class_count_, train_size, n_classes_,
      n_features_);
  cudaDeviceSynchronize();

  /* Learning Phase: Calculate conditional probabilities */
  float *d_row_sums;
  cudaMallocManaged(&d_row_sums, n_classes_ * sizeof(float));

  /* Find total number of terms in each class */
  for (unsigned int i = 0; i < n_classes_; ++i) {
    thrust::device_vector<float> temp_vec(feature_probs + (n_features_ * i),
                                           feature_probs +
                                               (n_features_ * (i + 1)));
    d_row_sums[i] =
        thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());
  }

  thrust::device_vector<float> temp_vec(d_row_sums, (d_row_sums + n_classes_));
  cout << "class_count_ 0 " << class_count_[0] << " class_count_ 1 "
       << class_count_[1] << endl;
  BernoulliNBLearnKernel<<<blocks_per_grid, threads_per_block>>>(
      feature_probs, class_count_, d_row_sums, train_size, n_classes_,
      n_features_);
  cudaDeviceSynchronize();

  return;
}

int BernoulliNB::predict(vector<float> data, vector<int> labels) {
  std::vector<int>::size_type test_size = labels.size();
  int total_score = 0;

  /* Moving test data to the device */
  float *d_data;
  cudaMallocManaged(&d_data, (n_features_ * test_size) * sizeof(float));
  cudaMemcpy(d_data, &data[0], (n_features_ * test_size) * sizeof(float),
             cudaMemcpyHostToDevice);
  int *d_labels;
  cudaMallocManaged(&d_labels, test_size * sizeof(int));
  cudaMemcpy(d_labels, &labels[0], test_size * sizeof(int),
             cudaMemcpyHostToDevice);

  /* NOTE: The class priors and conditional probabilities should already
  be on the device after train */

  /* Score keeper : 0 or 1 corresponding to each test sample */
  int *score;
  cudaMallocManaged(&score, test_size * sizeof(int));

  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(test_size) / float(threads_per_block.x)));
  MultinomialNBTestKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_probs, class_count_, test_size, n_classes_,
      n_features_, score);
  cudaDeviceSynchronize();

  // Reduce score to a total score using thrust reduction
  thrust::device_vector<int> temp_vec(score, score + test_size);
  total_score =
      thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());

  return total_score;
}

// **********************************************

// ComplementNB

ComplementNB ::ComplementNB() {}

ComplementNB ::~ComplementNB() {}

__global__ void
ComplementNBCalcKernel(const float *d_data, const int *d_labels,
                       float *per_class_feature_sum_, float *per_feature_sum_,
                       unsigned int n_samples_, unsigned int n_features_) {

  // Each thread will take care of one term for all docs
  unsigned int tidx = threadIdx.x;
  unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, row = 0;

  if (feat_col < n_features_) { // End condition check
    // For each document / sample
    for (i = 0; i < n_samples_; ++i) {
      row = d_labels[i];

      // No race condition since each thread deals with one feature only
      // This is embarrasingly parallel
      per_class_feature_sum_[RM_Index(row, feat_col, n_features_)] +=
          d_data[RM_Index(i, feat_col, n_features_)];

      per_feature_sum_[feat_col] += d_data[RM_Index(i, feat_col, n_features_)];
    }
  }
  return;
}

__global__ void ComplementNBLearnKernel(float *feature_weights_,
                                        float *per_class_feature_sum_,
                                        float *per_feature_sum_,
                                        float *per_class_sum_, float all_sum_,
                                        unsigned int n_classes_,
                                        unsigned int n_features_) {
  // Each thread will take one feature
  unsigned int tidx = threadIdx.x;
  int feat_col = tidx + (blockIdx.x * blockDim.x);

  unsigned int i = 0;
  float den_sum = 0;
  float num_sum = 0;

  if (feat_col < n_features_) {        /* Boundary check */
    for (i = 0; i < n_classes_; ++i) { /* For each class */
      den_sum = all_sum_ - per_class_sum_[i];
      num_sum = per_feature_sum_[feat_col] -
                per_class_feature_sum_[RM_Index(i, feat_col, n_features_)];

      feature_weights_[RM_Index(i, feat_col, n_features_)] =log(num_sum + 1.0) -log(den_sum + n_features_);
    }
  }
}

__global__ void ComplementNBNormalizeKernel(float *feature_weights_,
                                            float *per_class_sum_,
                                            unsigned int n_classes_,
                                            unsigned int n_features_) {
  // Each thread will take one feature
  int feat_col = threadIdx.x + (blockIdx.x * blockDim.x);
  unsigned int i = 0;

  if (feat_col < n_features_) {        /* Boundary condition check */
    for (i = 0; i < n_classes_; ++i) { // For each class
      feature_weights_[RM_Index(i, feat_col, n_features_)] /= per_class_sum_[i];
    }
  }
}

__global__ void ComplementNBTestKernel(const float *d_data,
                                       const int *d_labels,
                                       const float *feature_weights_,
                                       int test_size, int n_classes_,
                                       int n_features_, int *score) {
  // Each thread will take one term
  unsigned int sample_num = threadIdx.x + (blockIdx.x * blockDim.x);
  unsigned int i = 0, j = 0;
  float prob_class = 0.0;
  float min = DBL_MAX;
  int result = 0;

  if (sample_num < test_size) { /* Boundary condition check */

    /* Find the poorest complement match */
    for (i = 0; i < n_classes_; ++i) { /* For each class */
      prob_class = 0.0;

      for (j = 0; j < n_features_; ++j) { /* For each feature */
        prob_class += feature_weights_[RM_Index(i, j, n_features_)] *
                      (float)d_data[RM_Index(sample_num, j, n_features_)];
      }

      if (min > prob_class) {
        min = prob_class;
        result = i;
      }
    }

    if (result == d_labels[sample_num]) {
      score[sample_num] = 1;
    } else {
      score[sample_num] = 0;
    }
  }

  return;
}

void ComplementNB::train(vector<float> data, vector<int> labels) {
  unsigned int train_size = labels.size();
  n_features_ = data.size() / train_size;
  unsigned int i = 0;

  // Move data and labels to GPU memory
  // NOTE: Memory Operation, put checks later
  float *d_data;
  cudaMallocManaged(&d_data, (n_features_ * train_size) * sizeof(float));
  cudaMemcpy(d_data, &data[0], (n_features_ * train_size) * sizeof(float),
             cudaMemcpyHostToDevice);

  // NOTE: Memory Operation, put checks later
  int *d_labels;
  cudaMallocManaged(&d_labels, train_size * sizeof(int));
  cudaMemcpy(d_labels, &labels[0], train_size * sizeof(int),
             cudaMemcpyHostToDevice);

  // Use thrust to find unique labels -- declare a thrust label list
  thrust::device_vector<int> thr_label_list(d_labels, d_labels + train_size);
  thrust::sort(thr_label_list.begin(), thr_label_list.end());
  auto new_last = thrust::unique(thr_label_list.begin(), thr_label_list.end());
  thr_label_list.erase(new_last, thr_label_list.end());
  n_classes_ = new_last - thr_label_list.begin();

  int *d_label_list; // List of unique labels
  cudaMallocManaged(&d_label_list, n_classes_ * sizeof(int));
  cudaMemcpy(d_label_list, thrust::raw_pointer_cast(thr_label_list.data()),
             n_classes_ * sizeof(int), cudaMemcpyHostToDevice);

  /* Other initializations */
  cudaMallocManaged(&per_class_feature_sum_,
                    (n_classes_ * n_features_) * sizeof(float));
  cudaMallocManaged(&feature_weights_,
                    (n_classes_ * n_features_) * sizeof(float));
  cudaMallocManaged(&per_feature_sum_, (n_features_) * sizeof(float));
  cudaMallocManaged(&per_class_sum_, (n_classes_) * sizeof(float));

  /* Calculate frequency of occurence of each term : CalcKernel
  Individual thread for each term. Threads_per_block=1024 */
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(n_features_) / float(threads_per_block.x)));
  ComplementNBCalcKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, per_class_feature_sum_, per_feature_sum_, train_size,
      n_features_);

  cudaDeviceSynchronize();

  /* Calculate per class sums.
  Typically we have many more features than classes, so it may not be worth
  launching a separate kernel for it */
  for (i = 0; i < n_classes_; ++i) {
    thrust::device_vector<float> temp_vec(
        per_class_feature_sum_ + (n_features_ * i),
        per_class_feature_sum_ + (n_features_ * (i + 1)));
    per_class_sum_[i] =
        thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());
  }

  /* Find ALL occurences in the dataset */
  thrust::device_vector<float> temp_vec(per_class_sum_,
                                         per_class_sum_ + n_classes_);
  float all_sum_ =
      thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());

  /* Learning Phase: Calculate weights per feature per class */
  ComplementNBLearnKernel<<<blocks_per_grid, threads_per_block>>>(
      feature_weights_, per_class_feature_sum_, per_feature_sum_,
      per_class_sum_, all_sum_, n_classes_, n_features_);

  /* Normalize Phase for stable coefficients.
  Reuse per_class_sum_ variable */
  for (i = 0; i < n_classes_; ++i) {
    thrust::device_vector<float> temp_vec(feature_weights_ + (n_features_ * i),
                                           feature_weights_ +
                                               (n_features_ * (i + 1)));
    per_class_sum_[i] =
        thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());
  }
  cudaDeviceSynchronize();

  /* Somehow normalization is worsening the accuracy */
  // ComplementNBNormalizeKernel<<<blocks_per_grid, threads_per_block>>>(
  //     feature_weights_, per_class_sum_, n_classes_, n_features_);

  return;
}

int ComplementNB::predict(vector<float> data, vector<int> labels) {
  std::vector<int>::size_type test_size = labels.size();
  int total_score = 0;

  /* Moving test data to the device */
  float *d_data;
  cudaMallocManaged(&d_data, (n_features_ * test_size) * sizeof(float));
  cudaMemcpy(d_data, &data[0], (n_features_ * test_size) * sizeof(float),
             cudaMemcpyHostToDevice);
  int *d_labels;
  cudaMallocManaged(&d_labels, test_size * sizeof(int));
  cudaMemcpy(d_labels, &labels[0], test_size * sizeof(int),
             cudaMemcpyHostToDevice);

  /* Score keeper : 0 or 1 corresponding to each test sample */
  int *score;
  cudaMallocManaged(&score, test_size * sizeof(int));

  /* One thread for each test sample */
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(test_size) / float(threads_per_block.x)));
  ComplementNBTestKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_weights_, test_size, n_classes_, n_features_,
      score);

  cudaDeviceSynchronize();

  // Reduce score to a total score using thrust reduction
  vector<int> temp_vec(score, score + test_size);
  total_score = std::accumulate(temp_vec.begin(), temp_vec.end(), 0);

  return total_score;
}
