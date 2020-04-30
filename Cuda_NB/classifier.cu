#include "classifier.h"
#include <algorithm>
#include <assert.h>
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

__global__ void GaussianNBCalcKernel(const double *d_data, const int *d_labels,
                                     double *f_stats_, int *class_count,
                                     unsigned int n_samples_,
                                     unsigned int n_classes_,
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
      f_stats_[RM_Index(row, feat_col, n_features_)] +=
          d_data[RM_Index(i, feat_col, n_features_)];

      // WARNING: thread divergence :(
      if (feat_col == 0) {
        class_count[row] += 1;
      }
    }
  }
  return;
}

/* Kernel divides each row by a class count and takes pow */
__global__ void
GaussianNBLearnKernel(double *feature_probs, double *f_stats_, int *class_count,
                      const double *d_row_sums, unsigned int n_samples_,
                      unsigned int n_classes_, unsigned int n_features_) {

  /* Each thread will take one term */
  unsigned int tidx = threadIdx.x;
  unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0;

  if (feat_col < n_features_) { /* End condition check */
    /* For each label */
    for (i = 0; i < n_classes_; ++i) {
      // TODO: Add Laplace Smoothing
                                f_stats_[RM_Index(i, feat_col, n_features_)] /= class_count[i]);

                                if (feat_col == 0) {
                                  feature_probs[i] =
                                      class_count[i] * 1 / (double)(n_samples_);
                                }
    }
  }
}

// kernel to find prob

__global__ void GaussianNBProbKernel(double *feature_probs,
                                     double *class_priors, const double *d_data,
                                     unsigned int n_samples_,
                                     unsigned int n_classes_,
                                     unsigned int n_features_) {

  /* Each thread will take one term */
  unsigned int tidx = threadIdx.x;
  unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0;

  if (feat_col < n_features_) { /* End condition check */
    /* For each label */
    for (i = 0; i < n_classes_; ++i) {
      // TODO: Add Laplace Smoothing + check indexing once
      feature_probs[RM_Index(i, feat_col, n_features_)] +=
          pow(lfm[RM_Index(i, feat_col, n_features_)] -
                  f_stats_[RM_Index(row, feat_col, n_features_)],
              2);

      if (feat_col == 0) {
        // TODO: Change index of f_stats_ or introduce new variable for
        // 1/(sqrt() term to avoid indexing issue
        f_stats_[RM_Index(i, feat_col, n_features_)] /= class_count[i];
        f_stats_[RM_Index(i, feat_col, n_features_)] =
            1.0 /
            sqrt(2 * M_PI * feature_probs[RM_Index(i, feat_col, n_features_)]);
      }
    }
  }
}

__global__ void GaussianNBTestKernel(const double *d_data, const int *d_labels,
                                     const double *feature_probs,
                                     const double *class_priors, int test_size,
                                     int n_classes_, int n_features_,
                                     int *score) {
  /* Each thread will take one term */
  unsigned int tidx = threadIdx.x;
  unsigned int sample_num = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, j = 0;
  double prob_class = 0;
  int max = 0;
  int result = 0;

  if (sample_num < test_size) {
    for (i = 0; i < n_classes_; ++i) { /* For each class */
      prob_class = class_priors[i];

      for (j = 0; j < n_features_; ++j) { /* For each feature */
        // reference: p[lab] *= f_stats_[lab][2][i] * exp(-pow(vec[i] -
        // f_stats_[lab][0][i], 2) / (2 * f_stats_[lab][1][i]));
        // TODO: change indexing of f_stats
                                        prob_class *= (f_stats_[RM_Index(sample_num, j, n_features_)] *exp(-pow(d_data[RM_Index(sample_num, j, n_features_)] - \
						       feature_probs[RM_Index(i, j, n_features_)],2) / f_stats_[RM_Index(sample_num, j, n_features_)]
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

void GaussianNB::train(vector<double> data, vector<int> labels) {
  unsigned int train_size = labels.size();
  n_features_ = data.size() / train_size;

  // Move data and labels to GPU memory
  // NOTE: Memory Operation, put checks later
  double *d_data;
  cudaMallocManaged(&d_data, (n_features_ * train_size) * sizeof(double));
  cudaMemcpy(d_data, &data[0], (n_features_ * train_size) * sizeof(double),
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
  cudaMallocManaged(&f_stats_, (n_classes_ * n_features_) * sizeof(double));
  cudaMallocManaged(&feature_probs, n_classes_ * sizeof(double));
  cudaMallocManaged(&class_count, n_classes_ * sizeof(int));
  // Is the memset below required?
  // cudaMemset(feature_probs, 0, (n_classes_ * n_features_) * sizeof(double));

  /* Calculate frequency of occurence of each term : CalcKernel
  Individual thread for each term. Threads_per_block=1024 */
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(n_features_) / float(threads_per_block.x)));
  GaussianNBCalcKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, f_stats_, class_count, train_size, n_classes_,
      n_features_);
  cudaDeviceSynchronize();

  /* Learning Phase: Calculate conditional probabilities */
  double *lfm;
  cudaMallocManaged(&lfm, n_classes_ * sizeof(double));

  /* Find total number of terms in each class */
  for (unsigned int i = 0; i < n_classes_; ++i) {
    thrust::device_vector<double> temp_vec(f_stats_ + (n_features_ * i),
                                           f_stats_ + (n_features_ * (i + 1)));
    lfm[i] = thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());
  }

  thrust::device_vector<double> temp_vec(d_row_sums, (d_row_sums + n_classes_));

  GaussianNBLearnKernel<<<blocks_per_grid, threads_per_block>>>(
      f_stats_, feature_probs, class_count, lfm, train_size, n_classes_,
      n_features_);
  cudaDeviceSynchronize();

  // prob kernel

  GaussianNBProbKernel<<<blocks_per_grid, threads_per_block>>>(
      feature_probs, class_priors, *d_data, n_samples_, n_classes_, n_features_)

      cudaDeviceSynchronize();

  return;
}

int GaussianNB::predict(vector<double> data, vector<int> labels) {
  std::vector<int>::size_type test_size = labels.size();
  int total_score = 0;

  /* Moving test data to the device */
  double *d_data;
  cudaMallocManaged(&d_data, (n_features_ * test_size) * sizeof(double));
  cudaMemcpy(d_data, &data[0], (n_features_ * test_size) * sizeof(double),
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
  GaussianNBTestKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_probs, class_priors, test_size, n_classes_,
      n_features_, score);

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
MultinomialNBCalcKernel(const double *d_data, const int *d_labels,
                        double *feature_probs, double *class_priors,
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
MultinomialNBLearnKernel(double *feature_probs, double *class_priors,
                         const double *d_row_sums, unsigned int n_samples_,
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
        class_priors[i] = log(class_priors[i] / (double)n_samples_);
        printf("prior at %u = %lf\n", i, class_priors[i]);
      }
    }
  }
}

__global__ void MultinomialNBTestKernel(const double *d_data,
                                        const int *d_labels,
                                        const double *feature_probs,
                                        const double *class_priors,
                                        int test_size, int n_classes_,
                                        int n_features_, int *score) {
  /* Each thread will take one term */
  unsigned int tidx = threadIdx.x;
  unsigned int sample_num = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, j = 0;
  double prob_class = 0;
  int max = 0;
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

void MultinomialNB::train(vector<double> data, vector<int> labels) {
  unsigned int train_size = labels.size();
  n_features_ = data.size() / train_size;

  // Move data and labels to GPU memory
  // NOTE: Memory Operation, put checks later
  double *d_data;
  cudaMallocManaged(&d_data, (n_features_ * train_size) * sizeof(double));
  cudaMemcpy(d_data, &data[0], (n_features_ * train_size) * sizeof(double),
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
                    (n_classes_ * n_features_) * sizeof(double));
  cudaMallocManaged(&class_priors, n_classes_ * sizeof(double));
  // Is the memset below required?
  // cudaMemset(feature_probs, 0, (n_classes_ * n_features_) * sizeof(double));

  /* Calculate frequency of occurence of each term : CalcKernel
  Individual thread for each term. Threads_per_block=1024 */
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(n_features_) / float(threads_per_block.x)));
  MultinomialNBCalcKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_probs, class_priors, train_size, n_classes_,
      n_features_);
  cudaDeviceSynchronize();

  /* Learning Phase: Calculate conditional probabilities */
  double *d_row_sums;
  cudaMallocManaged(&d_row_sums, n_classes_ * sizeof(double));

  /* Find total number of terms in each class */
  for (unsigned int i = 0; i < n_classes_; ++i) {
    thrust::device_vector<double> temp_vec(feature_probs + (n_features_ * i),
                                           feature_probs +
                                               (n_features_ * (i + 1)));
    d_row_sums[i] =
        thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());
  }

  thrust::device_vector<double> temp_vec(d_row_sums, (d_row_sums + n_classes_));

  MultinomialNBLearnKernel<<<blocks_per_grid, threads_per_block>>>(
      feature_probs, class_priors, d_row_sums, train_size, n_classes_,
      n_features_);
  cudaDeviceSynchronize();

  return;
}

int MultinomialNB::predict(vector<double> data, vector<int> labels) {
  std::vector<int>::size_type test_size = labels.size();
  int total_score = 0;

  /* Moving test data to the device */
  double *d_data;
  cudaMallocManaged(&d_data, (n_features_ * test_size) * sizeof(double));
  cudaMemcpy(d_data, &data[0], (n_features_ * test_size) * sizeof(double),
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

__global__ void BernoulliNBCalcKernel(const double *d_data, const int *d_labels,
                                      double *feature_probs,
                                      double *class_count_,
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
BernoulliNBLearnKernel(double *feature_probs, double *class_count_,
                       const double *d_row_sums, unsigned int n_samples_,
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
        class_count_[i] = class_count_[i] / (double)n_samples_;
        printf("prior at %u = %lf\n", i, class_count_[i]);
      }
    }
  }
}

__global__ void BernoulliNBTestKernel(const double *d_data, const int *d_labels,
                                      const double *feature_probs,
                                      const double *class_count_, int test_size,
                                      int n_classes_, int n_features_,
                                      int *score) {
  // Each thread will take one term
  unsigned int tidx = threadIdx.x;
  unsigned int sample_num = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, j = 0;
  double prob_class = 0;
  int max = 0;
  int result = 0;

  if (sample_num < test_size) {
    for (i = 0; i < n_classes_; ++i) { // For each class
      prob_class = class_count_[i];

      for (j = 0; j < n_features_; ++j) { // For each feature
        prob_class *= pow(feature_probs[RM_Index(i, j, n_features_)],
                          d_data[RM_Index(sample_num, j, n_features_)]);
        prob_class *= pow((1 - feature_probs[RM_Index(i, j, n_features_)]),
                          (1 - d_data[RM_Index(sample_num, j, n_features_)]));
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

BernoulliNB::BernoulliNB() {}

BernoulliNB::~BernoulliNB() {}

void BernoulliNB::train(vector<double> data, vector<int> labels) {
  unsigned int train_size = labels.size();
  n_features_ = data.size() / train_size;

  // Move data and labels to GPU memory
  // NOTE: Memory Operation, put checks later
  double *d_data;
  cudaMallocManaged(&d_data, (n_features_ * train_size) * sizeof(double));
  cudaMemcpy(d_data, &data[0], (n_features_ * train_size) * sizeof(double),
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
                    (n_classes_ * n_features_) * sizeof(double));
  cudaMallocManaged(&class_count_, n_classes_ * sizeof(double));
  // Is the memset below required?
  // cudaMemset(feature_probs, 0, (n_classes_ * n_features_) * sizeof(double));

  /* Calculate frequency of occurence of each term : CalcKernel
  Individual thread for each term. Threads_per_block=1024 */
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(n_features_) / float(threads_per_block.x)));
  BernoulliNBCalcKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_probs, class_count_, train_size, n_classes_,
      n_features_);
  cudaDeviceSynchronize();

  /* Learning Phase: Calculate conditional probabilities */
  double *d_row_sums;
  cudaMallocManaged(&d_row_sums, n_classes_ * sizeof(double));

  /* Find total number of terms in each class */
  for (unsigned int i = 0; i < n_classes_; ++i) {
    thrust::device_vector<double> temp_vec(feature_probs + (n_features_ * i),
                                           feature_probs +
                                               (n_features_ * (i + 1)));
    d_row_sums[i] =
        thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());
  }

  thrust::device_vector<double> temp_vec(d_row_sums, (d_row_sums + n_classes_));
  cout << "class_count_ 0 " << class_count_[0] << " class_count_ 1 "
       << class_count_[1] << endl;
  BernoulliNBLearnKernel<<<blocks_per_grid, threads_per_block>>>(
      feature_probs, class_count_, d_row_sums, train_size, n_classes_,
      n_features_);
  cudaDeviceSynchronize();

  return;
}

int BernoulliNB::predict(vector<double> data, vector<int> labels) {
  std::vector<int>::size_type test_size = labels.size();
  int total_score = 0;

  /* Moving test data to the device */
  double *d_data;
  cudaMallocManaged(&d_data, (n_features_ * test_size) * sizeof(double));
  cudaMemcpy(d_data, &data[0], (n_features_ * test_size) * sizeof(double),
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
ComplementNBCalcKernel(const double *d_data, const int *d_labels,
                       double *feature_probs, double *class_count_,
                       unsigned int n_samples_, unsigned int n_classes_,
                       unsigned int n_features_, int *all_occur_per_term) {

  // Each thread will take care of one term for all docs
  unsigned int tidx = threadIdx.x;
  unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, row = 0;
  std::printf("inside cal \n");
  if (feat_col < n_features_) { // End condition check

    // For each document / sample
    for (i = 0; i < n_samples_; ++i) {
      row = d_labels[i];

      // No race condition since each thread deals with one feature only
      feature_probs[RM_Index(row, feat_col, n_features_)] +=
          d_data[RM_Index(i, feat_col, n_features_)];
      all_occur_per_term[feat_col] +=
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
ComplementNBLearnKernel(double *feature_probs, double *d_row_sums,
                        double *class_count_, int *all_occur_per_term,
                        double all_occur, unsigned int n_samples_,
                        unsigned int n_classes_, unsigned int n_features_) {

  // Each thread will take one term
  unsigned int tidx = threadIdx.x;
  int feat_col = tidx + (blockIdx.x * blockDim.x);
  std::printf(" feat_col \n ");
  unsigned int i = 0;
  double den_sum = 0;
  double num_sum = 0;
  if (feat_col < n_classes_) { // for each class
    std::printf(" feat_col %d \n", feat_col);
    for (i = 0; i < n_features_; ++i) { // For each feature

      if (i == 0) {
        den_sum = all_occur - d_row_sums[feat_col];
      }

      num_sum = all_occur_per_term[i] -
                feature_probs[RM_Index(i, feat_col, n_features_)];
      feature_probs[RM_Index(i, feat_col, n_features_)] =
          log(((double)num_sum + 1) / ((double)den_sum + 1));
    }
  }
}

// kernel to normalize feature weights

__global__ void ComplementNBNormalizeKernel(double *feature_probs,
                                            double *d_row_sums,
                                            unsigned int n_classes_,
                                            unsigned int n_features_) {

  // Each thread will take one term
  unsigned int tidx = threadIdx.x;
  int feat_col = tidx + (blockIdx.x * blockDim.x);
  std::printf(" feat_col \n ");
  unsigned int i = 0;
  double den_sum = 0;

  if (feat_col < n_classes_) { // for each class
    std::printf(" feat_col %d \n", feat_col);
    for (i = 0; i < n_features_; ++i) { // For each feature

      if (i == 0) {
        den_sum = d_row_sums[feat_col];
      }

      feature_probs[RM_Index(i, feat_col, n_features_)] /= den_sum;
    }
  }
}

__global__ void ComplementNBTestKernel(const double *d_data,
                                       const int *d_labels,
                                       const double *feature_probs,
                                       const double *class_count_,
                                       int test_size, int n_classes_,
                                       int n_features_, int *score) {
  // Each thread will take one term
  unsigned int tidx = threadIdx.x;
  unsigned int sample_num = tidx + (blockIdx.x * blockDim.x);
  unsigned int i = 0, j = 0;
  double prob_class = 0;
  double min = 0;
  int result = 0;

  if (sample_num < test_size) {
    for (i = 0; i < n_classes_; ++i) { // For each class
      prob_class = 0.0;

      for (j = 0; j < n_features_; ++j) { // For each feature
        prob_class += pow(feature_probs[RM_Index(i, j, n_features_)],
                          d_data[RM_Index(sample_num, j, n_features_)]);
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

void ComplementNB::train(vector<double> data, vector<int> labels) {
  unsigned int train_size = labels.size();
  n_features_ = data.size() / train_size;

  // Move data and labels to GPU memory
  // NOTE: Memory Operation, put checks later
  double *d_data;
  cudaMallocManaged(&d_data, (n_features_ * train_size) * sizeof(double));
  cudaMemcpy(d_data, &data[0], (n_features_ * train_size) * sizeof(double),
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
  // cudaMallocManaged(&feature_frequencies_, (n_classes_ * n_features_) *
  // sizeof(double));
  cudaMallocManaged(&feature_probs,
                    (n_classes_ * n_features_) * sizeof(double));
  cudaMallocManaged(&all_occur_per_term, (n_features_) * sizeof(int));
  // cudaMallocManaged(&acc_feat_sum_, n_classes_ * sizeof(double));

  cout << "all set " << endl;

  // Is the memset below required?
  // cudaMemset(all_occur_per_term, 0, (n_features_) * sizeof(int));

  /* Calculate frequency of occurence of each term : CalcKernel
  Individual thread for each term. Threads_per_block=1024 */
  dim3 threads_per_block(THREADS_PER_BLOCK);
  dim3 blocks_per_grid(ceil(float(n_features_) / float(threads_per_block.x)));
  ComplementNBCalcKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_probs, class_priors, train_size, n_classes_,
      n_features_, all_occur_per_term);
  cudaDeviceSynchronize();

  /* Learning Phase: Calculate conditional probabilities */
  double *d_row_sums;
  cudaMallocManaged(&d_row_sums, n_classes_ * sizeof(double));
  cout << "sum up " << endl;
  /* Find total number of terms in each class */
  cout << "n_classes_ " << n_classes_ << endl;

  for (unsigned int i = 0; i < n_classes_; i++) {
    thrust::device_vector<double> temp_vec(feature_probs + (n_features_ * i),
                                           feature_probs +
                                               (n_features_ * (i + 1)));
    d_row_sums[i] =
        thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());
  }
  cout << "accim" << endl;

  // vector<double> temp_vec(d_row_sums, (d_row_sums + n_classes_));
  thrust::device_vector<double> temp_vec(d_row_sums, (d_row_sums + n_classes_));
  double all_occur;
  all_occur = (double)accumulate(all_occur_per_term,
                                 all_occur_per_term + n_features_, 0);
  cout << " all occur " << all_occur << endl;

  cout << "n_classes_ " << n_classes_ << "fb sdfhjdfjs " << endl;
  ComplementNBLearnKernel<<<blocks_per_grid, threads_per_block>>>(
      feature_probs, d_row_sums, class_priors, all_occur_per_term, all_occur,
      train_size, n_classes_, n_features_);
  cudaDeviceSynchronize();

  // normalize weights

  ComplementNBNormalizeKernel<<<blocks_per_grid, threads_per_block>>>(
      feature_probs, d_row_sums, n_classes_, n_features_);
  cudaDeviceSynchronize();

  cout << "over " << n_classes_ << "fb sdfhjdfjs " << endl;

  return;
}

int ComplementNB::predict(vector<double> data, vector<int> labels) {

  cout << "in test" << endl;
  std::vector<int>::size_type test_size = labels.size();
  int total_score = 0;

  /* Moving test data to the device */
  double *d_data;
  cudaMallocManaged(&d_data, (n_features_ * test_size) * sizeof(double));
  cudaMemcpy(d_data, &data[0], (n_features_ * test_size) * sizeof(double),
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
  cout << "callling test ker" << endl;
  ComplementNBTestKernel<<<blocks_per_grid, threads_per_block>>>(
      d_data, d_labels, feature_probs, class_priors, test_size, n_classes_,
      n_features_, score);

  // Reduce score to a total score using thrust reduction
  vector<int> temp_vec(score, score + test_size);
  total_score = std::accumulate(temp_vec.begin(), temp_vec.end(), 0);

  return total_score;
}
