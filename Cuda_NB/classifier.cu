#include "classifier.h"

// Row major index
__device__
inline unsigned int RM_Index(unsigned int row, unsigned int col, unsigned int width) {
	return (row*width + col);
}

MultinomialNB::MultinomialNB() {}

MultinomialNB::~MultinomialNB() {}


__global__ void CalcKernel(const double *d_data, const int *d_labels,
	double *feature_probs, double *class_priors, unsigned int n_samples_, unsigned int n_classes_,
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
				feature_probs[RM_Index(row, feat_col, n_features_)] += d_data[RM_Index(i, feat_col, n_features_)];

				// WARNING: thread divergence :(
				if (feat_col == 0) {
					class_priors[row] += 1;
				}
			}
		}
		return;

}

/* Kernel divides each row by a number and takes log */
__global__ void LearnKernel(double *feature_probs, double *class_priors,
	const double *d_row_sums, unsigned int n_samples_, unsigned int n_classes_,
	unsigned int n_features_) {

		/* Each thread will take one term */
		unsigned int tidx = threadIdx.x;
		unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
		unsigned int i = 0;

		if (feat_col < n_features_) { /* End condition check */
			/* For each label */
			for (i = 0; i < n_classes_; ++i) {
				feature_probs[RM_Index(i, feat_col, n_features_)] =
				log (feature_probs[RM_Index(i, feat_col, n_features_)] / d_row_sums[i]);

				if (feat_col == 0) {
					class_priors[i] = log(class_priors[i] / (double)n_samples_);
					printf("prior at %u = %lf\n", i, class_priors[i]);
				}
			}
		}
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
	device_vector<int> thr_label_list(d_labels, d_labels + train_size);
	thrust::sort(	thr_label_list.begin(), thr_label_list.end());
	auto new_last = thrust::unique(thr_label_list.begin(),
																	thr_label_list.end());
	thr_label_list.erase(new_last, thr_label_list.end());
	n_classes_ = new_last - thr_label_list.begin();

	int *d_label_list;
	cudaMallocManaged(&d_label_list, n_classes_ * sizeof(int));
	cudaMemcpy(d_label_list, thrust::raw_pointer_cast(thr_label_list.data()),
															n_classes_ * sizeof(int), cudaMemcpyHostToDevice);

	/* Other initializations */
	cudaMallocManaged(&feature_probs, (n_classes_ * n_features_) * sizeof(double));
	cudaMallocManaged(&class_priors, n_classes_ * sizeof(double));
	// Is the memset below required?
	// cudaMemset(feature_probs, 0, (n_classes_ * n_features_) * sizeof(double));

	/* Calculate frequency of occurence of each term : CalcKernel
	Individual thread for each term. Threads_per_block=1024 */
	dim3 threads_per_block(THREADS_PER_BLOCK);
	dim3 blocks_per_grid(ceil(float(n_features_) / float(threads_per_block.x)));
	CalcKernel<<<blocks_per_grid, threads_per_block>>>(d_data, d_labels,
		feature_probs, class_priors, train_size, n_classes_, n_features_);

	/* Learning Phase: Calculate conditional probabilities */
	double *d_row_sums;
	cudaMallocManaged(&d_row_sums, n_classes_ * sizeof(double));

	/* Find total number of terms in each class */
	for (unsigned int i = 0; i < n_classes_; ++i) {
		thrust::device_vector<double> temp_vec(feature_probs + (n_features_*i),
																					feature_probs + (n_features_*(i+1)));
		d_row_sums[i] = thrust::reduce(thrust::device, temp_vec.begin(), temp_vec.end());
		}

	thrust::device_vector<double> temp_vec(d_row_sums, (d_row_sums + n_classes_));

	LearnKernel<<<blocks_per_grid, threads_per_block>>>(feature_probs, class_priors,
		d_row_sums, train_size, n_classes_, n_features_);

	}


int MultinomialNB::predict(vector<double> data, vector<int>  labels) {
	return 420;
}
