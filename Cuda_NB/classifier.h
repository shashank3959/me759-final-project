#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>

// Thrust headers
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>



using namespace std;
using thrust::host_vector;
using thrust::device_vector;
using thrust::device_ptr;

/* Utilities */


class MultinomialNB {

public:
  vector<int>::size_type n_features_ = 1; // Number of features
  vector <int> labels_list_; // List of unique labels

	map <int, vector<double>> feature_probs_;
	map <int, int> class_count_; // Number of samples in each class
	map <int, double> class_priors_; // Class prior probabilities

	map <int, int> feat_count_; // Total feats/words in each class

	MultinomialNB();

	virtual ~MultinomialNB();

	void train(vector<double> data, vector<int> labels);

	int predict(vector<double> data, vector<int>  labels);

};


#endif
