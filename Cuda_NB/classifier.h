#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <stdio.h>

using namespace std;

/* Configs */
#define THREADS_PER_BLOCK 1024

/* Utilities */
#include <thrust/sort.h>
#include <thrust/reduce.h>

using thrust::device_vector;


class MultinomialNB {

public:
  vector<int>::size_type n_features_ = 1; // Number of features
  unsigned int n_classes_ = 1; // Number of unique labels

  double *feature_probs; // shape: n_classes_ * n_features_
  double *class_priors; // shape: n_classes_

	MultinomialNB();

	virtual ~MultinomialNB();

	void train(vector<double> data, vector<int> labels);

	int predict(vector<double> data, vector<int>  labels);

};


#endif
