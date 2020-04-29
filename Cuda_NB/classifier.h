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


class GaussianNB {
public:

	double *p_class_;
	double *f_stats_; // 0 - mean; 1 - var
	double *labels_list_;
	double *f_stats_;
	int *class_count;
	unsigned int features_count_=1;

	GaussianNB();

	virtual ~GaussianNB();

	void train(vector<double> data, vector<int>  labels);

	int predict(vector<double> data, vector<int>  labels);

};

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


class BernoulliNB {

public:
	double *feature_probs;
	vector<int>::size_type n_features_ = 1; // Number of features
	unsigned int n_classes_ = 1; // Number of samples in each class
	double *class_count_; // Class prior probabilities
	
	BernoulliNB();

	virtual ~BernoulliNB();

	void train(vector<double>  data, vector<int>  labels);

	int predict(vector<double> data, vector<int>  labels);

};

class ComplementNB {

public:
	double *feature_probs;
	vector<int>::size_type n_features_ = 1; // Number of features
	unsigned int n_classes_ = 1; // Number of samples in each class
	double *class_count_; // Class prior probabilities
	double *feat_count_;
	double *class_priors;
	double *feature_frequencies_;
	double *acc_feat_sum_;
	int  *all_occur_per_term;
	ComplementNB();

	virtual ~ComplementNB();

	void train(vector<double>  data, vector<int>  labels);

	int predict(vector<double> data, vector<int>  labels);

};

 
#endif
