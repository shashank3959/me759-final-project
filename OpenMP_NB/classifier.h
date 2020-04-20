
#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>

using namespace std;

class GaussianNB {
public:

	map <int, double> p_class_;
	map <int, vector<vector<double>>> f_stats_; // 0 - mean; 1 - var
	vector <int> labels_list_;
	int features_count_=1;

	GaussianNB();

	virtual ~GaussianNB();

	void train(vector<vector<double> > data, vector<int>  labels);

	int predict(vector<vector<double> > data, vector<int>  labels);

};


class MultionomialGB {
public:

	unsigned int n_total = 0;
	map<int, vector<vector<double> > > data;
	map<int, int> n;
	map<int, double> priors;
	map<int, vector<double> > multinomial_likelihoods;
	map<int, int> multinomial_sums;
	map<int, vector<double> > sum_x;
	map<int, vector<double> > means;
	map<int, vector<double> > variances;

	int features_count_=1;

	MultionomialGB();


	virtual ~MultionomialGB();

	void train(vector<vector<double> > data, vector<int>  labels);

	int predict(vector<vector<double> > data, vector<int>  labels);

};

class BernoulliNB {

public:
	map <int, vector<double>> feature_probs_;
	std::vector<int>::size_type n_features_ = 1; // Number of features
	map <int, int> class_count_; // Number of samples in each class
	map <int, double> class_priors_; // Class prior probabilities
	vector <int> labels_list_; // List of unique labels

	BernoulliNB();

	virtual ~BernoulliNB();

	void train(vector<vector<double> > data, vector<int>  labels);

	int predict(vector<vector<double> > data, vector<int>  labels);

};

class MultinomialNB {

public:
	map <int, vector<double>> feature_probs_;
	std::vector<int>::size_type n_features_ = 1; // Number of features
	map <int, int> class_count_; // Number of samples in each class
	map <int, double> class_priors_; // Class prior probabilities
	vector <int> labels_list_; // List of unique labels
	map <int, int> feat_count_; // Total feats/words in each class

	MultinomialNB();

	virtual ~MultinomialNB();

	void train(vector<vector<double> > data, vector<int>  labels);

	int predict(vector<vector<double> > data, vector<int>  labels);

};

#endif
