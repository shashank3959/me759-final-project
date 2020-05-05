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

	map <int, float> p_class_;
	map <int, vector<vector<float>>> f_stats_;  // mean and variance 
	vector <int> labels_list_; // label 
	int features_count_=1;

	GaussianNB();

	virtual ~GaussianNB();

	void train(vector<vector<float> > data, vector<int>  labels);

	int predict(vector<vector<float> > data, vector<int>  labels);

};


class BernoulliNB {

public:
	map <int, vector<float>> feature_probs_;
	std::vector<int>::size_type n_features_ = 1; // Number of features
	map <int, int> class_count_; // Number of samples in each class
	map <int, float> class_priors_; // Class prior probabilities
	vector <int> labels_list_; // List of unique labels

	BernoulliNB();

	virtual ~BernoulliNB();

	void train(vector<vector<float> > data, vector<int>  labels);

	int predict(vector<vector<float> > data, vector<int>  labels);

};

class MultinomialNB {

public:
	map <int, vector<float>> feature_probs_;
	std::vector<int>::size_type n_features_ = 1; // Number of features
	map <int, int> class_count_; // Number of samples in each class
	map <int, float> class_priors_; // Class prior probabilities
	vector <int> labels_list_; // List of unique labels
	map <int, int> feat_count_; // Total feats/words in each class

	MultinomialNB();

	virtual ~MultinomialNB();

	void train(vector<vector<float> > data, vector<int>  labels);

	int predict(vector<vector<float> > data, vector<int>  labels);

};


class ComplementNB {

public:
	map <int, vector<int>> feature_frequencies_; // Feature frequencies for each class
	map <int, vector<double>> feature_weights_; // Feature weights for each class
	vector <int> all_occur_per_term; // Total occurences of a term in whole dataset
	int all_occur; // Total count of all occurences of all words
	std::vector<int>::size_type n_features_ = 1; // Number of features
	map <int, int> class_count_; // Number of samples in each class
	map <int, float> class_priors_; // Class prior probabilities
	vector <int> labels_list_; // List of unique labels
	map <int, int> feat_count_; // Total feats/words in each class


	ComplementNB();

	virtual ~ComplementNB();

	void train(vector<vector<float> > data, vector<int>  labels);

	int predict(vector<vector<float> > data, vector<int>  labels);

};

#endif
