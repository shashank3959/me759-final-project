#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <vector>

using namespace std;

/* Configs */
#define THREADS_PER_BLOCK 1024

/* Utilities */
#include <thrust/reduce.h>
#include <thrust/sort.h>

class GaussianNB {
public:
  unsigned int n_features_ = 1; // Number of features
  unsigned int n_classes_ = 1;  // Number of unique labels
  double *feature_means_;       // Per feature per class means
  double *feature_vars_;        // Per feature per class variances
  double *class_priors_;        // shape: n_classes_
  int *class_count_;            // Number of items per class; shape: n_classes_

  GaussianNB();

  virtual ~GaussianNB();

  void train(vector<double> data, vector<int> labels);

  int predict(vector<double> data, vector<int> labels);
};

class MultinomialNB {

public:
  vector<int>::size_type n_features_ = 1; // Number of features
  unsigned int n_classes_ = 1;            // Number of unique labels

  double *feature_probs; // shape: n_classes_ * n_features_
  double *class_priors;  // shape: n_classes_

  MultinomialNB();

  virtual ~MultinomialNB();

  void train(vector<double> data, vector<int> labels);

  int predict(vector<double> data, vector<int> labels);
};

class BernoulliNB {

public:
  double *feature_probs;
  vector<int>::size_type n_features_ = 1; // Number of features
  unsigned int n_classes_ = 1;            // Number of samples in each class
  double *class_count_;                   // Class prior probabilities

  BernoulliNB();

  virtual ~BernoulliNB();

  void train(vector<double> data, vector<int> labels);

  int predict(vector<double> data, vector<int> labels);
};

class ComplementNB {

public:
  vector<int>::size_type n_features_ = 1; // Number of features
  unsigned int n_classes_ = 1;            // Number of samples in each class
  double *feature_weights_; // Learned weights of the model; n_features_ * n_classes_
  double *per_class_feature_sum_; // n_features_ * n_classes_
  double *per_feature_sum_;
  double *per_class_sum_;
  double all_sum_;

  ComplementNB();

  virtual ~ComplementNB();

  void train(vector<double> data, vector<int> labels);

  int predict(vector<double> data, vector<int> labels);
};

#endif
