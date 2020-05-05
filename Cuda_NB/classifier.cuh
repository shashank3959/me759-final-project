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
  float *feature_means_;       // Per feature per class means
  float *feature_vars_;        // Per feature per class variances
  float *class_priors_;        // shape: n_classes_
  int *class_count_;            // Number of items per class; shape: n_classes_

  GaussianNB();

  virtual ~GaussianNB();

  void train(vector<float> data, vector<int> labels);

  int predict(vector<float> data, vector<int> labels);
};

class MultinomialNB {

public:
  vector<int>::size_type n_features_ = 1; // Number of features
  unsigned int n_classes_ = 1;            // Number of unique labels

  float *feature_probs; // shape: n_classes_ * n_features_
  float *class_priors;  // shape: n_classes_

  MultinomialNB();

  virtual ~MultinomialNB();

  void train(vector<float> data, vector<int> labels);

  int predict(vector<float> data, vector<int> labels);
};

class BernoulliNB {

public:
  float *feature_probs;
  vector<int>::size_type n_features_ = 1; // Number of features
  unsigned int n_classes_ = 1;            // Number of samples in each class
  float *class_count_;                   // Class prior probabilities

  BernoulliNB();

  virtual ~BernoulliNB();

  void train(vector<float> data, vector<int> labels);

  int predict(vector<float> data, vector<int> labels);
};

class ComplementNB {

public:
  vector<int>::size_type n_features_ = 1; // Number of features
  unsigned int n_classes_ = 1;            // Number of samples in each class
  float *feature_weights_; // Learned weights of the model; n_features_ * n_classes_
  float *per_class_feature_sum_; // n_features_ * n_classes_
  float *per_feature_sum_;
  float *per_class_sum_;
  float all_sum_;

  ComplementNB();

  virtual ~ComplementNB();

  void train(vector<float> data, vector<int> labels);

  int predict(vector<float> data, vector<int> labels);
};

#endif
