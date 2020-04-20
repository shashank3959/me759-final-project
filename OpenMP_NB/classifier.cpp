#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <assert.h>
#include <algorithm>
#include "classifier.h"
#include <omp.h>

using namespace std;

GaussianNB::GaussianNB() {

}

GaussianNB::~GaussianNB() {}

void GaussianNB::train(vector<vector<double>> data, vector<int> labels)
{
	map <int, vector<vector<double>>> lfm;
	map <int, int> class_count;
	int train_size = labels.size();
	int i,j;

	labels_list_ = labels;

	features_count_ = int(data[0].size());

	std::vector<int>::size_type lab = 0, l = 0;

	#pragma omp parallel num_threads(1)
	{
		#pragma omp for
		for (lab = 0; lab < labels_list_.size(); lab++) {
			class_count[lab] = 0;
			vector <double> temp(data[0].size(), 0.0);
			f_stats_[lab].push_back(temp);
			f_stats_[lab].push_back(temp);
			f_stats_[lab].push_back(temp);
		}

		#pragma omp for
		for (i = 0; i < train_size; i++) {
			lfm[labels[i]].push_back(data[i]); // x_train per class
			class_count[labels[i]] += 1; // class count
			for (j = 0; j < features_count_; j++) {
				f_stats_[labels[i]][0][j] += data[i][j]; // sum per feature
			}
		}


		#pragma omp for
		for (lab = 0; lab < labels_list_.size(); lab++) {
			for (j = 0; j < features_count_; j++) {
				f_stats_[lab][0][j] /= class_count[lab];
			}
			p_class_[lab] = class_count[lab] * 1.0 / labels.size();

		}

		#pragma omp for
		for (lab = 0; lab < labels_list_.size(); lab++) {
			for (j = 0; j < features_count_; j++) {
				for (l = 0; l < lfm[lab].size(); l++) {
					f_stats_[lab][1][j] += pow(lfm[lab][l][j] - f_stats_[lab][0][j], 2);
				}
				f_stats_[lab][1][j] /= class_count[lab];
				f_stats_[lab][2][j] = 1.0 / sqrt(2 * M_PI * f_stats_[lab][1][j]); // 1/sqrt(2*PI*var)
			}
		}




	}


}

int GaussianNB::predict(vector<vector<double>> X_test, vector<int> Y_test)
{

	map <int, double> p;

	int result = 0;
	int score = 0;
	int i = 0;
	std::vector<int>::size_type lab = 0, k;
	#pragma omp parallel private(lab,k,i) num_threads(1)
	{
		#pragma omp for
		for (k = 0; k < X_test.size(); k++)
		{
			vector<double> vec = X_test[k];

			double max = 0;
			for (lab = 0; lab < labels_list_.size(); lab++) {
				p[lab] = p_class_[lab];
				for (i = 0; i < features_count_; i++) {
					p[lab] *= f_stats_[lab][2][i] * exp(-pow(vec[i] - f_stats_[lab][0][i], 2) / (2 * f_stats_[lab][1][i]));
				}

				if (max < p[lab]) {
					max = p[lab];
					result = lab;
				}
			}

			#pragma omp critical
			if (result == Y_test[k])
			{
				score += 1;
			}
		}
	}

	return score;
}

// ******************************** Multinomial Navie Bayes************************************************

/*
MultionomialGB::MultionomialGB() {

}

MultionomialGB::~MultionomialGB() {}

void MultionomialGB::train(vector<vector<double>> train, vector<int> label)
{

	//calculate multinomial_sums to find mean and variance

	double flag;
	double alpha = 1.0;
	// Decision rule
	//int decision = 1;
	// Verbose
	//int verbose = 0;
	cout << "Inside train " << endl;
	int labels;
		//#pragma omp parallel
	for (auto i = 0; i < train.size(); i++)
	{
		labels = label[i];
		vector<double> values;
		//cout << "lables" << labels << endl;
		for (auto l = 1; l < train[0].size(); l++)
		{
			flag = train[i][l];
			//cout << "flag" << flag << endl;
			values.push_back(flag);
			//cout << "train 0 size" << train[0].size() << endl;
			if (sum_x.find(labels) == sum_x.end()) {
				vector<double> empty;
				for (unsigned int j = 1; j < train[0].size(); j++) {
					empty.push_back(0.0);
				}
				sum_x[labels] = empty;
			}

			sum_x[labels][l - 1] += values[l - 1];
			multinomial_sums[labels] += values[l - 1];
			//cout << values.size() << endl;
			//cout << sum_x[labels].size() << endl;

			data[labels].push_back(values);
			n[labels]++;
			n_total++;

		}
	}

	cout << "sum x cal over" << endl;

	for (auto it = sum_x.begin(); it != sum_x.end(); it++) {

		priors[it->first] = (double)n[it->first] / n_total;

		if (it->first > 0) {
			cout<<"class +%i prior: %1.3f\n" <<it->first<<priors[it->first];
		}
		else {
			cout<<"class %i prior: %1.3f\n"<< it->first<<priors[it->first];
		}
		cout << "feature\tmean\tvar\tstddev\tmnl" << endl;

		// Calculate means
		vector<double> feature_means;
		for (unsigned int i = 0; i < it->second.size(); i++) {
			feature_means.push_back(sum_x[it->first][i] / n[it->first]);
		}

		// Calculate variances
		vector<double> feature_variances(feature_means.size());
		for (unsigned int i = 0; i < data[it->first].size(); i++) {
			for (unsigned int j = 0; j < data[it->first][i].size(); j++) {
				feature_variances[j] += (data[it->first][i][j] - feature_means[j]) * (data[it->first][i][j] - feature_means[j]);
			}
		}
		for (unsigned int i = 0; i < feature_variances.size(); i++) {
			feature_variances[i] /= data[it->first].size();
		}

		// Calculate multinomial likelihoods
		for (unsigned int i = 0; i < feature_means.size(); i++) {
			double mnl = (sum_x[it->first][i] + alpha) / (multinomial_sums[it->first] + (alpha * feature_means.size()));
			//cout << sum_x[it->first][i] << " + 1 / " << multinomial_sums[it->first] << " + " << feature_means.size() << endl;
			multinomial_likelihoods[it->first].push_back(mnl);
		}

		for (unsigned int i = 0; i < feature_means.size(); i++) {
			printf("%i\t%2.3f\t%2.3f\t%2.3f\t%2.3f\n", i + 1, feature_means[i], feature_variances[i], sqrt(feature_variances[i]), multinomial_likelihoods[it->first][i]);
			//cout << feature_means[i] << "\t" << sqrt(feature_variances[i]) << endl;
		}
		means[it->first] = feature_means;
		variances[it->first] = feature_variances;

	}

	cout << "training over" << endl;

}

int MultionomialGB::predict(vector<vector<double>> X_test, vector<int> Y_test)
{

	map <string, double> p;

	int result;
	int score=1;

	cout << "inside prediction" << endl;
	for (auto i = 0; i < X_test.size(); i++)
	{
		vector<double> vec = X_test[i];

		//assert(features_count_ == vec.size());


		double max = 0;


		for (auto it = priors.begin(); it != priors.end(); it++) {
			double numer = priors[it->first];
			for (auto j = 0; j < features_count_; j++) {
				numer *= pow(multinomial_likelihoods[it->first][j], vec[j]);
			}

			if (max < numer) {
				max = numer;
				result = it->first;
			}
		}

				//#pragma omp critical

		if (result==Y_test[i])
		{
			score += 1;
		}
	}

	return score;


}
*/

// ******************************** Bernoulli Naive Bayes ********************/

BernoulliNB::BernoulliNB() {}

BernoulliNB::~BernoulliNB() {}

void BernoulliNB::train(vector<vector<double>> data, vector<int> labels)
{
	int train_size = labels.size();
	n_features_ = data[0].size();

	/* Number of unique labels */
	labels_list_ = labels;
	std::sort(labels_list_.begin(), labels_list_.end());
	auto newEnd = std::unique(labels_list_.begin(), labels_list_.end());
	labels_list_.erase(newEnd, labels_list_.end());

	/* Initializing class variables */
	for (auto lab : labels_list_) {
		class_count_[lab] = 0;
		class_priors_[lab] = 0.0;
		vector <double> temp(data[0].size(), 0.0); /* 1 x n_features */
		feature_probs_[lab] = temp; /* n_labels x n_features */
	}

	/* How many documents contain each term (per label) */
	for (int i = 0; i < train_size; ++i) { /* For each example */
		for (uint j = 0; j < n_features_; ++j) { /* For each feature*/
			// TOFIX: INDIRECTION */
			// TOFIX: Currently this requires labels to be in format 0, 1, 2, 3 ...
			feature_probs_[labels[i]][j] += data[i][j];
		}
		class_count_[labels[i]] += 1;
	}

	/* Convert the frequency to probability */
	for (uint i = 0; i < labels_list_.size(); ++i) {
		// TODO: Use stl transform for this
		for (uint j = 0; j < n_features_; ++j) {
			feature_probs_[labels[i]][j] /= class_count_[i];
		}
		/* Calculate class priors for each class */
		class_priors_[i] = (double)class_count_[i] / train_size;
	}

  return;
}

int BernoulliNB::predict(vector<vector<double>> X_test, vector<int> Y_test) {
	// TODO :n_features_ is actually redundant and SHOULD NOT change from train */
	n_features_ = X_test[0].size();
	std::vector<int>::size_type test_size = Y_test.size();
	int result = 0;
	int score = 0;
	uint i = 0;
	double max = 0.0;
	std::vector<int>::size_type lab = 0, feat = 0;

	// probability for element belonging in a particular class
	map <int, double> prob_class;

	/* For each example in the test set */
	for (i = 0; i < test_size; ++i) {

		vector<double> test_vec = X_test[i];
		max = 0.0;

		/* For each class.
		Note that labels_list_ is populated in the train function */
		for (lab = 0; lab < labels_list_.size(); ++lab) {

			prob_class[lab] = class_priors_[lab];

			/* For each feature */
			for (feat = 0; feat < n_features_; ++feat) {
				prob_class[lab] *= pow(feature_probs_[lab][feat], test_vec[feat]);
				prob_class[lab] *= pow((1 - feature_probs_[lab][feat]),
																											(1- test_vec[feat]));
			}
			if (max < prob_class[lab]) {
				max = prob_class[lab];
				result = lab;
			}
		}
		if (result == Y_test[i]) {
			score += 1;
		}
	}

	return score;
}
