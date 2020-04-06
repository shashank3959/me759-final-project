#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <assert.h>
#include <algorithm>
#include "classifier.h"
using namespace std;
#define M_PI 3.14159265359

GaussianNB::GaussianNB() {

}

GaussianNB::~GaussianNB() {}

void GaussianNB::train(vector<vector<double>> data, vector<int> labels)
{
	map <int, vector<vector<double>>> lfm;
	map <int, int> class_count;
	int train_size = labels.size();

	// assign get unique labels to labels_list
	labels_list_ = labels;
	std::sort(labels_list_.begin(), labels_list_.end());
	std::vector<string>::iterator newEnd;
	// Override duplicate elements
	//newEnd = std::unique(labels_list_.begin(), labels_list_.end());
	//labels_list_.erase(newEnd, labels_list_.end());

	features_count_ = data[0].size();

	// init mean and var
	for (auto lab : labels_list_) {
		class_count[lab] = 0;
		vector <double> temp(data[0].size(), 0.0);
		f_stats_[lab].push_back(temp);
		f_stats_[lab].push_back(temp);
		f_stats_[lab].push_back(temp);
	}

	//gathering data list per class; count classes; sum per class
	for (auto i = 0; i < train_size; i++) {
		lfm[labels[i]].push_back(data[i]); // x_train per class
		class_count[labels[i]] += 1; // class count
		for (auto j = 0; j < features_count_; j++) {
			f_stats_[labels[i]][0][j] += data[i][j]; // sum per feature
		}
	}

	// transforming f_stats 0 sum into mean
	for (auto lab : labels_list_) {
		for (auto j = 0; j < features_count_; j++) {
			f_stats_[lab][0][j] /= class_count[lab];
		}
		p_class_[lab] = class_count[lab] * 1.0 / labels.size();
	}

	// calc var by classes
	for (auto lab : labels_list_) {
		for (auto j = 0; j < features_count_; j++) {
			for (auto i = 0; i < lfm[lab].size(); i++) {
				f_stats_[lab][1][j] += pow(lfm[lab][i][j] - f_stats_[lab][0][j], 2);
			}
			f_stats_[lab][1][j] /= class_count[lab];
			f_stats_[lab][2][j] = 1.0 / sqrt(2 * M_PI * f_stats_[lab][1][j]); // 1/sqrt(2*PI*var)
		}
	}

}

int GaussianNB::predict(vector<vector<double>> X_test, vector<int> Y_test)
{

	map <int, double> p;

	int result;
	int score =0;

	for (auto i = 0; i < X_test.size(); i++)
	{
		vector<double> vec = X_test[i];

		assert(features_count_ == vec.size());


		double max = 0;
		for (auto lab : labels_list_) {
			p[lab] = p_class_[lab];
			for (auto i = 0; i < features_count_; i++) {
				p[lab] *= f_stats_[lab][2][i] * exp(-pow(vec[i] - f_stats_[lab][0][i], 2) / (2 * f_stats_[lab][1][i]));
			}

			if (max < p[lab]) {
				max = p[lab];
				result = lab;
			}
		}

		if (result==Y_test[i])
		{
			score += 1;
		}
	}

	return score;
}

// ******************************** Multinomial Navie Bayes************************************************

MultionomialGB::MultionomialGB() {

}

MultionomialGB::~MultionomialGB() {}

void MultionomialGB::train(vector<vector<double>> train, vector<int> label)
{

	//calculate multinomial_sums to find mean and variance
	
	double flag;
	double alpha = 1.0;
	// Decision rule
	int decision = 1;
	// Verbose
	int verbose = 0;
	cout << "Inside train " << endl;
	int labels;
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


		if (result==Y_test[i])
		{
			score += 1;
		}
	}

	return score;


}