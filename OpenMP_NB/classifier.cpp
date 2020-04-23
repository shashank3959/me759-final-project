#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <assert.h>
#include <algorithm>
#include <numeric>
#include "classifier.h"
#include <omp.h>
#include <functional> 

using namespace std;

GaussianNB::GaussianNB() {

}

GaussianNB::~GaussianNB() {}

void GaussianNB::train(vector<vector<double>> data, vector<int> labels)
{
	map <int, vector<vector<double>>> lfm;
	map <int, int> class_count;
	int train_size = labels.size();
	int i,j,k;
	
	labels_list_ = labels;
	std::sort(labels_list_.begin(), labels_list_.end());
	std::vector<int>::iterator newEnd;
	// Override duplicate elements
	newEnd = std::unique(labels_list_.begin(), labels_list_.end());
	labels_list_.erase(newEnd, labels_list_.end());

	features_count_ = int(data[0].size());

	std::vector<int>::size_type lab = 0, l = 0;
	
	#pragma omp parallel 
	{
		#pragma omp for 
		for (lab = 0; lab < labels_list_.size(); lab++) {
			
			class_count[lab] = 0;
			vector <double> temp(data[0].size(), 0.0);
			f_stats_[lab].push_back(temp);
			f_stats_[lab].push_back(temp);
			f_stats_[lab].push_back(temp);
		}
	
		//gathering data list per class; count classes; sum per class
		#pragma omp for collapse(2)
		for (auto i = 0; i < train_size; i++) {
			
			for (auto j = 0; j < features_count_; j++) {
				f_stats_[labels[i]][0][j] += data[i][j]; // sum per feature
				if (j==0)
				  {
					#pragma omp critical
					{
					lfm[labels[i]].push_back(data[i]); // x_train per class
					class_count[labels[i]] += 1; // class count
					}
				  }
			}
		}
      
		// transforming f_stats 0 sum into mean	
		#pragma omp for collapse(2)
		for (auto lab =0 ; lab< labels_list_.size();++lab) {
			for (auto j = 0; j < features_count_; j++) {
				f_stats_[lab][0][j] /= class_count[lab];
				if(j==0)
				   p_class_[lab] = class_count[lab] * 1.0 / labels.size();
			}
			
		}
		
		#pragma omp for collapse(2)
		for (j = 0; j < features_count_; j++) {
		for (lab = 0; lab < labels_list_.size(); lab++) {
				 
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
	#pragma omp parallel num_threads(1)
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
	#pragma omp parallel 
	{
		#pragma omp for 
		for (auto lab =0; lab < labels_list_.size(); ++lab) {
			class_count_[lab] = 0;
			class_priors_[lab] = 0.0;
			vector <double> temp(data[0].size(), 0.0); /* 1 x n_features */
			feature_probs_[lab] = temp; /* n_labels x n_features */
		}

		/* How many documents contain each term (per label) */
		#pragma omp for collapse(2) 
		for (int i = 0; i < train_size; ++i) { /* For each example */
			for (unsigned int j = 0; j < n_features_; ++j) { /* For each feature*/
				// TOFIX: INDIRECTION */
				// TOFIX: Currently this requires labels to be in format 0, 1, 2, 3 ...
				//#pragma omp critical
				feature_probs_[labels[i]][j] += data[i][j];
				if (j==0)
				   class_count_[labels[i]] += 1;
					
			}
			
		}

		/* Convert the frequency to probability */
		#pragma omp for collapse(2)
		for (unsigned int i = 0; i < labels_list_.size(); ++i) {
			// TODO: Use stl transform for this
			for (unsigned int j = 0; j < n_features_; ++j) {
				//#pragma omp critical
				feature_probs_[i][j] /= class_count_[i];
				if (j==0)
				   class_priors_[i] = (double)class_count_[i] / train_size;
					
			}
			/* Calculate class priors for each class */
			
		}
	}
  return;
}

int BernoulliNB::predict(vector<vector<double>> X_test, vector<int> Y_test) {
	// TODO :n_features_ is actually redundant and SHOULD NOT change from train */
	n_features_ = X_test[0].size();
	std::vector<int>::size_type test_size = Y_test.size();
	int result = 0;
	int score = 0;
	unsigned int i = 0;
	
	std::vector<int>::size_type lab = 0, feat = 0;

	// probability for element belonging in a particular class
	map <int, double> prob_class;
	//double prob =0.0;
	/* For each example in the test set */
	#pragma omp parallel num_threads(1)
	{
		#pragma omp for 
		for (i = 0; i < test_size; ++i) {

			vector<double> test_vec = X_test[i];
			double max = 0.0;
			
			/* For each class.
			Note that labels_list_ is populated in the train function */
			for (lab = 0; lab < labels_list_.size(); ++lab) {

				prob_class[lab]= class_priors_[lab];

				/* For each feature */
				for (feat = 0; feat < n_features_; ++feat) {
					// TODO: Use a reduction technique here
					prob_class[lab]*= pow(feature_probs_[lab][feat], test_vec[feat]);
					prob_class[lab]*= pow((1 - feature_probs_[lab][feat]),(1- test_vec[feat]));
				}

				//#pragma omp atomic update
				//	prob_class[lab] += prob;
				if (max < prob_class[lab]) {
					max = prob_class[lab];
					result = lab;
				}
			}
			#pragma omp critical
			if (result == Y_test[i]) {
				score += 1;
			}
		}
	}
	return score;
}

/************Multinomial Naive Bayes (Text Classification)********************/
MultinomialNB::MultinomialNB() {}

MultinomialNB::~MultinomialNB() {}

void MultinomialNB::train(vector<vector<double>> data, vector<int> labels) {
	int train_size = labels.size();
	n_features_ = data[0].size();

	/* For alpha = 1 for laplacian smoothing, and alpha < 1 for lidstone smoothing */
	double alpha = 1.0;
	unsigned int vocab_size = n_features_; /* As many features as words in train set */

	/* Number of unique labels */
	labels_list_ = labels;
	std::sort(labels_list_.begin(), labels_list_.end());
	auto newEnd = std::unique(labels_list_.begin(), labels_list_.end());
	labels_list_.erase(newEnd, labels_list_.end());


	/* Initializing class variables */
	#pragma omp parallel 
	{
		#pragma omp for 
		for (auto lab =0 ;lab <labels_list_.size();++lab) {
			class_count_[lab] = 0;
			class_priors_[lab] = 0.0;
			feat_count_[lab] = 0;
			vector <double> temp(data[0].size(), 0.0); /* 1 x n_features */
			feature_probs_[lab] = temp; /* n_labels x n_features */
		}

		/* frequency of occurence of each feature */
		#pragma omp for collapse(2)
		for (int i = 0; i < train_size; ++i) { /* For each example */
			for (unsigned int j = 0; j < n_features_; ++j) { /* For each feature*/
				// TOFIX: INDIRECTION
				// TOFIX: Currently this requires labels to be in format 0, 1, 2, 3 ...
				feature_probs_[labels[i]][j] += data[i][j];

				// Sum of all occurence of words for each class
				feat_count_[labels[i]] += data[i][j];

				if(j==0)
					class_count_[labels[i]] += 1;
			}
			
		}

		/* Calculate word occurence probabilities per label */
		#pragma omp for collapse(2)
		for (unsigned int i = 0; i < labels_list_.size(); ++i) {
			// TODO: Use stl transform for this
			for (unsigned int j = 0; j < n_features_; ++j) {
				// Conditional probs with laplacian smoothing
				feature_probs_[i][j] =  log(((double)feature_probs_[labels[i]][j] + alpha) / ((double)(feat_count_[i] + vocab_size))+1);
				//cout<<"feature_probs_[i][j]  "<<feature_probs_[i][j] <<endl;
				if(j==0)
					class_priors_[i] = (double)class_count_[i] / train_size;
			}
			/* Calculate class priors for each class */
			
		}
	}
	return;
}


int MultinomialNB::predict(vector<vector<double>> X_test, vector<int> Y_test) {
	// TODO :n_features_ is actually redundant and SHOULD NOT change from train */
	n_features_ = X_test[0].size();
	std::vector<int>::size_type test_size = Y_test.size();
	int result = 0;
	int score = 0;
	unsigned int i = 0;
	//double max = 0.0;
	std::vector<int>::size_type lab = 0, feat = 0;

	/* Note that labels_list_ is populated in the train function */
	unsigned int n_labels = labels_list_.size();

	// probability for element belonging in a particular class
	map <int, double> prob_class;

	/* For each example in the test set */
	#pragma omp parallel num_threads(1)
	{
		#pragma omp for
		for (i = 0; i < test_size; ++i) {
			vector<double> test_vec = X_test[i];
			map <int, double> prob_class;
			double max = 0.0;
			
			/* For each class.
			Note that labels_list_ is populated in the train function */
			for (lab = 0; lab < n_labels; ++lab) {
				
				prob_class[lab]= log(class_priors_[lab]);
				
				for (feat = 0; feat < n_features_; ++feat) {
								
				prob_class[lab]+= (double)((feature_probs_[lab][feat])*(double)(test_vec[feat]));
				
				}
				cout<<" prob_class[lab] "<<prob_class[lab]<<endl;
				if (max < (prob_class[lab])) {
					max = prob_class[lab];
					result = lab;
					cout<<"chnged"<<endl;
				}
			}
			
			#pragma omp critical
			if (result == Y_test[i]) {
				score += 1;
			}
		}
	}
	return score;
}


/************Complement Naive Bayes (Text Classification)********************/
ComplementNB::ComplementNB() {}

ComplementNB::~ComplementNB() {}

/* 
Implementation reference:
https://scikit-learn.org/stable/modules/naive_bayes.html#complement-naive-bayes
*/
void ComplementNB::train(vector<vector<double>> data, vector<int> labels) {
	unsigned int train_size = labels.size();
	n_features_ = data[0].size();
	unsigned int n_unique_labels = 0;

	/* Params for laplacian smoothing */
	double alpha_feat = 1.0; // alpha for each feature
	double alpha = (double)n_features_ * alpha_feat;

	/* Number of unique labels */
	labels_list_ = labels;
	std::sort(labels_list_.begin(), labels_list_.end());
	auto newEnd = std::unique(labels_list_.begin(), labels_list_.end());
	labels_list_.erase(newEnd, labels_list_.end());

	n_unique_labels = labels_list_.size();

	/* Initializing total occurence vectors */
	vector <int> temp(data[0].size(), 0);
	all_occur_per_term = temp;
	all_occur = 0;

	// Variables required for next loop
	unsigned int num_sum = 0, den_sum = 0;
	//extern const _Placeholder<1> _1;
	/* Initializing class variables */
	#pragma omp parallel 
	{
		#pragma omp for 
		for (auto  lab=0; lab <labels_list_.size();++lab) {
			class_count_[lab] = 0; // Total docs in each class
			feat_count_[lab] = 0; // Total words in each class

			// TODO: Move temp declaration out of the loop
			vector <int> temp(data[0].size(), 0); /* 1 x n_features */
			feature_frequencies_[lab] = temp;

			// TODO: Move temp declaration out of the loop
			vector <double> temp_double(data[0].size(), 0.0); /* 1 x n_features */
			feature_weights_[lab] = temp_double;
		}

		

		/* frequency of occurence of each feature by class */
		#pragma omp for collapse(2) 
		for (unsigned int i = 0; i < train_size; ++i) { /* For each example */
			for (unsigned int j = 0; j < n_features_; ++j) { /* For each feature*/
				// TOFIX: INDIRECTION
				// TOFIX: Currently this requires labels to be in format 0, 1, 2, 3 ...
				
				feature_frequencies_[labels[i]][j] += data[i][j];
				all_occur_per_term[j] += data[i][j];

				// Sum of occurence of all words for each class
				feat_count_[labels[i]] += data[i][j];
				
				if (j==0)
				class_count_[labels[i]] += 1;
			}
			
		}

		/* Total occurences of all words in training set */
		#pragma omp single
		{
		all_occur = accumulate(all_occur_per_term.begin(), all_occur_per_term.end(), 0);
		//cout<<"all_occur"<<all_occur<<endl;
		}
		

		/* Complement calculations for getting feature weights */
		#pragma omp for collapse(2)
		for (unsigned int i = 0; i < n_unique_labels; ++i) { /* For each class */
			
			for (unsigned int j = 0; j < n_features_; ++j) { /* For each feature */

				
				/* Complement Calculations */
						
				num_sum = all_occur_per_term[j] - feature_frequencies_[i][j];

				feature_weights_[i][j] = log(((double)num_sum + alpha_feat) /((double)den_sum + alpha));
				
				if(j==0)
				   den_sum = all_occur - accumulate(feature_frequencies_[i].begin(), feature_frequencies_[i].end(), 0);
				  
				  
			}
						
		}
		
		/* Normalizing feature weights for each class. 
		This supposedly alleviates class imbalance */ 
		#pragma omp for collapse(2)
		for (unsigned int i = 0; i < n_unique_labels; ++i) { /* For each class */
		
		// TODO: Use stl transform for this, this is super inefficient
		for (unsigned int j = 0; j < n_features_; ++j) { /* For each feature */
			if(j=0)
			   den_sum = accumulate(feature_weights_[i].begin(),feature_weights_[i].end(), 0.0);
			feature_weights_[i][j] /= den_sum;
		}
		}
	}
	
	return;
}

int ComplementNB::predict(vector<vector<double>> X_test, vector<int> Y_test) {
	//double min = 0.0;
	std::vector<int>::size_type lab = 0, feat = 0;
	std::vector<int>::size_type test_size = Y_test.size();
	int result = 0;
	unsigned int score = 0;
	unsigned int i = 0;

	/* Note that labels_list_ is populated in the train function */
	unsigned int n_labels = labels_list_.size();

	// probability for element belonging in a particular class
	map <int, double> prob_class;

	/* For each example in the test set */
	#pragma omp parallel num_threads(1)
	{
		#pragma omp for 
		for (i = 0; i < test_size; ++i) {
			/* Move test_vec declaration outside the loop */
			vector<double> test_vec = X_test[i];
			double min = 0.0;

			/* For each class check if it's the poorest complement match */
			
			for (lab = 0; lab < n_labels; ++lab) {
				prob_class[lab] = 0.0;
				for (feat = 0; feat < n_features_; ++feat) {
				// TODO: Use a reduction technique here
				prob_class[lab] += feature_weights_[lab][feat] * (double)test_vec[feat];
				}
				if (min > prob_class[lab]) {
					min = prob_class[lab];
					result = lab;
				}
			}
			#pragma omp critical
			if (result == Y_test[i]) {
				score += 1;
			}
		}
	}	
	return score;

}
