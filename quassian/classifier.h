
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

    map <string, double> p_class_;
    map <string, vector<vector<double>>> f_stats_; // 0 - mean; 1 - var
    vector <string> labels_list_;
    int features_count_;

    GaussianNB();

    virtual ~GaussianNB();

    void train(vector<vector<double> > data, vector<string>  labels);

    string predict(vector<double> vec);

};


class MultionomialGB {
public:

    map <string, double> p_class_;
    map <string, vector<vector<double>>> f_stats_; // 0 - mean; 1 - var
    vector <string> labels_list_;
    int features_count_;

    MultionomialGB();


    virtual ~MultionomialGB();

    void train(vector<vector<double> > data, vector<string>  labels);

    string predict(vector<double> vec);

};

#endif



