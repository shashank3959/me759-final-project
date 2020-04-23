#include "classifier.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <omp.h>
#include <vector>

using namespace std;
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

vector<vector<double>> Load_State(string file_name) {
  ifstream in_state_(file_name.c_str(), ifstream::in);
  vector<vector<double>> state_out;
  string start;

  while (getline(in_state_, start)) {

    vector<double> x_coord;

    istringstream ss(start);
    double a;
    ss >> a;
    x_coord.push_back(a);

    string value;

    while (getline(ss, value, ',')) {
      double b;
      ss >> b;
      x_coord.push_back(b);
    }

    state_out.push_back(x_coord);
  }

  return state_out;
}
vector<int> Load_Label(string file_name) {
  ifstream in_label_(file_name.c_str(), ifstream::in);
  vector<int> label_out;
  string line;
  while (getline(in_label_, line)) {
    istringstream iss(line);
    int label;
    iss >> label;

    label_out.push_back(label);
  }
  return label_out;
}

int main(int argc, char *argv[]) {

  /*
  algoID
  0: GaussianNB
  1: BernoulliNB
  2: MultinomialNB
  3: ComplementNB
  */
  int algoID = 0;
  /*
  if (string(argv[i]) == "-d") {
          algoID = atoi(argv[i + 1]);
  }
  else {
          cout << "Invalid option. Code is exiting" << endl;
          exit(1);
  }
  */
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;

  vector<vector<double>> X_train;
  vector<vector<double>> X_test;
  vector<int> Y_train;
  vector<int> Y_test;

  /* GaussianNB or MultionomialNB */
  if (algoID == 0 ) {
    #pragma omp parallel sections
    {
      #pragma omp section
      X_train = Load_State("train_states.csv");

      #pragma omp section
      X_test = Load_State("test_states.csv");

      #pragma omp section
      Y_train = Load_Label("train_labels.csv");

      #pragma omp section
      Y_test = Load_Label("test_labels.csv");
    }
  } else if (algoID == 1) {
  /* BernoulliNB */
  #pragma omp parallel sections
    {
      #pragma omp section
      X_train = Load_State("X_train_onehot.csv");

      #pragma omp section
      X_test = Load_State("X_test_onehot.csv");

      #pragma omp section
      Y_train = Load_Label("y_train_onehot.csv");

      #pragma omp section
      Y_test = Load_Label("y_test_onehot.csv");
    }
  } else if (algoID == 2 || algoID == 3) {
    /* MultinomialNB or ComplementNB */
    #pragma omp parallel sections
      {
        #pragma omp section
        X_train = Load_State("X_train_bow.csv");

        #pragma omp section
        X_test = Load_State("X_test_bow.csv");

        #pragma omp section
        Y_train = Load_Label("y_train_bow.csv");

        #pragma omp section
        Y_test = Load_Label("y_test_bow.csv");
      }
  }

  cout << "X_train number of elements " << X_train.size() << endl;
  cout << "X_train element size " << X_train[0].size() << endl;
  cout << "Y_train number of elements " << Y_train.size() << endl << endl;

  cout << "X_test number of elements " << X_test.size() << endl;
  cout << "X_test element size " << X_test[0].size() << endl;
  cout << "Y_test number of elements " << Y_test.size() << endl << endl; 
  float tt=0;
  /* TODO: Remove code duplication below */
  if (algoID == 0) {
    
    start = high_resolution_clock::now();
    cout << "calling GaussianNB" << endl;
    GaussianNB model = GaussianNB();
    model.train(X_train, Y_train);

    end = high_resolution_clock::now();
    duration_sec =
        std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    cout << "training time " << duration_sec.count() << endl;

    int score = 0;
    score = model.predict(X_test, Y_test);

    double fraction_correct = double(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " correct" << endl;

  } else if (algoID == 1) {
    /* BernoulliNB */
    cout<<"Training a Bernoulli NB classifier"<<endl;
    BernoulliNB model = BernoulliNB();
    for(int i=0;i<10;i++)
   {
   start = high_resolution_clock::now();

    model.train(X_train, Y_train);

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    tt+=duration_sec.count();
  
  } 
    cout << "training time " << tt/10<< endl;
    int score = 0;

    score = model.predict(X_test, Y_test);

    double fraction_correct = double(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " correct" << endl;

  } else if (algoID == 2) {
    /* MultinomialNB */
    cout<<"Training a Multinomial NB classifier"<<endl;
    MultinomialNB model = MultinomialNB();
    for(int i=0;i<10;i++)
   {
    start = high_resolution_clock::now();
    model.train(X_train, Y_train);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
  
    tt+=duration_sec.count();
    
   } 
    cout << "training time " << tt/10<< endl;

    int score = 0;

    score = model.predict(X_test, Y_test);

    double fraction_correct = double(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " correct" << endl;
  } else if (algoID == 3) {
    /* ComplementNB */
    cout<<"Training a ComplementNB classifier"<<endl;
    ComplementNB model = ComplementNB();
    for(int i=0;i<10;i++)
   {
    start = high_resolution_clock::now();

    model.train(X_train, Y_train);
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
	
    tt+=duration_sec.count();
    
   } 
    cout << "training time " << tt/10<< endl;

    int score = 0;

    score = model.predict(X_test, Y_test);

    double fraction_correct = double(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " correct" << endl;
  }

  return 0;
}
