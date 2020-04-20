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

int main() {

  /*
  algoID
  0: GaussianNB
  1: MultionomialGB
  2: BernoulliNB
  3: ComplementNB
  */
  int algoID = 2;
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

  vector<vector<double>> X_train;
  vector<vector<double>> X_test;
  vector<int> Y_train;
  vector<int> Y_test;

  /* GaussianNB or MultionomialNB */
  if (algoID == 0 || algoID == 1) {
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
  } else if (algoID == 2) {
  /* BernoulliNB */
  #pragma omp parallel sections
    {
      #pragma omp section
      X_train = Load_State("X_train.csv");

      #pragma omp section
      X_test = Load_State("X_test.csv");

      #pragma omp section
      Y_train = Load_Label("y_train.csv");

      #pragma omp section
      Y_test = Load_Label("y_test.csv");
    }
  }

  cout << "X_train number of elements " << X_train.size() << endl;
  cout << "X_train element size " << X_train[0].size() << endl;
  cout << "Y_train number of elements " << Y_train.size() << endl << endl;

  cout << "X_test number of elements " << X_test.size() << endl;
  cout << "X_test element size " << X_test[0].size() << endl;
  cout << "Y_test number of elements " << Y_test.size() << endl << endl;

  if (algoID == 0) {
    duration<double, std::milli> duration_sec;
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

    float fraction_correct = float(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " correct" << endl;

  } else if (algoID == 1) {
    MultionomialGB model = MultionomialGB();
    cout << "Calling train" << endl;
    model.train(X_train, Y_train);

    int score = 0;

    score = model.predict(X_test, Y_test);

    float fraction_correct = float(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " correct" << endl;
  }

  else if (algoID == 2) {
    cout<<"Training a Bernoulli NB classifier"<<endl;

    BernoulliNB model = BernoulliNB();
    model.train(X_train, Y_train);

    int score = 0;

    score = model.predict(X_test, Y_test);

    float fraction_correct = float(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " correct" << endl;

  }

  return 0;
}
