#include "classifier.h"
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <vector>
#include <omp.h>

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

vector<double> Load_State_1D(string file_name, unsigned int &n_rows) {
  ifstream in_state_(file_name.c_str(), ifstream::in);
  vector<double> state_out;
  string start;
  n_rows = 0;

  while (getline(in_state_, start)) {

    // vector<double> x_coord;

    istringstream ss(start);
    double a;
    ss >> a;
    state_out.push_back(a);
    // x_coord.push_back(a);

    string value;

    while (getline(ss, value, ',')) {
      double b;
      ss >> b;
      // x_coord.push_back(b);
      state_out.push_back(b);
    }
    ++n_rows;

    // state_out.push_back(x_coord);
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
  1: MultionomialGB
  2: BernoulliNB
  3: MultinomialNB
  4: ComplementNB
  */
  int algoID = atoi(argv[1]);

  vector<double> X_train;
  vector<double> X_test;
  vector<int> Y_train;
  vector<int> Y_test;
  unsigned int n_rows_train;
  unsigned int n_rows_test;

if (algoID == 3) {
    /* MultinomialNB */
    #pragma omp parallel sections
      {
        #pragma omp section
        X_train = Load_State_1D("../OpenMP_NB/X_train_bow.csv", n_rows_train);

        #pragma omp section
        X_test = Load_State_1D("../OpenMP_NB/X_test_bow.csv", n_rows_test);

        #pragma omp section
        Y_train = Load_Label("../OpenMP_NB/y_train_bow.csv");

        #pragma omp section
        Y_test = Load_Label("../OpenMP_NB/y_test_bow.csv");
      }
  }

  cout << "X_train number of elements: " << X_train.size() << endl;
  cout << "Y_train number of elements: " << Y_train.size() << endl << endl;

  cout << "X_test number of elements: " << X_test.size() << endl;
  cout << "Y_test number of elements: " << Y_test.size() << endl << endl;

  unsigned int n_cols = X_train.size() / n_rows_train;

  cout<<"Number of rows:"<<n_rows_train<<endl;

  cout<<"Number of cols:"<<n_cols<<endl;


  // Timing CUDA events
  cudaEvent_t start, stop;
  float milliseconds = 0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (algoID == 3) {
    /* MultinomialNB */
    cout<<"Training a Multinomial NB classifier"<<endl;

    cudaEventRecord(start);

    MultinomialNB model = MultinomialNB();
    model.train(X_train, Y_train);

    int score = 0;

    score = model.predict(X_test, Y_test);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    float fraction_correct = float(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " correct" << endl;

    // Prints the time taken to run the code in ms
    cout << "Time taken: "<<milliseconds << " ms"<<endl;
  } else {
    cout << "Algo not implemented yet!" << endl;
  }

  return 0;
}