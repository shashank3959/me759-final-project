#include "classifier.cuh"
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <vector>

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
  1: BernoulliNB
  2: MultinomialNB
  3: ComplementNB
  */
  int algoID = atoi(argv[1]);

  vector<double> X_train;
  vector<double> X_test;
  vector<int> Y_train;
  vector<int> Y_test;
  unsigned int n_rows_train;
  unsigned int n_rows_test;

  if (algoID == 0) {
    /* GaussianNB */
    #pragma omp parallel sections
    {
      #pragma omp section
      X_train = Load_State_1D("../OpenMP_NB/train_states.csv", n_rows_train);

      #pragma omp section
      X_test = Load_State_1D("../OpenMP_NB/test_states.csv", n_rows_test);

      #pragma omp section
      Y_train = Load_Label("../OpenMP_NB/train_labels.csv");

      #pragma omp section
      Y_test = Load_Label("../OpenMP_NB/test_labels.csv");
    }
  } else if (algoID == 1) {
    /* BernoulliNB */
    #pragma omp parallel sections
    {
      #pragma omp section
      X_train = Load_State_1D("../OpenMP_NB/X_train_onehot.csv", n_rows_train);

      #pragma omp section
      X_test = Load_State_1D("../OpenMP_NB/X_test_onehot.csv", n_rows_test);

      #pragma omp section
      Y_train = Load_Label("../OpenMP_NB/y_train_onehot.csv");

      #pragma omp section
      Y_test = Load_Label("../OpenMP_NB/y_test_onehot.csv");
    }
  } else if (algoID == 2 || algoID == 3) {
    /* MultinomialNB  or ComplementNB */
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

  cout << "Number of rows:" << n_rows_train << endl;

  cout << "Number of cols:" << n_cols << endl;

  // Timing CUDA events
  cudaEvent_t start, stop;
  float milliseconds = 0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (algoID == 0) {
    // GaussianNB
    cout << "Training a GaussianNB classifier" << endl;

    GaussianNB model = GaussianNB();
    model.train(X_train, Y_train);

    int score = 0;
    score = model.predict(X_test, Y_test);

    double fraction_correct = double(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " percent correct" << endl;

  } else if (algoID == 1) {
    // BernoulliNB

    cout << "Training a BernoulliNB NB classifier" << endl;

    cudaEventRecord(start);

    BernoulliNB model = BernoulliNB();
    model.train(X_train, Y_train);

    int score = 0;

    score = model.predict(X_test, Y_test);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    float fraction_correct = float(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " percent correct" << endl;

    // Prints the time taken to run the code in ms
    cout << "Time taken: " << milliseconds << " ms" << endl;

  } else if (algoID == 2) {
    /* MultinomialNB */
    cout << "Training a Multinomial NB classifier" << endl;

    cudaEventRecord(start);

    MultinomialNB model = MultinomialNB();
    model.train(X_train, Y_train);

    int score = 0;

    score = model.predict(X_test, Y_test);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    float fraction_correct = float(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " percent correct" << endl;

    // Prints the time taken to run the code in ms
    cout << "Time taken: " << milliseconds << " ms" << endl;

  } else if (algoID == 3) {
    // ComplementNB
    cout << "Training a ComplementNB classifier" << endl;

    cudaEventRecord(start);

    ComplementNB model = ComplementNB();
    model.train(X_train, Y_train);
    int score = 0;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    score = model.predict(X_test, Y_test);

    float fraction_correct = float(score) / Y_test.size();
    cout << "You got " << (100 * fraction_correct) << " percent correct" << endl;

    // Prints the time taken to run the code in ms
    cout << "Time taken: " << milliseconds << " ms" << endl;
  }

  return 0;
}
