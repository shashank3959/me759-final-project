#include "classifier.cuh"
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <vector>
#include <string>

vector<vector<float>> Load_State(string file_name) {
  ifstream in_state_(file_name.c_str(), ifstream::in);
  vector<vector<float>> state_out;
  string start;

  while (getline(in_state_, start)) {

    vector<float> x_coord;

    istringstream ss(start);
    float a;
    ss >> a;
    x_coord.push_back(a);

    string value;

    while (getline(ss, value, ',')) {
      float b;
      ss >> b;
      x_coord.push_back(b);
    }

    state_out.push_back(x_coord);
  }

  return state_out;
}

vector<float> Load_State_1D(string file_name, unsigned int &n_rows) {
  ifstream in_state_(file_name.c_str(), ifstream::in);
  vector<float> state_out;
  string start;
  n_rows = 0;

  while (getline(in_state_, start)) {

    // vector<float> x_coord;

    istringstream ss(start);
    float a;
    ss >> a;
    state_out.push_back(a);
    // x_coord.push_back(a);

    string value;

    while (getline(ss, value, ',')) {
      float b;
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
  cout<<"Selected algoID: "<<algoID <<endl;
  vector<float> X_train;
  vector<float> X_test;
  vector<int> Y_train;
  vector<int> Y_test;
  unsigned int n_rows_train;
  unsigned int n_rows_test;

  if (algoID == 0) {
    /* GaussianNB */
    #pragma omp parallel sections
    {
      #pragma omp section
      X_train = Load_State_1D("../data/train_states.csv", n_rows_train);

      #pragma omp section
      X_test = Load_State_1D("../data/test_states.csv", n_rows_test);

      #pragma omp section
      Y_train = Load_Label("../data/train_labels.csv");

      #pragma omp section
      Y_test = Load_Label("../data/test_labels.csv");
    }
  } else if (algoID == 1) {
    /* BernoulliNB */
   #pragma omp parallel sections
    {
      #pragma omp section
      X_train = Load_State_1D("../data/X_train_onehot.csv", n_rows_train);

      #pragma omp section
      X_test = Load_State_1D("../data/X_test_onehot.csv", n_rows_test);

      #pragma omp section
      Y_train = Load_Label("../data/y_train_onehot.csv");

      #pragma omp section
      Y_test = Load_Label("../data/y_test_onehot.csv");
    }
  } else if (algoID == 2 || algoID == 3) {
    /* MultinomialNB  or ComplementNB */
   #pragma omp parallel sections
    {
      #pragma omp section
      X_train = Load_State_1D("../data/X_train_bow.csv", n_rows_train);

      #pragma omp section
      X_test = Load_State_1D("../data/X_test_bow.csv", n_rows_test);

      #pragma omp section
      Y_train = Load_Label("../data/y_train_bow.csv");

      #pragma omp section
      Y_test = Load_Label("../data/y_test_bow.csv");
    }
  }

  cout << "X_train number of elements: " << X_train.size() << endl;
  cout << "Y_train number of elements: " << Y_train.size() << endl;
  cout << "X_test number of elements: " << X_test.size() << endl;
  cout << "Y_test number of elements: " << Y_test.size() << endl;

  unsigned int n_cols = X_train.size() / n_rows_train;

  cout << "Number of rows:" << n_rows_train << endl;
  cout << "Number of cols:" << n_cols << endl;

  // Timing CUDA events
  cudaEvent_t training_start, training_stop, testing_start, testing_stop;
  float ms_train = 0.0, ms_test = 0.0;
  cudaEventCreate(&training_start);
  cudaEventCreate(&training_stop);
  cudaEventCreate(&testing_start);
  cudaEventCreate(&testing_stop);

  // Classifier name
  string classifier;

  if (algoID == 0) {
    classifier = "Gaussian";

    GaussianNB model = GaussianNB();

    /* Training  */
    cout << "Training a " << classifier << " Naive Bayes classifier" << endl;

    cudaEventRecord(training_start);
    model.train(X_train, Y_train);
    cudaDeviceSynchronize();
    cudaEventRecord(training_stop);
    cudaEventSynchronize(training_stop);

    cudaEventElapsedTime(&ms_train, training_start, training_stop);
    cout << "Training time: " << ms_train << " ms" << endl;

    /* Testing */
    cout << "Testing..." << endl;

    int score = 0;
    cudaEventRecord(testing_start);
    score = model.predict(X_test, Y_test);
    cudaEventRecord(testing_stop);
    cudaEventSynchronize(testing_stop);

    float fraction_correct = float(score) / Y_test.size();
    cout << "Test accuracy: " << (100 * fraction_correct) << " percent" << endl;

    // Prints the time taken to run the code in ms
    cudaEventElapsedTime(&ms_test, testing_start, testing_stop);
    cout << "Testing time: " << ms_test << " ms" << endl;

  } else if (algoID == 1) {
    classifier = "Bernoulli";
    BernoulliNB model = BernoulliNB();

    /* Training  */
    cout << "Training a " << classifier << " Naive Bayes classifier" << endl;

    cudaEventRecord(training_start);
    model.train(X_train, Y_train);
    cudaEventRecord(training_stop);
    cudaEventSynchronize(training_stop);

    cudaEventElapsedTime(&ms_train, training_start, training_stop);
    cout << "Training time: " << ms_train << " ms" << endl;

    /* Testing */
    cout << "Testing..." << endl;

    int score = 0;
    cudaEventRecord(testing_start);
    score = model.predict(X_test, Y_test);
    cudaEventRecord(testing_stop);
    cudaEventSynchronize(testing_stop);

    float fraction_correct = float(score) / Y_test.size();
    cout << "Test accuracy: " << (100 * fraction_correct) << " percent" << endl;

    // Prints the time taken to run the code in ms
    cudaEventElapsedTime(&ms_test, testing_start, testing_stop);
    cout << "Testing time: " << ms_test << " ms" << endl;

  } else if (algoID == 2) {
    classifier = "Multinomial";
    MultinomialNB model = MultinomialNB();

    /* Training  */
    cout << "Training a " << classifier << " Naive Bayes classifier" << endl;

    cudaEventRecord(training_start);
    model.train(X_train, Y_train);
    cudaEventRecord(training_stop);
    cudaEventSynchronize(training_stop);

    cudaEventElapsedTime(&ms_train, training_start, training_stop);
    cout << "Training time: " << ms_train << " ms" << endl;

    /* Testing */
    cout << "Testing..." << endl;

    int score = 0;
    cudaEventRecord(testing_start);
    score = model.predict(X_test, Y_test);
    cudaEventRecord(testing_stop);
    cudaEventSynchronize(testing_stop);

    float fraction_correct = float(score) / Y_test.size();
    cout << "Test accuracy: " << (100 * fraction_correct) << " percent" << endl;

    // Prints the time taken to run the code in ms
    cudaEventElapsedTime(&ms_test, testing_start, testing_stop);
    cout << "Testing time: " << ms_test << " ms" << endl;

  } else if (algoID == 3) {
    classifier = "Complement";
    ComplementNB model = ComplementNB();

    /* Training  */
    cout << "Training a " << classifier << " Naive Bayes classifier" << endl;

    cudaEventRecord(training_start);
    model.train(X_train, Y_train);
    cudaEventRecord(training_stop);
    cudaEventSynchronize(training_stop);

    cudaEventElapsedTime(&ms_train, training_start, training_stop);
    cout << "Training time: " << ms_train << " ms" << endl;

    /* Testing */
    cout << "Testing..." << endl;

    int score = 0;
    cudaEventRecord(testing_start);
    score = model.predict(X_test, Y_test);
    cudaEventRecord(testing_stop);
    cudaEventSynchronize(testing_stop);

    float fraction_correct = float(score) / Y_test.size();
    cout << "Test accuracy: " << (100 * fraction_correct) << " percent" << endl;

    // Prints the time taken to run the code in ms
    cudaEventElapsedTime(&ms_test, testing_start, testing_stop);
    cout << "Testing time: " << ms_test << " ms" << endl;

  }

  /* Cleanup */
  cudaEventDestroy(training_start);
  cudaEventDestroy(training_stop);
  return 0;
}
