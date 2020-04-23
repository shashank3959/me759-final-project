#include "classifier.h"

// Row major index
unsigned int RM_Index(unsigned int row, unsigned int col, unsigned int width) {
  return (row*width + col);
}

MultinomialNB::MultinomialNB() {}

MultinomialNB::~MultinomialNB() {}

void MultinomialNB::train(vector<double> data, vector<int> labels) {
  unsigned int train_size = labels.size();
  n_features_ = data.size() / train_size;
  unsigned int n_classes = 0;

  // Declare a host vector
  host_vector<double> h_data(data.begin(), data.end());
  host_vector<int> h_labels(labels.begin(), labels.end());

  // Move data to the device
  device_vector<double> d_data = h_data;
  device_vector<int> d_labels = h_labels;

  device_vector<int> d_label_list = h_labels;

  // Number of unique labels
  thrust::sort(	d_label_list.begin(), d_label_list.end());
  auto new_last = thrust::unique(d_label_list.begin(),
                                  d_label_list.end());
  d_label_list.erase(new_last, d_label_list.end());
  n_classes = new_last - d_label_list.begin();

  // cout<<"dvec at: "<<d_data[RM_Index(10, 223, n_features_)]<<endl;

  // Find the number of unique labels


  cout<<endl;



}

int MultinomialNB::predict(vector<double> data, vector<int>  labels) {
  return 420;
}
