#include "classifier.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>

using namespace std;


vector<vector<double> > Load_State(string file_name)
{
	ifstream in_state_(file_name.c_str(), ifstream::in);
	vector< vector<double >> state_out;
	string start;

	while (getline(in_state_, start))
	{

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
vector<int> Load_Label(string file_name)
{
	ifstream in_label_(file_name.c_str(), ifstream::in);
	vector< int > label_out;
	string line;
	while (getline(in_label_, line))
	{
		istringstream iss(line);
		int label;
		iss >> label;

		label_out.push_back(label);
	}
	return label_out;

}

int main() {

	int algoID = 1;
	/*
	if (string(argv[i]) == "-d") {
		algoID = atoi(argv[i + 1]);
	}
	else {
		cout << "Invalid option. Code is exiting" << endl;
		exit(1);
	}
	*/

	vector< vector<double> > X_train = Load_State("train_states.csv");
	vector< vector<double> > X_test = Load_State("test_states.csv");
	vector< int > Y_train = Load_Label("train_labels.csv");
	vector< int > Y_test = Load_Label("test_labels.csv");

	cout << "X_train number of elements " << X_train.size() << endl;
	cout << "X_train element size " << X_train[0].size() << endl;
	cout << "Y_train number of elements " << Y_train.size() << endl << endl;


	cout << "X_test number of elements " << X_test.size() << endl;
	cout << "X_test element size " << X_test[0].size() << endl;
	cout << "Y_test number of elements " << Y_test.size() << endl << endl;

	if (algoID == 0)
	{
		GaussianNB model = GaussianNB();
		model.train(X_train, Y_train);


		int score = 0;
		score = model.predict(X_test, Y_test);

		float fraction_correct = float(score) / Y_test.size();
		cout << "You got " << (100 * fraction_correct) << " correct" << endl;
	}
	
	else if (algoID == 1)
	{
		
		MultionomialGB model = MultionomialGB();
		cout << "Calling train" << endl;
		model.train(X_train, Y_train);


		int score = 0;
		score = model.predict(X_test, Y_test);

		float fraction_correct = float(score) / Y_test.size();
		cout << "You got " << (100 * fraction_correct) << " correct" << endl;
	}
	




	return 0;
}


