'''
  Command to run the script: python main.py --algoID 1  
  Choose algoID
  1 for GaussianNB
  2 for MultinomialNB
  3 for BernoulliNB
  4 for ComplementNB

'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
import pandas as pd 
import sys 
import argparse
import numpy as np

def train_test(ID):

	if ID== "1":

		gnb = GaussianNB()
		
		#load data
		print("Loading train and test data")
		X_train = pd.read_csv("./data/train_states.csv")
		X_test = pd.read_csv("./data/test_labels.csv")
		Y_train = np.ravel(pd.read_csv("./data/train_labels.csv"))
		Y_test = np.ravel(pd.read_csv("./data/test_labels.csv"))

		#train model 
		print("Training models ")
		gnb.fit(X_train, Y_train)

		#predict on test data
		pred = gnb.predict(X_test)

		print("GaussianNB model accuracy :",accuracy_score(Y_test, pred))

	elif ID == "2":
		clf = MultinomialNB()

		#load data
		print("Loading train and test data")
		X_train = pd.read_csv("./data/X_train_bow.csv")
		X_test = pd.read_csv("./data/X_test_bow.csv")
		Y_train = np.ravel(pd.read_csv("./data/y_train_bow.csv"))
		Y_test = np.ravel(pd.read_csv("./data/y_test_bow.csv"))

		#train model 
		print("Training model")
		clf.fit(X_train, Y_train)

		#predict on test data
		pred = clf.predict(X_test)

		print("MultinomialNB model accuracy :",accuracy_score(Y_test, pred))

	elif ID == "3":
		    
		clf = BernoulliNB()

		#load data
		print("Loading train and test data")
		X_train = pd.read_csv("./data/X_train_onehot.csv")
		X_test = pd.read_csv("./data/X_test_onehot.csv")
		Y_train = np.ravel(pd.read_csv("./data/y_train_onehot.csv"))
		Y_test = np.ravel(pd.read_csv("./data/y_test_onehot.csv"))

		#train model 
		print("Training model")
		clf.fit(X_train, Y_train)

		#predict on test data
		pred = clf.predict(X_test)

		print("BernoulliNB model accuracy :", accuracy_score(Y_test, pred))

	elif ID == "4":
		clf = ComplementNB()

		#load data
		print("Loading train and test data")
		X_train = pd.read_csv("./data/X_train_bow.csv")
		X_test = pd.read_csv("./data/X_test_bow.csv")
		Y_train = np.ravel(pd.read_csv("./data/y_train_bow.csv"))
		Y_test = np.ravel(pd.read_csv("./data/y_test_bow.csv"))

		#train model 
		print("Training model")
		clf.fit(X_train, Y_train)

		#predict on test data
		pred = clf.predict(X_test)

		print("ComplementNB model accuracy :",accuracy_score(Y_test, pred))

	else:
		print("invalid ID exiting the code")
		sys.exit()

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--algoID", help="choose naive_bayes algorithm variant")
	args = parser.parse_args()
	id = args.algoID
	train_test(id)


