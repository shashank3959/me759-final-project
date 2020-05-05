'''
  Command to run the script: python preprocessData.py --algoID 1
  Choose algoID
  1 for GaussianNB
  2 for BernoulliNB
  3 for MultinomialNB
  4 for ComplementNB

'''
import numpy as np
import pandas as pd
import os
import re
import sys
import argparse
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# set seed for reproducible results
seed = 42

def cleanData():
	print("Loading Data...")
	reviews_train = []
	try:
		for line in open('./Dataset/aclImdb/movie_data/full_train.txt', 'r',encoding="utf8"):

			reviews_train.append(line.strip())

		reviews_test = []
		for line in open('./Dataset/aclImdb/movie_data/full_test.txt', 'r',encoding="utf8"):

			reviews_test.append(line.strip())
	except:
		print("Error in dataset path location")
		sys.exit()

	target = [1 if i < 12500 else 0 for i in range(25000)]

	REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
	REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
	NO_SPACE = ""
	SPACE = " "

	def preprocess_reviews(reviews):

		reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
		reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

		return reviews

	reviews_train_clean = preprocess_reviews(reviews_train)
	reviews_test_clean = preprocess_reviews(reviews_test)

	stemmer = PorterStemmer()
	english_stop_words = stopwords.words('english')

	def remove_stop_words_stemmer(corpus):
		removed_stop_words = []
		for review in corpus:
			removed_stop_words.append(
				' '.join([word for word in review.split()
						  if word not in english_stop_words])
			)
		return [' '.join([stemmer.stem(word) for word in review.split()]) for review in removed_stop_words]

	cleaned_train = remove_stop_words_stemmer(reviews_train_clean)
	cleaned_test = remove_stop_words_stemmer(reviews_test_clean)
	data = cleaned_test + cleaned_train
	return data,target


def save_file(X, target, ID):

	if not os.path.exists('data'):
		os.makedirs('data')

	if ID == "1":
		X, y = load_iris(return_X_y=True)

		# Test - Validation Split 80 % training and 20% testing
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

		#  Save to csv (from scipy sparse matrix representation)
		np.savetxt("./data/test_states.csv", X_test, delimiter=",")
		np.savetxt("./data/train_states.csv", X_train, delimiter=",")
		np.savetxt("./data/train_labels.csv", y_train, delimiter=",")
		np.savetxt("./data/test_labels.csv", y_test, delimiter=",")

	if ID == "2":
		cvb = CountVectorizer(binary=True, max_features=3000)
		cvb.fit(X)
		X = cvb.transform(X)

		# Test - Validation Split 75 % training and 25% testing
		X_train, X_val, y_train, y_val = train_test_split(
			X, target, train_size = 0.9, random_state=seed
		)

		# Bag of Words: Save to csv (from scipy sparse matrix representation)
		x = pd.DataFrame.sparse.from_spmatrix(X_val)
		x.to_csv('./data/X_test_onehot.csv',index=False, header=False)

		x = pd.DataFrame.sparse.from_spmatrix(X_train)
		x.to_csv('./data/X_train_onehot.csv',index=False, header=False)

		y_train_df = pd.DataFrame(data={"col1": y_train})
		y_train_df.to_csv("./data/y_train_onehot.csv", sep=',',index=False, header=False)

		y_test_df = pd.DataFrame(data={"col1": y_val})
		y_test_df.to_csv("./data/y_test_onehot.csv", sep=',',index=False, header=False)

	elif ID == "3" or ID == "4":
		# Binary = False will make sure counts show up
		cvw = CountVectorizer(binary=False, max_features=3000)
		cvw.fit(X)
		X = cvw.transform(X)

		# Test - Validation Split 75 % training and 25% testing
		X_train, X_val, y_train, y_val = train_test_split(
			X, target, train_size = 0.9, random_state=seed
		)

		# Bag of Words: Save to csv (from scipy sparse matrix representation)
		x = pd.DataFrame.sparse.from_spmatrix(X_val)
		x.to_csv('./data/X_test_bow.csv',index=False, header=False)

		x = pd.DataFrame.sparse.from_spmatrix(X_train)
		x.to_csv('./data/X_train_bow.csv',index=False, header=False)

		y_train_df = pd.DataFrame(data={"col1": y_train})
		y_train_df.to_csv("./data/y_train_bow.csv", sep=',',index=False, header=False)

		y_test_df = pd.DataFrame(data={"col1": y_val})
		y_test_df.to_csv("./data/y_test_bow.csv", sep=',',index=False, header=False)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--algoID", help="choose Naive Bayes algorithm variant")
	args = parser.parse_args()
	print("Chosen algoID is: ", args.algoID)

	if args.algoID in ["1", "2", "3", "4"]:
		if args.algoID != "1":
			X,target = cleanData()
			save_file(X,target,args.algoID)
			print("Successfully preprocessed the IMDB Dataset")
		else:
			X,target =[],[]
			save_file(X,target,args.algoID)
			print("Successfully preprocessed the Iris Dataset")

	else:
		print("Invalid algoID")
		sys.exit()
