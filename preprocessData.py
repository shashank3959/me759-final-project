'''
  Command to run the script: python preprocessData.py --algoID 1  
  Choose algoID
  2 for MultinomialNB
  3 for BernoulliNB
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


# set seed for reproducible results
seed = 42

def cleanData():
	print("Loading Data")
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
		
	return cleaned_train,cleaned_train,target


def save_file(cleaned_train,cleaned_test,target,ID):

	if not os.path.exists('data'):
		os.makedirs('data')	

	if ID== "2":
		cvb = CountVectorizer(binary=True,max_features=3000)
		cvb.fit(cleaned_train)
		X = cvb.transform(cleaned_train)
		X_test = cvb.transform(cleaned_test)

		# Test - Validation Split 75 % training and 25% testing 
		X_train, X_val, y_train, y_val = train_test_split(
		    X, target, train_size = 0.75, random_state=seed
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

	elif ID== "3" or ID =="4":
		# Binary = False will make sure counts show up
		cvw = CountVectorizer(binary=False, max_features=3000)
		cvw.fit(cleaned_train)
		X = cvw.transform(cleaned_train)
		X_test = cvw.transform(cleaned_test)

		# Test - Validation Split 
		X_train, X_val, y_train, y_val = train_test_split(
		    X, target, train_size = 0.75, random_state=seed
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
	parser.add_argument("--algoID", help="choose naive_bayes algorithm variant")
	args = parser.parse_args()


	if args.algoID== "2" or args.algoID== "3" or args.algoID== "4":
		save_file(train_data,test_data,target,args.algoID)
		train_data,test_data,target = cleanData()
	else:
		print("Invalid algoID")
		sys.exit()
	
	print("Successfully preprocessed the movie reviews")




