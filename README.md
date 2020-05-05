# me759-final-project
CUDA Accelerated Implementation of Naive Bayes and it’s variants

# Data Preprocessing
We used [IMDb movie review dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and Iris Dataset and preprocessed the text data into suitable format such as onehot and bag of words models using Python packages such as NLTK and scikit-learn. 

## Running on Euler
In order to run the data Preprocessing python script on Euler, open preprocessPython.sh file and change the algoID variable to the following options to create dataset for that particular algorithm, 

1 for GaussianNB <br>
2 for BernoulliNB <br> 
3 for MultinomialNB <br>
4 for ComplementNB <br>

and **run the script using** <br> 
```
sbatch python.sh 
```
This will create .csv files in the data folder 
## Running on Windows/Mac
If you are running this code on Windows/Mac, install python and install the dependencies by 
<br>
```
pip3 install -r requirements.txt
```
and run the data preprocessing step using (change algoID to generated dataset for different algorithm, it would take some time to store the data in csv format) <br> 
```
python preprocessData.py --algoID 2 
```
### Functionality check 
In order to check the functionality of our C++ implementation of Naive Bayes variants, we also run the Python machine learning package scikit-learn to compare our accuracy on the test set. For example, to test ComplementNB use algoID 4 <br>

## Running on Euler 
Change the algoID variable in the file name checkFunctionality.sh and run the script using the following command and view in the accuracy in the log file. 

```
sbatch checkFunctionality.sh 
```
## Running on Windows/Mac
```
python main.py --algoID 4 
```

<br>

Note: <br> 
For GaussianNB, we used the Iris dataset which is already in numerical format so no data processing step was performed and the data is stored directly in the data folder. 

# To compile

## Running on Windows/Mac 
### OpenMP
1. Go to the OpenMP\_NB folder <br>
2. g++ main.cpp classifier.cpp -Wall -O3 -o classifier -fopenmp 

### CUDA
1. Go to the Cuda\_NB folder <br>
2. nvcc main.cu classifier.cu -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -Wall -Xptxas -O3 -o classifier\_gpu <br>

## Running on Euler 
### OpenMP
1. Go to the OpenMP\_NB folder <br>
2. sbatch NBopenmp.sh 

### CUDA
1. Go to the Cuda\_NB folder <br>
2. sbatch NBcuda.sh 
