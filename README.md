# me759-final-project
CUDA Accelerated Implementation of Naive Bayes and it’s variants

## Data Preprocessing Step 
We used [IMDb movie review dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and preprocessed the text data into suitable format such as onehot and bag of words models using Python packages such as NLTK and scikit-learn. In order to run the python script in Euler, open preprocessPython.sh file and change the algoID variable as follows to create dataset for that particular algorithm, 

1: GaussianNB <br>
2: BernoulliNB <br> 
3: MultinomialNB <br>
4: ComplementNB <br>

and run the script using <br> 
```
sbatch python.sh 
```
This will create .csv files in the data folder or if your running this code in Windows/Mac, install python and install the dependencies by 
<br>
```
pip3 install -r requirements.txt
```
and run the data preprocessing step using (change algoID to generated dataset for different algorithm, it would take some time to store the data in csv format) <br> 
```
python preprocessData.py --algoID 2 
```
### Functionality check 
In order to check the functionality of our C++ implemenation of Navies Bayies variant, we used the Python machine learning package scikit-learn to verify the funcationality by comparing the accuracy on the test set data. For example, to test ComplementNB use algoID 4 <br>
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
