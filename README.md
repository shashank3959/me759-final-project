# me759-final-project
CUDA Accelerated Implementation of Naive Bayes and it’s variants

# Download dataset
```
bash download_dataset.sh
```

# Data Preprocessing
We used [IMDb movie review dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and Iris Dataset and preprocessed the text data into suitable format such as onehot and bag of words models using Python packages such as NLTK and scikit-learn.

## Running on Euler
In order to run the Data Preprocessing python script on Euler, open preprocessData.sh file and change the algoID variable to the following options to create a data-set for that particular algorithm,

--algoID 1 for GaussianNB <br>
--algoID 2 for BernoulliNB <br>
--algoID 3 for MultinomialNB <br>
--algoID 4 for ComplementNB <br>

1. install all dependencies using <br>
```
pip3 install --user -r requirements.txt
```

2. run the script using <br>
```
sbatch preprocessData.sh
```
**Note that this may take 30-40 minutes for algos 2-4**. It will create .csv files in the data folder

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
# Functionality check
In order to check the functionality of our C++ implementation of Naive Bayes variants, we also run the Python machine learning package scikit-learn to compare our accuracy on the test set. For example, to test ComplementNB use algoID 4 <br>

## Running on Euler
Change the algoID variable in the file name checkFunctionality.sh and run the script using the following command and view in the accuracy in the log file.
Make sure all the requirements are installed in "requirements.txt" before
running this test.
```
pip3 install --user -r requirements.txt

sbatch checkFunctionality.sh
```
## Running on Windows/Mac
```
python test_algos.py --algoID 4
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
1. Load the CUDA module using "module load cuda/10.0"
2. Go to the Cuda\_NB folder <br>
3. sbatch NBcuda.sh
