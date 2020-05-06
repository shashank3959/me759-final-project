# me759-final-project
CUDA and OpenMP Accelerated Implementation of Naive Bayes and it’s variants

This project implements 4 different variants of the Naive Bayes algorithm, in CUDA and OpenMP separately to leverage hardware acceleration for best efficiency. The objective is also to compare how well the two (OpenMP/CUDA) implementations fare against each other.

There are the following steps to run this code, more details on which are available below:
1. Download the pre-processed dataset (Preferred; 30 seconds) **OR** Pre-process dataset at your own (30-45 mins).
2. Compiling and Running the dataset
3. (Optional) Functionality check against Scikit-learn's Naive Bayes algorithm.

The following sections expand on the above steps.

---
---

## 1. Get the dataset
We used [IMDb movie review dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) for the Document Classification Task (Bernoulli NB, Multinomial NB, Complement NB) and the Iris Dataset (GaussianNB) for flower classification. We the text data into suitable format such as onehot and bag of words representation using Python packages such as NLTK and scikit-learn.

### Download and extract the already pre-processed data (Option 1)
If you do this step, you need not pre-process your data!
```
bash download_dataset.sh
```

### Preprocess your own data (Option 2)
This is in case you don't like to download random files from the internet...we can understand.

#### Running on Euler or a Linux based system
In order to run the Data Preprocessing python script on Euler, please follow this link to install anaconda on Euler [here](https://wacc.wisc.edu/resources/docs/thirdparty.html) to install python and change the algoID variable to the following options to create a data-set for that particular algorithm,

* --algoID 1 for GaussianNB <br>
* --algoID 2 for BernoulliNB <br>
* --algoID 3 for MultinomialNB <br>
* --algoID 4 for ComplementNB <br>

1. Install all dependencies using <br>
```
pip3 install -r requirements.txt
```

2. Run the script using <br>
```
python preprocessData.py --algoID 2
```
**Note that this may take 30-40 minutes for algos Bernoulli, Multinomial and Complement**. It will create .csv files in the data folder. It is taking a long time since we are storing a huge matrix in CSV file and in this project we are more concenered on the HPC using CUDA and OpenMP rather than data preprocessing data so feel free to use download_dataset.sh to fetch the data. 

---
#### Running on Windows/Mac
If you are running this code on Windows/Mac, install python and install the dependencies by
<br>
```
pip3 install -r requirements.txt
```
and run the data preprocessing step using (change algoID to generated dataset for different algorithm, it would take some time to store the data in csv format) <br>
```
python preprocessData.py --algoID 2
```

**Note:** For GaussianNB, we used the Iris dataset which is already in numerical format so no data processing step was performed and the data is stored directly in the data folder. <br>

---
---

## 2. Compile and run the algorithms

### Running on Euler
#### OpenMP accelerated version
Go to the OpenMP\_NB folder <br>
```
sbatch NBopenmp.sh
```

#### CUDA accelerated version
Go to the Cuda\_NB folder <br>
```
module load cuda/10.0

sbatch NBcuda.sh
```

You may go into the \*.sh script to choose which variant you wish to run. For instance, in the CUDA version: <br>
* ./CudaNB 0 # Gaussian Naive Bayes on Iris (Flower Classication) Dataset


| Naive Bayes Variant | Dataset                               | AlgoID |
|---------------------|---------------------------------------|--------|
| Gaussian            | Iris (Flower Classication) Dataset    | 0      |
| Bernoulli           | IMDb (Movie Sentiment Classification) | 1      |
| Multinomial         | IMDb (Movie Sentiment Classification) | 2      |
| Complement          | IMDb (Movie Sentiment Classification) | 3      |

---
### Running on Windows/Mac/Ubuntu
#### OpenMP accelerated version
1. Go to the OpenMP\_NB folder. Then use the following: <br>

```
g++ -std=c++0x main.cpp classifier.cpp -Wall -O3 -o OpenMP_NB -fopenmp

./OpenMP_NB #algoID

```

#### CUDA accelerated version
1. Go to the Cuda\_NB folder. Then use the following: <br>

```
nvcc main.cu classifier.cu -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -Wall -Xptxas -O3 -o CudaNB

./CudaNB #algoID

```

---
---


## 3. (Optional) Functionality check (against Scikit-learn Naive Bayes)
In order to check the functionality of our C++ implementation of Naive Bayes variants, we also run the Python machine learning package scikit-learn to compare our accuracy on the test set. For example, to test ComplementNB use algoID 4 <br>

### Running on Euler
Install packages if you have not already done by the following the steps above and change the algoID variable in the following command and view in the accuracy in the log file.

```
pip3 install --user -r requirements.txt

python test_algos.py --algoID 4 
```

---
### Running on Windows/Mac/Ubuntu
```
python test_algos.py --algoID 4
```
---
