# me759-final-project
CUDA Accelerated Implementation of Naive Bayes and it’s variants

## Data Preprocessing Step 
The data preprocessing is an important step in the machine learning pipeline and we used Python only to preprocess the text data into suitable format such as onehot and bag of words models. In order to run the python script in Euler, open python.sh file and change the algoID variable as follows, 

1: GaussianNB
2: BernoulliNB
3: MultinomialNB
4: ComplementNB

and run the script using 
sbatch python.sh 

This will create .csv files in the data folder or if your running this code in Windows/Mac, install python and install the dependencies by 

pip3 install -r requirements.txt

Example: To create dataset for ComplementNB use algoID 4 
python main.py --algoID 4 

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
