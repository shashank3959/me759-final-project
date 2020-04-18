#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J task1
#SBATCH -o task1-%j.out -e task1-%j.err
#SBATCH --nodes=1 --cpus-per-task=20
cd $SLURM_SUBMIT_DIR

g++ -std=c++0x main.cpp classifier.cpp -Wall -O3 -o model -fopenmp
./model 

