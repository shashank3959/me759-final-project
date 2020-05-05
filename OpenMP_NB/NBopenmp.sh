#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J OpenMP_NB 
#SBATCH -o OpenMP_NB-%j.out -e OpenMP_NB-%j.err
#SBATCH --nodes=1 --cpus-per-task=20
cd $SLURM_SUBMIT_DIR

g++ -std=c++0x main.cpp classifier.cpp -Wall -O3 -o OpenMP_NB -fopenmp
./OpenMP_NB 2
