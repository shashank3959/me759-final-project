#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J CudaNB 
#SBATCH -o CudaNB-%j.out -e CudaNB-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:gtx1080:1
#SBATCH --mem=16G

cd $SLURM_SUBMIT_DIR
module load cuda/10

nvcc main.cu classifier.cu -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -Wall -Xptxas -O3 -o CudaNB
./CudaNB 2
