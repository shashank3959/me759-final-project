#!/usr/bin/env bash

#SBATCH -p wacc
#SBATCH -J NBvariant
#SBATCH -o NBvariant-%j.out -e NBvariant-%j.err
#SBATCH --gres=gpu:1 -c 1
#SBATCH --mem=16G

cd $SLURM_SUBMIT_DIR
module load cuda/10


nvcc classifier.cu main.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o NBvariant

./NBvariant 3


