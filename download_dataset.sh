#!/usr/bin/env bash

echo -e "Fetching the dataset"

cd data
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1E_C8uM7Ej6kqoS5GhB2qMUFc9e6-eBEM" -O preprocessed_dataset.zip
unzip preprocessed_dataset.zip
