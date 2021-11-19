#!/usr/bin/bash

# ------------------- 1-5 Var -------------------
python dataset.py \
    --config ./configs/dataset_1-5var.json \
    --folder ./datasets/1-5Var_v2/Test/ \
    --seed 2023 \
    --testPoints 1 \
    --numSamplesEachEq 1 \
    --force_threshold 0 \
    --numSamples 200

python dataset.py \
    --config ./configs/dataset_1-5var.json \
    --folder ./datasets/1-5Var_v2/Val/ \
    --seed 2022 \
    --testPoints 1 \
    --numSamplesEachEq 1 \
    --force_threshold 0 \
    --numSamples 200

# Note: Original settings (numSamples=10000,numSamplesEachEq=50) should produce
# 2,500,000 samples (10000*50*5) but due to bugs the actual number was 2,262,058.
# Now we make 2,000,000 samples.
python dataset.py \
    --config ./configs/dataset_1-5var.json \
    --folder ./datasets/1-5Var_v2/Train/ \
    --seed 2021 \
    --testPoints 0 \
    --numSamplesEachEq 20 \
    --force_threshold 1 \
    --numSamples 20000


# ------------------- 1 Var -------------------
python dataset.py \
    --config ./configs/dataset_1var.json \
    --folder ./datasets/1Var_v2/Test/ \
    --seed 2023 \
    --testPoints 1 \
    --numSamplesEachEq 1 \
    --force_threshold 0 \
    --numSamples 200

python dataset.py \
    --config ./configs/dataset_1var.json \
    --folder ./datasets/1Var_v2/Val/ \
    --seed 2022 \
    --testPoints 1 \
    --numSamplesEachEq 1 \
    --force_threshold 0 \
    --numSamples 200

python dataset.py \
    --config ./configs/dataset_1var.json \
    --folder ./datasets/1Var_v2/Train/ \
    --seed 2021 \
    --testPoints 0 \
    --force_threshold 1


# ------------------- 2 Var -------------------
python dataset.py \
    --config ./configs/dataset_2var.json \
    --folder ./datasets/2Var_v2/Test/ \
    --seed 2023 \
    --testPoints 1 \
    --numSamplesEachEq 1 \
    --force_threshold 0 \
    --numSamples 200

python dataset.py \
    --config ./configs/dataset_2var.json \
    --folder ./datasets/2Var_v2/Val/ \
    --seed 2022 \
    --testPoints 1 \
    --numSamplesEachEq 1 \
    --force_threshold 0 \
    --numSamples 200

python dataset.py \
    --config ./configs/dataset_2var.json \
    --folder ./datasets/2Var_v2/Train/ \
    --seed 2021 \
    --testPoints 0 \
    --force_threshold 1


# ------------------- 3 Var -------------------
python dataset.py \
    --config ./configs/dataset_3var.json \
    --folder ./datasets/3Var_v2/Test/ \
    --seed 2023 \
    --testPoints 1 \
    --numSamplesEachEq 1 \
    --force_threshold 0 \
    --numSamples 200

python dataset.py \
    --config ./configs/dataset_3var.json \
    --folder ./datasets/3Var_v2/Val/ \
    --seed 2022 \
    --testPoints 1 \
    --numSamplesEachEq 1 \
    --force_threshold 0 \
    --numSamples 200

python dataset.py \
    --config ./configs/dataset_3var.json \
    --folder ./datasets/3Var_v2/Train/ \
    --seed 2021 \
    --testPoints 0 \
    --force_threshold 1


# ------------------- 1-9 Var -------------------
python dataset.py \
    --config ./configs/dataset_1-9var.json \
    --folder ./datasets/1-9Var_v2/Test/ \
    --seed 2023 \
    --testPoints 1 \
    --numSamplesEachEq 1 \
    --force_threshold 0 \
    --numSamples 111

python dataset.py \
    --config ./configs/dataset_1-9var.json \
    --folder ./datasets/1-9Var_v2/Val/ \
    --seed 2022 \
    --testPoints 1 \
    --numSamplesEachEq 1 \
    --force_threshold 0 \
    --numSamples 111

# Note: we make 2,250,000 samples.
python dataset.py \
    --config ./configs/dataset_1-9var.json \
    --folder ./datasets/1-9Var_v2/Train/ \
    --seed 2021 \
    --testPoints 0 \
    --numSamplesEachEq 25 \
    --force_threshold 1 \
    --numSamples 10000
