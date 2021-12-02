
This is the code for our proposed method, SymbolicGPT. We tried to keep the implementation as simple and clean as possible to make sure it's understandable and easy to reuse. Please feel free to add features and submit a pull-request.

## Notes
This repository has been modified by @t-taniai to fix bugs in the [authors' original code](https://github.com/mojivalipour/symbolicgpt). The bug fixes contain modifications in dataset generation, and thus results using this repository will be different from the authors' original report.

# Results/Models/Datasets
- ~~Download via [link](https://www.dropbox.com/sh/yq03daorth1h4kj/AADolbgySCjOO18qGoP5Abqfa?dl=0)~~
- These data for dataset v2 are not available yet.

# Mirror Repository:
If you want to pull, open an issue or follow this repository, you can use this github repo [link](https://github.com/mojivalipour/symbolicgpt) which is a mirror repo for this one. Unfortunately, the UWaterloo GITLAB is limited to users with @uwaterloo emails. Therefore, you cannot contribute to this repository. Why do I not use github directly? You can find the answer [here](https://github.com/1995parham/github-do-not-ban-us). It's because I no longer trust GITHUB as my primary repository. Once, I was adversely affected for no good reason.

Original Repo: [link](https://git.uwaterloo.ca/data-analytics-lab/symbolicgpt2)

# Abstract:
Symbolic regression is the task of identifying a mathematical expression that best fits a provided dataset of input and output values. Due to the richness of the space of mathematical expressions, symbolic regression is generally a challenging problem. While conventional approaches based on genetic evolution algorithms have been used for decades, deep learning-based methods are relatively new and an active research area. In this work, we present SymbolicGPT, a novel transformer-based language model for symbolic regression. This model exploits the advantages of probabilistic language models like GPT, including strength in performance and flexibility. Through comprehensive experiments, we show that our model performs strongly compared to competing models with respect to the accuracy, running time, and data efficiency.

Paper: [link](https://arxiv.org/abs/2106.14131)

# Setup the environment
- Install [Anaconda](https://www.anaconda.com/products/individual/)
- Create the environment from environment.yml, as an alternative we also provided the requirements.txt (using Conda)
```bash
conda env create -f environment.yml
```
- As an alternative you can install the following packages:
```bash
pip install numpy
pip install torch
pip install matplotlib
pip install scipy
pip install tqdm
```

# Dataset Generation
Run the following script to generate datasets.

```bash
$ . make_dataset.sh
```

# Train/Test the model
Specify the configuration file by `--config` augment to train a model for a dataset.

```bash
$ python symbolicGPT.py --config configs/train_1-9var.json
$ python symbolicGPT.py --config configs/train_1-5var.json
$ python symbolicGPT.py --config configs/train_3var.json
$ python symbolicGPT.py --config configs/train_2var.json
$ python symbolicGPT.py --config configs/train_1var.json
```

# Directories
```Diff
symbolicGPT
│   README.md --- This file
│   .gitignore --- Ignore tracking of large/unnecessary files in the repo
│   environment.yml --- Conda environment file
│   requirements.txt --- Pip environment file
│   models.py --- Class definitions of GPT and T-Net
│   trainer.py --- The code for training Pytorch based models
│   data_loader.py --- Dataset class for data IO
│   utils.py --- Useful functions
│   symbolicGPT.py --- Main script to train and test our proposed method
│   dataset.py --- Main script to generate data.
│   make_dataset.sh --- Script to generate all datasets
│
└───configs
|   |   json config files for generating datasets and training models
│   
└───generator
│   └───treeBased --- equation generator based on expression trees
│       │   generateData.py --- Base class for data generation
|
└───results
    │   symbolicGPT --- reported results for our proposed method
    │   DSR --- reported results for Deep Symbolic Regression paper: 
    │   GP --- reported results for GPLearn: 
    │   MLP --- reported results for a simple blackbox multi layer perceptron

```

# System Spec:
- Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- Single NVIDIA GeForce RTX 2080 11 GB
- 32.0 GB Ram

# Citation:
```
@inproceedings{
    SymbolicGPT2021,
    title={SymbolicGPT: A Generative Transformer Model for Symbolic Regression},
    author={Mojtaba Valipour, Maysum Panju, Bowen You, Ali Ghodsi},
    booktitle={Preprint Arxiv},
    year={2021},
    url={https://arxiv.org/abs/2106.14131},
    note={Under Review}
}
```

# REFERENCES: 
- https://github.com/agermanidis/OpenGPT-2
- https://github.com/imcaspar/gpt2-ml
- https://huggingface.co/blog/how-to-train
- https://github.com/bhargaviparanjape/clickbait
- https://github.com/hpandana/gradient-accumulation-tf-estimator
- https://github.com/karpathy/minGPT
- https://github.com/charlesq34/pointnet
- https://github.com/volpato30/PointNovo
- https://github.com/brencej/ProGED
- https://github.com/brendenpetersen/deep-symbolic-optimization

# License:
MIT