
This is the code for our proposed method, SymbolicGPT, and we tried to keep the implementation as simple and clean as possible to make sure it's understandable and easy to reuse. 

# Abstract:
Symbolic regression is the task of identifying a mathematical expression that best fits a provided dataset of input and output values. Due to the richness of the space of mathematical expressions, symbolic regression is generally a challenging problem. While conventional approaches based on genetic evolution algorithms have been used for decades, deep learning-based methods are relatively new and an active area of research. In this work, we present a novel transformer-based language model for symbolic regression. This model exploits the strength and other possible flexibilities that have been provided by probabilistic language models like GPT. We show that our model is state of the art in terms of scalability and performance through comprehensive experiments.

# Setup the environment
## Install Anaconda
## Create the environment from environment.yml, as an alternative we also provided the requirements.txt
```bash
conda env create -f environment.yml
```

# Dataset Generation

Skip this part, if you want to use the already generated datasets in this repository. Just make sure to extract the datasets, and change the configuration.

## How to generate the training data:
```bash
$ cd generator
$ for P in {1..2} ; do sleep 1;  echo $P; python dataset.py ; done
```

## Generate the test data:
```bash
$ cd generator
$ python datasetTest.py
```

# Train/Test the model

It's easy to train a new model and reproduce the results.

## Configure the parameters

Follow each dataset config file and change the parameters in the symbolicGPT.py script. 

## Run the script
```bash
python symbolicGPT.py
```

## Reproduce the experiments
### Use this in symbolicGPT.py to reproduce the results for General 1-5 Model
```python
numEpochs = 4 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=500 # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=5 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 100 # spatial extent of the model for its context
batchSize = 128 # batch size of training data
dataDir = './datasets/'
dataInfo = 'XYE_1-{}Var_100-{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "1-{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '1-5Var_RandSupport_RandLength_0to3_3.1to6_100to500Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT -> whether to concat the embedding or use summation. 
```
### Use this in symbolicGPT.py to reproduce the results for 1 Variable Model
```python
numEpochs = 4 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=30 # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=1 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 100 # spatial extent of the model for its context
batchSize = 128 # batch size of training data
dataDir = './datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '1Var_RandSupport_FixedLength_0to3_3.1to6_30Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT -> whether to concat the embedding or use summation. 
```

### Use this in symbolicGPT.py to reproduce the results for 2 Variable Model
```python
numEpochs = 4 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=200 # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=2 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 100 # spatial extent of the model for its context
batchSize = 128 # batch size of training data
dataDir = './datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '2Var_RandSupport_FixedLength_0to3_3.1to6_200Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT -> whether to concat the embedding or use summation. 
```

### Use this in symbolicGPT.py to reproduce the results for 3 Variable Model
```python
numEpochs = 4 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=500 # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=3 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 100 # spatial extent of the model for its context
batchSize = 128 # batch size of training data
dataDir = './datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '3Var_RandSupport_FixedLength_0to3_3.1to6_500Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT -> whether to concat the embedding or use summation. 
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
│   utils.py --- Useful functions
│   symbolicGPT.py --- Main script to train and test our proposed method
│
└───generator
│   │   
│   └───treeBased --- equation generator based on expression trees
│   │   │   dataset.py --- Main script to generate training data
│   │   │   datasetTest.py --- Main script to generate test data
│   │   │   generateData.py --- Base class for data generation
│   └───grammarBased --- equation generator based on context free grammar
│   │
│   └───templateBased --- quation generator based on templates of general equations
│   │
└───results
│   │   symbolicGPT --- reported results for our proposed method
│   │   DSR --- reported results for Deep Symbolic Regression paper: 
│   │   GP --- reported results for GPLearn: 
│   │   MLP --- reported results for a simple blackbox multi layer perceptron
└───
```

# System Spec:
- Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- Single NVIDIA GeForce RTX 2080 11 GB
- 32.0 GB Ram

# Citation:
@inproceedings{
anonymous2021symbolic,
title={Symbolic Regression Using a Generative Transformer Model},
author={Anonymous},
booktitle={Submitted to Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=bTrP-koP-KB},
note={under review}
}

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

# Stable Commit:
- 8c258e3e425b9cdec79bbf5b0984d347fad2a1f7

# TODO: 
- [x] Reproduce the results for 1-5 General Model
- [x] Reproduce the results for 1 Variable Model
- [x] Reproduce the results for 2 Variable Model
- [x] Reproduce the results for 3 Variable Model
- [X] Create a dataset and test our pipeline with new data

# License:
MIT