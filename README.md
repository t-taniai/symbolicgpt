
This is the code for our proposed method, SymbolicGPT. We tried to keep the implementation as simple and clean as possible to make sure it's understandable and easy to reuse. Please feel free to add features and submit a pull-request.

# Abstract:
Symbolic regression is the task of identifying a mathematical expression that best fits a provided dataset of input and output values. Due to the richness of the space of mathematical expressions, symbolic regression is generally a challenging problem. While conventional approaches based on genetic evolution algorithms have been used for decades, deep learning-based methods are relatively new and an active research area. In this work, we present SymbolicGPT, a novel transformer-based language model for symbolic regression. This model exploits the advantages of probabilistic language models like GPT, including strength in performance and flexibility. Through comprehensive experiments, we show that our model performs strongly compared to competing models with respect to the accuracy, running time, and data efficiency.

# Setup the environment
- Install [Anaconda](https://www.anaconda.com/products/individual/)
- Create the environment from environment.yml, as an alternative we also provided the requirements.txt
```bash
conda env create -f environment.yml
```

# Dataset Generation

You can skip this step if you already downloaded the datasets using this [link](https://drive.google.com/drive/folders/19HsGO_dMgx2xHQBJmgU8-ngy47cU-lNk?usp=sharing).

## How to generate the training data:
- Use the corresponding config file (config.txt) for each experiment
- Copy all the settings in config file into dataset.py
- Change the seed to 2021 in the dataset.py 
- Change the seed to 2021 in the generateData.py 
- Generate the data using the following command:
```bash
$ python dataset.py
```
- Move the generated data (./Datasets/\*.json) into the corresponding experiment directory (./datasets/{Experiment Name}/Train/\*.json)

## Generate the validation data:
- Use the corresponding config file (config.txt) for each experiment
- Copy all the settings in config file into dataset.py except the numSamples
- Make sure that the numSamples = 1000 // len(numVars) in the datasetTest.py 
- Change the seed to 2022 in the datasetTest.py 
- Change the seed to 2022 in the generateData.py 
- Generate the data using the following command:
```bash
$ python datasetTest.py
```
- Move the generated data (./Datasets/\*.json) into the corresponding experiment directory (./datasets/{Experiment Name}/Val/\*.json)

## Generate the test data:
- Use the corresponding config file (config.txt) for each experiment
- Copy all the settings in config file into dataset.py except the numSamples
- Make sure that the numSamples = 1000 // len(numVars) in the datasetTest.py 
- Change the seed to 2023 in the datasetTest.py 
- Change the seed to 2023 in the generateData.py 
- Generate the data using the following command:
```bash
$ python datasetTest.py
```
- Move the generated data (./Datasets/\*.json) into the corresponding experiment directory (./datasets/{Experiment Name}/Test/\*.json)

# Train/Test the model

It's easy to train a new model and reproduce the results.

## Configure the parameters

Follow each dataset config file and change the corresponding parameters (numVars, numPoints etc.) in the symbolicGPT.py script. 

## Reproduce the experiments

### Use this in symbolicGPT.py to reproduce the results for General 1-5 Model
```python
numEpochs = 4 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=200 # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=5 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 200 # spatial extent of the model for its context
batchSize = 64 # batch size of training data
dataDir = './datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '1-5Var_RandSupport_RandLength_-3to3_-5.0to-3.0-3.0to5.0_100to500Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
```
### Use this in symbolicGPT.py to reproduce the results for 1 Variable Model
```python
numEpochs = 4 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=30 # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=1 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 200 # spatial extent of the model for its context
batchSize = 64 # batch size of training data
dataDir = './datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '1Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_30Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
```

### Use this in symbolicGPT.py to reproduce the results for 2 Variable Model
```python
numEpochs = 4 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=200 # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=2 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 200 # spatial extent of the model for its context
batchSize = 64 # batch size of training data
dataDir = './datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '2Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_200Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
```

### Use this in symbolicGPT.py to reproduce the results for 3 Variable Model
```python
numEpochs = 4 # number of epochs to train the GPT+PT model
embeddingSize = 512 # the hidden dimension of the representation of both GPT and PT
numPoints=500 # number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum
numVars=3 # the dimenstion of input points x, if you don't know then use the maximum
numYs=1 # the dimension of output points y = f(x), if you don't know then use the maximum
blockSize = 200 # spatial extent of the model for its context
batchSize = 64 # batch size of training data
dataDir = './datasets/'
dataInfo = 'XYE_{}Var_{}Points_{}EmbeddingSize'.format(numVars, numPoints, embeddingSize)
titleTemplate = "{} equations of {} variables - Benchmark"
target = 'Skeleton' #'Skeleton' #'EQ'
dataFolder = '3Var_RandSupport_FixedLength_-3to3_-5.0to-3.0-3.0to5.0_500Points'
addr = './SavedModels/' # where to save model
method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation.
variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
```

## Run the script to train and test the model
```bash
python symbolicGPT.py
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
│   dataset.py --- Main script to generate training data
│   datasetTest.py --- Main script to generate test data
│
└───generator
│   │   
│   └───treeBased --- equation generator based on expression trees
│   │   │   generateData.py --- Base class for data generation
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