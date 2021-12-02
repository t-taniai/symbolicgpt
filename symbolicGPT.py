#!/usr/bin/env python
# coding: utf-8

# set up logging
import logging

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# load libraries
import os
import glob
import json
import random
import numpy as np
#from tqdm import tqdm
from numpy import * # to override the math functions
import argparse
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
#from torch.utils.data import Dataset

from utils import calc_area_under_curve_score, evaluate, set_seed, sample_from_model
from matplotlib import pyplot as plt
from trainer import Trainer, TrainerConfig
from models import GPT, GPTConfig, PointNetConfig
from utils import relativeErr, varScaledMSE, normScaledMSE, mse, validation_tester
from data_loader import processDataFiles, CharDataset, collate_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str, help='The config json fil.')
    parser.add_argument('--seed', default=42, type=int, help='The seed.')
    parser.add_argument('--device', default='cuda', type=str, help='cuda|gpu|cpu')
    parser.add_argument('--scratch', action='store_true', help='ignore the cache and start for scratch')
    parser.add_argument('--numEpochs', default=20, type=int, help='number of epochs to train the GPT+PT model')
    parser.add_argument('--embeddingSize', default=512, type=int, help='the hidden dimension of the representation of both GPT and PT')
    parser.add_argument('--numPoints', nargs="+", default=[10, 200], type=int, help="number of points that we are going to receive to make a prediction about f given x and y, if you don't know then use the maximum")
    parser.add_argument('--numVars', default=5, type=int, help="the dimenstion of input points x, if you don't know then use the maximum")
    parser.add_argument('--numYs', default=1, type=int, help="the dimension of output points y = f(x), if you don't know then use the maximum")
    parser.add_argument('--blockSize', default=200, type=int, help="spatial extent of the model for its context")
    parser.add_argument('--batchSize', default=128, type=int, help="batch size of training data")
    parser.add_argument('--target', default='Skeleton', choices=['Skeleton','EQ'], type=str, help="Compact|Skeleton|EQ")
    parser.add_argument('--const_range', nargs="+", default=[-2.1, 2.1], type=float, help="constant range to generate during training only if target is Skeleton or Compact")
    parser.add_argument('--decimals', default=8, type=int, help="decimals of the points only if target is Skeleton")
    parser.add_argument('--trainRange', nargs="+", default=[-3.0,3.0], type=float, help="support range to generate during training only if target is Skeleton or Compact")
    parser.add_argument('--dataDir', default='./datasets/', type=str, help="root diretory of the datasets")
    parser.add_argument('--dataFolder', default='1-5Var_v2', type=str, help="target dataset name")
    parser.add_argument('--addr', default='./results/', type=str, help="the root directory for saving results")
    parser.add_argument('--condMethod', default='EMB_SUM', choices=['EMB_SUM','EMB_SUM','OUT_CAT','OUT_SUM', 'EMB_CON'], type=str, help="whether to concat the embedding or use summation")
    parser.add_argument('--varEmb', default='NOT_VAR', choices=['NOT_VAR','LEA_EMB','STR_VAR'], type=str, help="whether to concat the embedding or use summation")
    parser.add_argument('--errorFunc', default='rel', choices=['rel','var', 'norm', 'mse'], type=str, help="choice of error metric func between Y and YHat.")
    
    # load the json config and use it as default values.
    args = parser.parse_args()
    if args.config is not None:
        try:
            with open(args.config, 'r') as file:
                config = json.load(file)
            default_values = {key: config[key] for key in vars(args) if key in config}
            parser.set_defaults(**default_values)

            # parse the args again with the defaults specified by config
            args = parser.parse_args()
            print('Loaded preset hyper-parameters from config file.')

        except Exception as e:
            print('Exception occurs with config file:', args.config)
            print(e)

    # set the random seed
    set_seed(args.seed)

    # method = 'EMB_SUM' # EMB_CAT/EMB_SUM/OUT_SUM/OUT_CAT/EMB_CON -> whether to concat the embedding or use summation. 
    # EMB_CAT: Concat point embedding to GPT token+pos embedding
    # EMB_SUM: Add point embedding to GPT tokens+pos embedding
    # OUT_CAT: Concat the output of the self-attention and point embedding
    # OUT_SUM: Add the output of the self-attention and point embedding
    # EMB_CON: Conditional Embedding, add the point embedding as the first token
    
    # variableEmbedding = 'NOT_VAR' # NOT_VAR/LEA_EMB/STR_VAR
    # NOT_VAR: Do nothing, will not pass any information from the number of variables in the equation to the GPT
    # LEA_EMB: Learnable embedding for the variables, added to the pointNET embedding
    # STR_VAR: Add the number of variables to the first token
    addVars = True if args.varEmb == 'STR_VAR' else False
    ckptPath = args.addr
    try:
        os.makedirs(args.addr)
    except:
        print('Folder already exists!')

    with open(os.path.join(args.addr, 'params.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # load the train dataset
    train_file = '.cache/train_dataset_{}_{}.pt'.format(args.dataFolder, args.target)
    symbl_file = '.cache/train_dataset_{}_{}_symbols.pt'.format(args.dataFolder, args.target)
    train_dataset = None
    # try to load the train set
    if os.path.isfile(train_file) and not args.scratch:
        try:
            print('Loading train dataset from cache...')
            with open(train_file, 'rb') as f:
                train_dataset, symbols = torch.load(f)
        except Exception as e:
            print(e)

    if train_dataset is None:
        # process training files from scratch
        path = os.path.join(args.dataDir, args.dataFolder, 'Train/*.json')
        data = processDataFiles(glob.glob(path))
        symbols = []
        train_dataset = CharDataset(data, args.blockSize, symbols, numVars=args.numVars, 
                    numYs=args.numYs, numPoints=args.numPoints, target=args.target, addVars=addVars,
                    const_range=args.const_range, xRange=args.trainRange, decimals=args.decimals, augment=False)
        del data

        try:
            # using new zipfile for huge data may be slow and memory-consuming.
            with open(train_file, 'wb') as f:
                torch.save((train_dataset, symbols), f, _use_new_zipfile_serialization=False)
        except Exception as e:
            print(e)

        # save for validation and testing
        with open(symbl_file, 'wb') as f:
            torch.save(symbols, f)
    
    # load the val dataset
    path = os.path.join(args.dataDir, args.dataFolder, 'Val/*.json')
    data = processDataFiles(glob.glob(path))
    val_dataset = CharDataset(data, args.blockSize, symbols, numVars=args.numVars, 
                    numYs=args.numYs, numPoints=args.numPoints, target=args.target, addVars=addVars)
    del data

    # load the test data
    path = os.path.join(args.dataDir, args.dataFolder, 'Test/*.json')
    data = processDataFiles(glob.glob(path))
    test_dataset = CharDataset(data, args.blockSize, symbols, numVars=args.numVars, 
                    numYs=args.numYs, numPoints=args.numPoints, target=args.target, addVars=addVars)
    del data

    # create the model
    pconf = PointNetConfig(embeddingSize=args.embeddingSize, 
                        numberofPoints=args.numPoints[1]-1, 
                        numberofVars=args.numVars, 
                        numberofYs=args.numYs,
                        method=args.condMethod,
                        variableEmbedding=args.varEmb)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                    n_layer=8, n_head=8, n_embd=args.embeddingSize, 
                    padding_idx=train_dataset.paddingID)
    model = GPT(mconf, pconf)
    
    _errorFunc =  {
        'rel': relativeErr,
        'var': varScaledMSE,
        'norm': normScaledMSE,
        'mse': mse
    }[args.errorFunc]
    _collate_fn = partial(collate_fn, padding_id=train_dataset.paddingID)
    
    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(
        max_epochs=args.numEpochs, 
        batch_size=args.batchSize, 
        learning_rate=6e-4,
        lr_decay=True, 
        warmup_samples=len(train_dataset)//10, 
        final_samples=len(train_dataset)*args.numEpochs,
        num_workers=0, ckpt_path=ckptPath
    )
    trainer = Trainer(
        model, train_dataset, val_dataset, tconf, 
        device=args.device,
        collate_fn=_collate_fn,
        tester=partial(validation_tester, errorFunc=_errorFunc)
    )

    # # load the best model before training
    # print('The following model {} has been loaded!'.format(ckptPath))
    # model.load_state_dict(torch.load(ckptPath))
    # model = model.eval().to(trainer.device)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    # load the best model
    best_model_path = os.path.join(ckptPath, f'best.pt')
    if os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print('The following model {} has been loaded!'.format(best_model_path))
    else:
        print(f'Checkpoint {best_model_path} not found. Evaluate the current model.')
    model = model.eval().to(args.device)

    ## Test the model
    loader = torch.utils.data.DataLoader(
        test_dataset, 
        shuffle=False, 
        collate_fn=_collate_fn,
        pin_memory=True,
        batch_size=1,
        num_workers=0
    )

    resultDict = {}
    try:
        with open(os.path.join(args.addr, f'scores.txt'), 'w', encoding="utf-8") as o:
            def resultPrinter(ret):
                text = \
                    f"Test Case {ret['index']+1}/{len(test_dataset)}.\n" \
                    f"Target:{ret['target']}\n" \
                    f"Skeleton:{ret['pred_sk']}\n" \
                    f"Equation:{ret['pred_eq']}\n" \
                    f"Error: {ret['error']}\n"
                print(text)
                o.write(text + '\n')
            
            results = evaluate(
                model, loader, args.device,
                verbose=True,
                printer=resultPrinter,
                errorFunc=_errorFunc)
            
        errors = [x['error'] for x in results]
        print('Avg Err:{}'.format(np.mean(errors)))
        print('Area Above Curve:{}'.format(1-calc_area_under_curve_score(errors)))

        resultDict = {'SymbolicGPT': errors}
        torch.save(resultDict, os.path.join(args.addr, f'scores_best.pt'))
        
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    except Exception as e:
        print(e)

    # plot the error frequency for model comparison
    num_eqns = len(resultDict['SymbolicGPT'])
    num_vars = pconf.numberofVars
    title = "{} equations of {} variables - Benchmark".format(num_eqns, num_vars)

    models = list(key for key in resultDict.keys() if len(resultDict[key])==num_eqns)
    lists_of_error_scores = [resultDict[key] for key in models if len(resultDict[key])==num_eqns]
    linestyles = ["-","dashdot","dotted","--"]

    eps = 0.00001
    y, x, _ = plt.hist([np.log([max(min(x+eps, 1e5),1e-5) for x in e]) for e in lists_of_error_scores],
                    label=models,
                    cumulative=True, 
                    histtype="step", 
                    bins=2000, 
                    density=True,
                    log=False)
    y = np.expand_dims(y,0)
    plt.figure(figsize=(15, 10))

    for idx, m in enumerate(models): 
        plt.plot(x[:-1], 
            y[idx] * 100, 
            linestyle=linestyles[idx], 
            label=m)

    plt.legend(loc="upper left")
    plt.title(title)
    plt.xlabel("Log of Relative Mean Square Error")
    plt.ylabel("Normalized Cumulative Frequency")

    name = os.path.join(args.addr, f'plots_best.png')
    plt.savefig(name)

if __name__ == '__main__':
    main()