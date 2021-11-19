#!/usr/bin/env python
# coding: utf-8

import re
import os
import json
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime
from generator.treeBased.generateData import dataGen
import argparse
import random
import warnings
from distutils.util import strtobool
import json

def generateXYDataFromEquation(eq, n_points=2, n_vars=3, decimals=4, supportPoints=None, min_x=0, max_x=3):
    # Do not use the safe wrapper functions in utils.py
    # so that invalid eq producing NaN will be rejected.
    ref_dict = {
        'sin': np.sin,
        'cos': np.cos,
        'log': np.log,
        'exp': np.exp,
        'abs': np.abs,
        'sqrt': np.sqrt,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'divide': np.divide,
    }
    
    if supportPoints is None:
        if isinstance(min_x, list):
            # when x-range is a union of intervals,
            # uniformly sample from this union using gumbel sampling.
            # For example, when interval1 = [0,1) and interval2 = [50,100),
            # then x is x50 more frequently sampled from interval2.

            # log-probability to select each interval
            logp = np.array([ma-mi for mi,ma in zip(min_x, max_x)])[None]
            logp = np.log(logp/logp.sum())
            x = []
            for _ in range(n_vars):
                # make random values for each interval
                r = [np.random.uniform(mi, ma, n_points) for mi,ma in zip(min_x, max_x)]
                r = np.stack(r, -1)

                # gumbel sampling
                g = -np.log(-np.log(np.random.uniform(size=r.shape)))
                ind = np.argmax(logp + g, 1)
                s = np.take_along_axis(r, ind[:,None], 1)
                x.append(s.reshape(-1))

            x = np.stack(x, -1)
            x = np.round(x, decimals)
        else:
            x = np.round(np.random.uniform(min_x, max_x, (n_points, n_vars)), decimals)
    else:
        x = np.array(supportPoints, dtype=np.float64)

    ref_dict.update({f'x{i+1}': x[:, i] for i in range(n_vars)})
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        y = np.round(eval(eq, ref_dict), decimals)
        
        # Case of "eq = C".
        if np.isscalar(y):
            y = np.full(n_points, y, dtype=y.dtype)

    return x, y

'''
This function samples an equation f(x) from a skeleton and 
computes y=f(x) for variables x sampled from the specififed range.
It returns (x, y, equation) if y is valid (eg, no inf nor nan).
Otherwise it returns (None, None, None).
'''
def generateEquationDatasetFromSkeleton(
    skeletonEqn,
    nv,
    const_range,
    xRange,
    numberofPoints,
    supportPoints,
    decimals,
    threshold,
    force_threshold,
    maxTrials,
    ):
    numC = skeletonEqn.count('C')
    for _ in range(maxTrials):
        # replace the constants with randam values
        Cs = np.random.uniform(const_range[0], const_range[1], size=numC)
        # replace: C -> ({}), Ce+5 -> ({}e+5)
        cleanEqn = re.sub(r'C(e[+-]\d+)?', r'({}\1)', skeletonEqn)
        cleanEqn = cleanEqn.format(*tuple(Cs.tolist()))

        # generate a set of points
        nPoints = np.random.randint(*numberofPoints) \
            if supportPoints is None else len(supportPoints)

        try:
            x, y = generateXYDataFromEquation(cleanEqn, n_points=nPoints, n_vars=nv,
                                    decimals=decimals, supportPoints=supportPoints, 
                                    min_x=xRange[0], max_x=xRange[1])
        
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            # retry
            continue

        if np.isnan(y).any() or np.isinf(y).any(): # TODO: Later find a more optimized solution
            # retry
            # print('nan inf retry: ', cleanEqn)
            continue

        # replace out of threshold with maximum numbers
        if force_threshold and threshold > 0:
            y = np.clip(y, -threshold, threshold)

        elif not force_threshold and threshold > 0:
            if (abs(y) > threshold).any():
                # print('threshold retry: ', cleanEqn)
                continue

        if len(y) == 0: # if for whatever reason the y is empty
            print('Empty y, x: {}, most of the time this is because of wrong numberofPoints: {}'.format(x, numberofPoints))
            # retry
            continue

        return x.tolist(), y.tolist(), cleanEqn
        
    return None, None, None

def processData(seed, numSamples, nv, decimals, 
                template, dataPath, fileID, time, 
                supportPoints=None, 
                supportPointsTest=None,
                numberofPoints=[20,250],
                xRange=[0.1,3.1], 
                testPoints=False,
                testRange=[0.0,6.0], 
                n_levels=3,
                allow_constants=True, 
                const_range=[-0.4, 0.4],
                const_ratio=0.8,
                op_list=[
                    "id", "add", "mul", "div", 
                    "sqrt", "sin", "exp", "log"],
                sortY=False,
                exponents=[3,4,5,6],
                numSamplesEachEq=1,
                threshold=100,
                force_threshold=True,
                templatesEQs=None,
                templateProb=0.4,
                ):

    random.seed(seed)
    np.random.seed(seed)

    sampleCount = 0
    for i in tqdm(range(numSamples*numSamplesEachEq)):
        structure = template.copy()

        # At each iteration, we make sure an equation instance is generated from a skeleton. 
        # If we fail to make an equation with a skeleton, we refresh the skeleton and repeat
        # the iteration without incrementing i. The same skeleton is repeatedly reused 
        # (numSamplesEachEq times) unless it shows issues such as nan and inf.
        succsess = False
        while not succsess:
            # generate a new skeleton if we created numSamplesEachEq of equation samples from the previous skeleton.
            if sampleCount % numSamplesEachEq == 0:
                sampleCount = 0
                try:
                    _, skeletonEqn, _ = dataGen(
                        nv=nv, 
                        decimals=decimals, 
                        numberofPoints=numberofPoints, 
                        supportPoints=supportPoints,
                        supportPointsTest=supportPointsTest,
                        xRange=xRange,
                        testPoints=testPoints,
                        testRange=testRange,
                        n_levels=n_levels,
                        op_list=op_list,
                        allow_constants=allow_constants, 
                        const_range=const_range,
                        const_ratio=const_ratio,
                        exponents=exponents
                        )
                    if templatesEQs != None and np.random.rand() < templateProb: 
                        # by a chance, replace the skeletonEqn with a given templates
                        idx = np.random.randint(len(templatesEQs[nv]))
                        skeletonEqn = templatesEQs[nv][idx]

                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    # retry to generate another skeleton
                    print("\n-->dataGen(.) was terminated!\n{}\n".format(e))
                    continue

                if 'I' in skeletonEqn or 'zoo' in skeletonEqn:
                    # retry to generate another skeleton
                    print('Skeleton rejected: {}'.format(skeletonEqn))
                    continue
                
                # fix exponents that are larger than our expected value, sometimes the data generator generates those odd numbers
                exps = re.findall(r"(\*\*[0-9\.]+)", skeletonEqn)
                for ex in exps:
                    # correct the exponent
                    cexp = '**'+str(eval(ex[2:]) if eval(ex[2:]) < exponents[-1] else np.random.randint(2,exponents[-1]+1))
                    # replace the exponent
                    skeletonEqn = skeletonEqn.replace(ex, cexp)

            eq_params = (nv, const_range, xRange, numberofPoints, supportPoints, decimals, threshold, force_threshold)
            x, y, cleanEqn = generateEquationDatasetFromSkeleton(skeletonEqn, *eq_params, maxTrials=10)
            
            if x is None:
                print('Failed to make valid equations. Skeleton rejected: {}'.format(skeletonEqn))
                    # refresh the invalid skeleton and retry the iteration.
                sampleCount = 0
                continue

            # sort data based on Y
            if sortY:
                x, y = zip(*sorted(zip(x,y), key=lambda d: d[1]))

            # hold data in the structure
            structure['X'] = x
            structure['Y'] = y
            structure['EQ'] = cleanEqn
            structure['Skeleton'] = skeletonEqn

            if testPoints:
                eq_params = (nv, const_range, testRange, numberofPoints, \
                    supportPointsTest, decimals, threshold, force_threshold)
                xT, yT, _ = generateEquationDatasetFromSkeleton(cleanEqn, *eq_params, maxTrials=10)

                if xT is None:
                    print('Failed to make valid test equations. Skeleton rejected: {}'.format(skeletonEqn))
                    # refresh the invalid skeleton and retry the iteration.
                    sampleCount = 0
                    continue

                if sortY:
                    xT, yT = zip(*sorted(zip(xT,yT), key=lambda d: d[1]))
                structure['XT'] = xT
                structure['YT'] = yT

            outputPath = dataPath.format(fileID, nv, time)
            if os.path.exists(outputPath):
                fileSize = os.path.getsize(outputPath)
                if fileSize > 500000000:  # 500 MB
                    fileID += 1

            with open(outputPath, "a", encoding="utf-8") as h:
                json.dump(structure, h, ensure_ascii=False)
                h.write('\n')

            # successfully created an equation instance.
            sampleCount += 1
            succsess = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", required=True, type=str, help="json file for setting-up default params")
    parser.add_argument('--seed', default=2021, type=int, help="")
    parser.add_argument('--numVars', nargs="+", default=[1,2,3,4,5,6,7,8,9], type=int, help="")
    parser.add_argument('--numberofPoints', nargs="+", default=[20, 250], type=int, help="")
    parser.add_argument('--numSamples', default=10000, type=int, help="")
    parser.add_argument('--testPoints', default=False, type=strtobool, help="")
    parser.add_argument('--folder', default='./Dataset', type=str, help="")
    parser.add_argument('--const_range', nargs="+", default=[-2.1, 2.1], type=float, help="")
    parser.add_argument('--numSamplesEachEq', default=5, type=int, help="")
    parser.add_argument('--threshold', default=5000, type=int, help="")
    parser.add_argument('--force_threshold', default=True, type=strtobool, help="")
    parser.add_argument('--parallel', action='store_true', help="")

    # load the json config and use it as default values.
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = json.load(file)
    default_values = {key: config[key] for key in vars(args) if key in config}
    parser.set_defaults(**default_values)

    # parse the args again with the defaults specified by config
    args = parser.parse_args()
    for key in vars(args):
        config[key] = getattr(args, key)
    print(config)

    #NOTE: For linux you can only use unique numVars, in Windows, it is possible to use [1,2,3,4] * 10!
    numVars = config['numVars']
    decimals = config['decimals']
    numberofPoints = config['numberofPoints']
    numSamples = config['numSamples']
    folder = config['folder']
    dataPath = os.path.join(folder, '{}_{}_{}.json')

    testPoints = config['testPoints']
    trainRange = config['trainRange']
    testRange = config['testRange']

    supportPoints = config['supportPoints']
    supportPointsTest = config['supportPointsTest']
    n_levels = config['n_levels']
    allow_constants = config['allow_constants']
    const_range = config['const_range']
    const_ratio = config['const_ratio']
    op_list = config['op_list']
    exponents = config['exponents']

    sortY = config['sortY']
    numSamplesEachEq = config['numSamplesEachEq']
    threshold = config['threshold']
    force_threshold = config['force_threshold']
    templateProb = config['templateProb'] # the probability of generating an equation from the templates
    templatesEQs = config['templatesEQs']
    
    print(os.makedirs(folder) if not os.path.isdir(folder) \
        else 'We do have the path already!')

    template = {'X':[], 'Y':0.0, 'EQ':''}
    fileID = 0
    now = datetime.now()

    # For reproducibility, set the seed for the main thread
    # and then set a modified seed inside each thread.
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)

    if args.parallel:
        Parallel(n_jobs=len(numVars))(
            delayed(processData)(
                seed*100+i,
                numSamples, nv, decimals, template, 
                dataPath, fileID,
                '{}_'.format(i) + now.strftime("%d%m%Y_%H%M%S"),
                supportPoints, 
                supportPointsTest,
                numberofPoints,
                trainRange, testPoints, testRange, n_levels, 
                allow_constants, const_range,
                const_ratio, op_list, sortY, exponents,
                numSamplesEachEq,
                threshold,
                force_threshold,
                templatesEQs,
                templateProb
            ) for i,nv in enumerate(numVars)
        )
    else:
        # TODO: workaround for avoiding freezing when running by threads 
        for i,nv in enumerate(numVars):
            processData(
                seed*100+i,
                numSamples, nv, decimals, template, 
                dataPath, fileID,
                '{}_'.format(i) + now.strftime("%d%m%Y_%H%M%S"),
                supportPoints, 
                supportPointsTest,
                numberofPoints,
                trainRange, testPoints, testRange, n_levels, 
                allow_constants, const_range,
                const_ratio, op_list, sortY, exponents,
                numSamplesEachEq,
                threshold,
                force_threshold,
                templatesEQs,
                templateProb
            )

if __name__ == '__main__':
    main()
