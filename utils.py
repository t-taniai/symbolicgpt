from functools import partial
import re
import json
import random
import torch
import numpy as np
from scipy.optimize import minimize
from torch.nn import functional as F
from numpy import * # to override the math functions
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import warnings


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

# @torch.no_grad()
# def sample_from_model(model, x, steps, points=None, variables=None, temperature=1.0, sample=False, top_k=None):
#     """
#     take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
#     the sequence, feeding the predictions back into the model each time. Clearly the sampling
#     has quadratic complexity unlike an RNN that is only linear, and has a finite context window
#     of block_size, unlike an RNN that has an infinite context window.
#     """
#     block_size = model.get_block_size()
#     model.eval()
#     for k in range(steps):
#         x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
#         logits, _ = model(x_cond, points=points, variables=variables)
#         # pluck the logits at the final step and scale by temperature
#         logits = logits[:, -1, :] / temperature
#         # optionally crop probabilities to only the top k options
#         if top_k is not None:
#             logits = top_k_logits(logits, top_k)
#         # apply softmax to convert to probabilities
#         probs = F.softmax(logits, dim=-1)
#         # sample from the distribution or take the most likely
#         if sample:
#             ix = torch.multinomial(probs, num_samples=1)
#         else:
#             _, ix = torch.topk(probs, k=1, dim=-1)
#         # append to the sequence and continue
#         x = torch.cat((x, ix), dim=1)

#     return x

#use nucleus sampling from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering(logits, top_k=0.0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        
        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    #TODO: support for batch size more than 1
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

@torch.no_grad()
def sample_from_model(model, x, steps, points=None, index=None, variables=None, temperature=1.0, sample=False, top_k=0.0, top_p=0.0, is_finished=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        if is_finished is not None and is_finished(x):
            break

        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond, points=points, index=index, variables=variables)
        # pluck the logits at the final step and scale by temperature
        logits = logits[0, -1, :] / temperature
        # optionally crop probabilities to only the top k options
#         if top_k is not None:
#             logits = top_k_logits(logits, top_k)
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix.unsqueeze(0)), dim=1)

    return x

def plot_and_save_results(resultDict, fName, pconf, titleTemplate, textTest, modelKey='SymbolicGPT'):
    if isinstance(resultDict, dict):
        # plot the error frequency for model comparison
        num_eqns = len(resultDict[fName][modelKey]['err'])
        num_vars = pconf.numberofVars
        title = titleTemplate.format(num_eqns, num_vars)

        models = list(key for key in resultDict[fName].keys() if len(resultDict[fName][key]['err'])==num_eqns)
        lists_of_error_scores = [resultDict[fName][key]['err'] for key in models if len(resultDict[fName][key]['err'])==num_eqns]
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

        name = '{}.png'.format(fName.split('.txt')[0])
        plt.savefig(name)

        with open(fName, 'w', encoding="utf-8") as o:
            for i in range(num_eqns):
                err = resultDict[fName][modelKey]['err'][i]
                eq = resultDict[fName][modelKey]['trg'][i]
                predicted = resultDict[fName][modelKey]['prd'][i]
                print('Test Case {}.'.format(i))
                print('Target:{}\nSkeleton:{}'.format(eq, predicted))
                print('Err:{}'.format(err))
                print('') # just an empty line

            
                o.write('Test Case {}/{}.\n'.format(i,len(textTest)-1))

                o.write('{}\n'.format(eq))
                o.write('{}:\n'.format(modelKey))
                o.write('{}\n'.format(predicted))

                o.write('{}\n{}\n\n'.format( 
                                        predicted,
                                        err
                                        ))

                print('Avg Err:{}'.format(np.mean(resultDict[fName][modelKey]['err'])))

def tokenize_predict_and_evaluate(i, inputs, points, outputs, variables, 
                                  train_dataset, textTest, trainer, model, resultDict,
                                  numTests, variableEmbedding, blockSize, fName,
                                  modelKey='SymbolicGPT', device='cpu'):
    
    eq = ''.join([train_dataset.itos[int(i)] for i in outputs[0]])
    eq = eq.strip(train_dataset.paddingToken).split('>')
    eq = eq[0] #if len(eq[0])>=1 else eq[1]
    eq = eq.strip('<').strip(">")
    print(eq)
    if variableEmbedding == 'STR_VAR':
            eq = eq.split(':')[-1]

    t = json.loads(textTest[i])

    inputs = inputs[:,0:1].to(device)
    points = points.to(device)
    # points = points[:,:numPoints] # filter anything more than maximum number of points
    variables = variables.to(device)

    bestErr = 10000000
    bestPredicted = 'C'
    for i in range(numTests):
        
        predicted, err = generate_sample_and_evaluate(
                            model, t, eq, inputs, 
                            blockSize, points, variables, 
                            train_dataset, variableEmbedding)

        if err < bestErr:
            bestErr = err
            bestPredicted = predicted
    
    resultDict[fName][modelKey]['err'].append(bestErr)
    resultDict[fName][modelKey]['trg'].append(eq)
    resultDict[fName][modelKey]['prd'].append(bestPredicted)

    return eq, bestPredicted, bestErr

def generate_sample_and_evaluate(model, t, eq, inputs, 
                                 blockSize, points, variables, 
                                 train_dataset, variableEmbedding):

    
    outputsHat = sample_from_model(model, 
                        inputs, 
                        blockSize, 
                        points=points,
                        variables=variables,
                        temperature=0.9, 
                        sample=True, 
                        top_k=40,
                        top_p=0.7,
                        )[0]

    # filter out predicted
    predicted = ''.join([train_dataset.itos[int(i)] for i in outputsHat])

    if variableEmbedding == 'STR_VAR':
        predicted = predicted.split(':')[-1]

    predicted = predicted.strip(train_dataset.paddingToken).split('>')
    predicted = predicted[0] #if len(predicted[0])>=1 else predicted[1]
    predicted = predicted.strip('<').strip(">")
    predicted = predicted.replace('Ce','C*e')

    # train a regressor to find the constants (too slow)
    c = [1.0 for i,x in enumerate(predicted) if x=='C'] # initialize coefficients as 1
    # c[-1] = 0 # initialize the constant as zero
    b = [(-2,2) for i,x in enumerate(predicted) if x=='C']  # bounds on variables
    try:
        if len(c) != 0:
            # This is the bottleneck in our algorithm
            # for easier comparison, we are using minimize package  
            cHat = minimize(lossFunc, c, #bounds=b,
                        args=(predicted, t['X'], t['Y'])) 

            predicted = predicted.replace('C','{}').format(*cHat.x)
    except ValueError:
        raise 'Err: Wrong Equation {}'.format(predicted)
    except Exception as e:
        raise 'Err: Wrong Equation {}, Err: {}'.format(predicted, e)
    
    Ys = [] #t['YT']
    Yhats = []
    for xs in t['XT']:
        try:
            eqTmp = eq + '' # copy eq
            eqTmp = eqTmp.replace(' ','')
            eqTmp = eqTmp.replace('\n','')
            for i,x in enumerate(xs):
                # replace xi with the value in the eq
                eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                if ',' in eqTmp:
                    assert 'There is a , in the equation!'
            YEval = eval(eqTmp)
            # YEval = 0 if np.isnan(YEval) else YEval
            # YEval = 100 if np.isinf(YEval) else YEval
        except:
            print('TA: For some reason, we used the default value. Eq:{}'.format(eqTmp))
            print(i)
            raise
            continue # if there is any point in the target equation that has any problem, ignore it
            YEval = 100 #TODO: Maybe I have to punish the model for each wrong template not for each point
        Ys.append(YEval)
        try:
            eqTmp = predicted + '' # copy eq
            eqTmp = eqTmp.replace(' ','')
            eqTmp = eqTmp.replace('\n','')
            for i,x in enumerate(xs):
                # replace xi with the value in the eq
                eqTmp = eqTmp.replace('x{}'.format(i+1), str(x))
                if ',' in eqTmp:
                    assert 'There is a , in the equation!'
            Yhat = eval(eqTmp)
            # Yhat = 0 if np.isnan(Yhat) else Yhat
            # Yhat = 100 if np.isinf(Yhat) else Yhat
        except:
            print('PR: For some reason, we used the default value. Eq:{}'.format(eqTmp))
            Yhat = 100
        Yhats.append(Yhat)
    err = relativeErr(Ys,Yhats, info=True)
    
    print('\nTarget:{}'.format(eq))
    print('Skeleton+LS:{}'.format(predicted))
    print('Err:{}'.format(err))
    print('-'*10)

    if type(err) is np.complex128 or np.complex:
        err = abs(err.real)

    return predicted, err

# helper class and functions
# add a safe wrapper for numpy math functions

def divide(x, y):
  x = np.nan_to_num(x)
  y = np.nan_to_num(y)
  return np.divide(x,max(y,1.0))

def sqrt(x):
  x = np.nan_to_num(x)
  x = np.abs(x)
  return np.sqrt(x) 

def log(x, eps=1e-5):
  x = np.nan_to_num(x)
  x = np.sqrt(x*x+eps)
  return np.log(x)

def exp(x, eps=1e-5):
    x = np.nan_to_num(x)
    #x = np.minimum(x,5) # to avoid overflow
    return np.exp(x)

# Mean square error
def mse(y, yHat):
    y = y.reshape(-1)
    yHat = yHat.reshape(-1)
    if len(y) > 0 and len(y)==len(yHat):
        err = np.mean((yHat - y)**2)
    else:
        err = float('inf')

    return err

# Relative Mean Square Error
def relativeErr(y, yHat, eps=1e-5):
    y, yHat = y.reshape(-1), yHat.reshape(-1)
    if len(y) > 0 and len(y)==len(yHat):
        err = ((yHat - y)**2).mean() / (np.linalg.norm(y)+eps)
    else:
        err = float('inf')

    return err

def normScaledMSE(y, yHat, eps=1e-5):
    y, yHat = y.reshape(-1), yHat.reshape(-1)
    if len(y) > 0 and len(y)==len(yHat):
        err = sum((yHat - y)**2) / (sum(y**2)+eps)
    else:
        err = float('inf')

    return err

def varScaledMSE(y, yHat, eps=1e-5):
    y, yHat = y.reshape(-1), yHat.reshape(-1)
    if len(y) > 0 and len(y)==len(yHat):
        err = sum((yHat - y)**2) / (sum((y-y.mean())**2)+eps)
    else:
        err = float('inf')

    return err


def lossFunc(constants, eq, X, Y, errorFunc=relativeErr, imagPenalty=0):
    yHat = evalFunc(eq, X, constants, allowComplex=imagPenalty>0)
    err = errorFunc(Y, yHat)
    if imagPenalty > 0:
        err = np.real(err) + imagPenalty*np.imag(err)
    return err

def replaceConstantsByNumbered(eq):
    eq = eq.replace('C', '{}').format(*(f'C{i+1}' for i in range(eq.count('C'))))
    eq = eq.replace('D', '{}').format(*(f'D{i+1}' for i in range(eq.count('D'))))
    eq = eq.replace('F', '{}').format(*(f'F{i+1}' for i in range(eq.count('F'))))
    return eq


def substituteConstants(constants, eq):
    eq = eq + ''
    num_C = eq.count('C')
    num_F = eq.count('F')
    num_D = eq.count('D')
    C = constants[:num_C]
    D = constants[num_C:num_C+num_D]
    F = constants[num_C+num_D:num_C+num_D+num_F]
    if len(C) > 0:
        eq = re.sub('C[0-9]?', '({})', eq).format(*C)
    if len(D) > 0:
        D = [10.0**d for d in D]
        eq = re.sub('D[0-9]?', '({})', eq).format(*D)
    if len(F) > 0:
        F = [exp(f) for f in F]
        eq = re.sub('F[0-9]?', '({})', eq).format(*F)
    return eq

def evalFunc(eq, X, c=None, verbose=False, allowComplex=False):
    # make dicts for eval func.
    # transpose X as it its shape is (L, dim).
    cp = 0j if allowComplex else 0
    var_dict = {f'x{i+1}':(x+cp) for i,x in enumerate(X.T)}
    myabs = lambda x: np.abs(np.real(x))+1j*np.abs(np.imag(x))
    fun_dict = {
        'sin': np.sin,
        'cos': np.cos,
        'log': np.log if allowComplex else log,
        'exp': np.exp if allowComplex else exp,
        'abs': myabs if allowComplex else np.abs,
        'sqrt': np.sqrt if allowComplex else sqrt,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'divide': divide,
    }
    if c is not None and len(c)>0:
        num_C = eq.count('C')
        num_F = eq.count('F')
        num_D = eq.count('D')
        
        var_dict.update({f'C{i+1}':v+cp for i,v in enumerate(c[:num_C])})
        var_dict.update({f'D{i+1}':10**v+cp for i,v in enumerate(c[num_C:num_C+num_D])})
        var_dict.update({f'F{i+1}':exp(v)+cp for i,v in enumerate(c[num_C+num_D:num_C+num_D+num_F])})

    try:
        yHat = eval(eq, var_dict, fun_dict)
        # broadcast yHat if it is a single constant (ie, eq=C).
        if np.isscalar(yHat):
            yHat = np.full((X.shape[0], 1), yHat, dtype=X.dtype)

    except Exception as e:
        if verbose:
            print('Exception has been occured! EQ: {}'.format(eq))
        yHat = np.full((X.shape[0], 1), 1e5, dtype=X.dtype)
    return yHat


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


def evaluate(model, loader, device, printer=None, errorFunc=relativeErr, verbose=False):
    model.eval()
    variableEmbedding = model.pointNetConfig.varibleEmbedding
    dataset = loader.dataset
    end_token_id = dataset.stoi.get('>')
    is_finished = lambda x: (x[:,-1]==end_token_id).all()

    results = []
    pbar = tqdm(enumerate(loader), total=len(loader))
    for i, batch in pbar:
        inputs,outputs,points,index,variables = batch

        t = dataset.data[i]
        X = np.array(t['X'], dtype=np.float64)
        Y = np.array(t['Y'], dtype=np.float64)
        XT = np.array(t['XT'], dtype=np.float64)

        inputs = inputs[:,0:1].to(device)
        points = points.to(device)
        index = index.to(device)
        variables = variables.to(device)

        outputsHat = sample_from_model(
                    model, 
                    inputs, 
                    dataset.block_size, 
                    points=points,
                    index=index,
                    variables=variables,
                    temperature=1.0, 
                    sample=True, 
                    top_k=0.0,
                    top_p=0.7,
                    is_finished=is_finished)[0]

        # filter out predicted
        target = t['EQ']
        predicted = [dataset.itos[int(i)] for i in outputsHat]

        predicted = ''.join(predicted)
        if variableEmbedding == 'STR_VAR':
            #target = target.split(':')[-1]
            predicted = predicted.split(':')[-1]

        predicted = predicted.strip(dataset.paddingToken).split('>')
        predicted = predicted[0]
        predicted = predicted.strip('<').strip(">")

        # replace C, D, F with C1, D1, F1...
        predicted = replaceConstantsByNumbered(predicted)

        target = target.replace(' ','').replace('\n','')
        predicted = predicted.replace(' ','').replace('\n','')
        
        # train a regressor to find the constants (too slow)
        c = [1.0 for i,x in enumerate(predicted) if x=='C'] # initialize coefficients as 1
        c += [0.0 for i,x in enumerate(predicted) if x=='D']    # D: 10**x
        c += [0.0 for i,x in enumerate(predicted) if x=='F']    # F: exp(x)

        # c[-1] = 0 # initialize the constant as zero
        b = [(-2,2) for i,x in enumerate(predicted) if x=='C']  # bounds on variables
        b += [(0,32) for i,x in enumerate(predicted) if x=='D']  # bounds on variables
        b += [(-10,10) for i,x in enumerate(predicted) if x=='F']  # bounds on variables
        try:
            _lossFunc = partial(lossFunc, errorFunc=errorFunc)
            if len(c) != 0:
                cHat = minimize(_lossFunc, c, #bounds=b,
                            args=(predicted, X, Y)) 
                c = cHat.x
        except ValueError:
            raise 'Err: Wrong Equation {}'.format(predicted)
        except Exception as e:
            raise 'Err: Wrong Equation {}, Err: {}'.format(predicted, e)

        YT = evalFunc(target, XT)
        YThat = evalFunc(predicted, XT, c, verbose=verbose)
        err = errorFunc(YT, YThat)

        if type(err) is np.complex128 or complex:
            err = abs(err.real)
        
        newResult = {
            'index': i,
            'error': err,
            'target': target,
            'pred_sk': predicted,
            'pred_eq': substituteConstants(c, predicted)
        }
        results.append(newResult)
        if printer is not None:
            printer(newResult)
        
    return results

def calc_area_under_curve_score(scores, lb=1e-5, ub=1e+5, bins=1000):
    '''
    Compute a normalized cumulative histogram of log of scores and
    calculate the area under the curve (1: best, 0: worst).
    '''
    scores = torch.clamp(torch.tensor(scores), lb, ub)
    scores = torch.log10(scores)
    h = torch.histc(scores, bins, math.log10(lb), math.log10(ub)) / scores.numel()
    h = torch.cumsum(h, 0)
    return h.mean().item()

def calc_area_above_curve_score(scores, lb=1e-5, ub=1e+5, bins=1000):
    '''
    Compute a normalized cumulative histogram of log of scores and
    calculate the area above the curve (1: worst, 0: best).
    '''
    return 1 - calc_area_under_curve_score(scores, lb, ub, bins)

def validation_tester(model, loader, device, errorFunc=relativeErr):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        results = evaluate(model, loader, device, errorFunc=errorFunc)
        scores = [x['error'] for x in results]
    return calc_area_above_curve_score(scores)
