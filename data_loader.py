
from tqdm import tqdm
from torch.utils.data import Dataset
import json
import random
import re
import numpy as np
import torch
from utils import generateXYDataFromEquation
from joblib import Parallel, delayed

def processDataFiles(files):
    def loader(entry):
        try:
            entry = json.loads(entry)
            # to save memory, hold arrays as float32 tensors instead of float64.
            for key in entry:
                if isinstance(entry[key], list):
                    entry[key] = torch.tensor(entry[key], dtype=torch.float32)
        except Exception as e:
            print("Couldn't convert to json: {} \n error is: {}".format(entry, e))
            return None
        return entry

    result = []
    for f in tqdm(files):
        with open(f, 'r') as h: 
            lines = h.readlines()
            if len(lines[-1]) <= 1:
                lines = lines[:-1]

            entries = Parallel(n_jobs=-1)([delayed(loader)(line) for line in lines])
            del lines
            result.extend([e for e in entries if e is not None])
            
    return result

def collate_fn(batch, padding_id=None):
    inputs, outputs, points, numVars = list(zip(*batch))
    pad_seq = torch.nn.utils.rnn.pad_sequence
    
    if padding_id is None:
        padding_id = inputs[0][-1]
    inputs = pad_seq(inputs, batch_first=True, padding_value=padding_id)
    outputs = pad_seq(outputs, batch_first=True, padding_value=padding_id)

    # points is a list of [(L, dim)] * B.
    # Make a batch of shape (L1+L2+...+LN, dim) with index.
    indices = [torch.full((p.shape[0],) ,i,dtype=torch.long) for i,p in enumerate(points)]
    indices = torch.cat(indices,dim=0)
    points = torch.cat(points, dim=0)

    numVars = torch.stack(numVars)
    return inputs, outputs, points, indices, numVars

class CharDataset(Dataset):
    def __init__(self, data, block_size, symbols, 
                 numVars, numYs, target='EQ', 
                 addVars=False, augment=False, numPoints=[20,200], 
                 const_range=[-0.4, 0.4],
                 xRange=[-3.0,3.0], decimals=4, ):

        # initialize chars set if not defined
        if len(symbols) == 0:
            text = ''.join([chunk[target] for chunk in data] + ['_T<>:-.0123456789'])
            symbols += sorted(list(set(text)))
            print('Updated chars set:', symbols)

        data_size, vocab_size = len(data), len(symbols)
        print('data has %d examples, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(symbols) }
        self.itos = { i:ch for i,ch in enumerate(symbols) }

        self.numVars = numVars
        self.numYs = numYs
        
        # padding token
        self.paddingToken = '_'
        self.paddingID = self.stoi[self.paddingToken]
        self.stoi[self.paddingToken] = self.paddingID
        self.itos[self.paddingID] = self.paddingToken
        self.threshold = 5000
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data # it should be a list of examples
        self.target = target
        self.addVars = addVars

        # settings only for augmentation
        self.augment = augment
        self.numPoints = numPoints
        self.const_range = const_range
        self.xRange = xRange
        self.decimals = decimals
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx] # sequence of tokens including x, y, eq, etc.
        
        # find the number of variables in the equation
        printInfoCondition = random.random() < 0.0000001
        eq = chunk[self.target]

        if printInfoCondition:
            print(f'\nEquation: {eq}')

        if self.target == 'Skeleton' and self.augment:
            Xy = self.sample_from_skelton(chunk['Skeleton'])
            if Xy is not None:
                chunk['X'], chunk['Y'] = Xy

        # extract points from the input sequence as a (L, dim) shape tensor.
        # zero padding is done later in collate function.
        X, Y = chunk['X'], chunk['Y']
        if not torch.is_tensor(X) or not torch.is_tensor(Y):
            X = torch.tensor(X, dtype=torch.float32)
            Y = torch.tensor(Y, dtype=torch.float32)
        X = X.reshape(-1, 1) if X.dim() == 1 else X
        Y = Y.reshape(-1, 1) if Y.dim() == 1 else Y
        numPoints = max(X.shape[0], Y.shape[0])
        points = torch.zeros(numPoints, self.numVars+self.numYs)
        x_len = min(X.shape[0], numPoints)
        x_dim = min(X.shape[1], self.numVars)
        y_len = min(Y.shape[0], numPoints)
        y_dim = min(Y.shape[1], self.numYs)
        points[:x_len, :x_dim] = X[:x_len, :x_dim]
        points[:y_len, self.numVars:self.numVars+y_dim] = Y[:y_len, :y_dim]
        points = torch.nan_to_num(points, nan=self.threshold, 
                                        posinf=self.threshold, 
                                        neginf=-self.threshold)

        # The definition of numVars differs from the original code.
        # For a skeleton of "C*x1" with global numVars=2 (ie, there is unused x2),
        # the original code sets numVars=1 but it leaks some hidden info.
        # Instead, we set numVars=2.
        numVars = X.shape[1]

        # encode every character in the equation to an integer
        # < is SOS, > is EOS
        EOS = '>'
        if self.addVars:
            dix = [self.stoi[s] for s in '<'+str(numVars)+':'+eq+EOS]
        else:
            dix = [self.stoi[s] for s in '<'+eq+EOS]

        # If input is  ">ABCDE" then we expect the transformer's output to be "ABCDE>". 
        inputs = dix[:-1]
        outputs = dix[1:]
        
        if self.block_size > 0 and len(inputs) > self.block_size:
            inputs = inputs[:self.block_size]
            outputs = outputs[:self.block_size]
        
        inputs  = torch.tensor(inputs,  dtype=torch.long)
        outputs = torch.tensor(outputs, dtype=torch.long)
        numVars = torch.tensor(numVars, dtype=torch.long)

        return inputs, outputs, points, numVars

    def sample_from_skelton(self, eq):
        # randomly generate the constants
        cleanEqn = ''
        for chr in eq:
            if chr == 'C':
                # genereate a new random number
                chr = '({})'.format(np.random.uniform(self.const_range[0], self.const_range[1]))
            cleanEqn += chr

        # update the points
        nPoints = np.random.randint(*self.numPoints) #if supportPoints is None else len(supportPoints)
        try:
            X, y = generateXYDataFromEquation(
                cleanEqn, n_points=nPoints, n_vars=self.numVars,
                decimals=self.decimals, min_x=self.xRange[0], max_x=self.xRange[1])

            # replace out of threshold with maximum numbers
            y = np.clip(y, -self.threshold, self.threshold)

            # check if there is nan/inf/very large numbers in the y
            conditions = (np.isnan(y).any() or np.isinf(y).any()) or len(y) == 0
            if conditions:
                return None

        except Exception as e: 
            # for different reason this might happend including but not limited to division by zero
            print("".join([
                f"We just used the original equation and support points because of {e}. ",
                f"The equation is {eq}, and we update the equation to {cleanEqn}",
            ]))
            return None

        return X, y
