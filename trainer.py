"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import os

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from utils import evaluate
import warnings

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_samples = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_samples = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    save_every_epoch = True

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, device='gpu', collate_fn=None, tester=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn
        self.config = config
        self.tester = tester

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if device in ('cuda','gpu') and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            print('We are using the gpu now! device={}'.format(self.device))

        self.best_loss = None

    def save_checkpoint(self, name):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        file = os.path.join(self.config.ckpt_path, name)
        logger.info("saving %s", file)
        torch.save(raw_model.state_dict(), file)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                                batch_size=config.batch_size,
                                collate_fn=self.collate_fn,
                                num_workers=config.num_workers,
                                drop_last=is_train)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, p, ind, v) in pbar:

                # place data on the correct device
                x = x.to(self.device) # input equation
                y = y.to(self.device) # output equation
                v = v.to(self.device) # number of variables
                p = p.to(self.device) # points with indices
                ind = ind.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y, p, ind, v, tokenizer=self.train_dataset.itos)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.samples += y.shape[0] # number of samples processed this step
                        if self.samples < config.warmup_samples:
                            # linear warmup
                            lr_mult = float(self.samples) / float(max(1, config.warmup_samples))
                        else:
                            # cosine learning rate decay
                            progress = float(self.samples - config.warmup_samples) / float(max(1, config.final_samples - config.warmup_samples))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        self.best_loss = float('inf') if self.best_loss is None else self.best_loss
        self.samples = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            self.save_checkpoint('latest.pt')
            if self.config.save_every_epoch:
                self.save_checkpoint(f'epoch{epoch+1:03d}.pt')

            if self.test_dataset is not None:
                if self.tester is not None:
                    loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                                        batch_size=1,
                                        collate_fn=self.collate_fn,
                                        num_workers=self.config.num_workers,
                                        drop_last=False)
                    test_loss = self.tester(raw_model, loader, self.device)
                else:
                    test_loss = run_epoch('test')
                logger.info("test loss: %f", test_loss)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < self.best_loss
            if self.config.ckpt_path is not None and good_model:
                self.best_loss = test_loss
                self.save_checkpoint('best.pt')
