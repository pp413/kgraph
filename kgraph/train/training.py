#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/26 17:57:27
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
import torch as th
import numpy as np
from functools import wraps
from tqdm import trange, tqdm
from .opt import Opt
from ..log.log import save_checkpoint


class TrainBase():
    
    def __init__(self, DataLoader, model, opt='Adam', lr=1e-4, weight_decay=0,
                 scheduler_step=None, scheduler_gamme=0.75, device='cpu'):
        self.dataloader = DataLoader
        self.__loss = None
        self.num_batch = DataLoader.num_batch
        self.device = device
        self.model = model.to(self.device)
        self.opt = Opt(self.model, opt=opt, lr=lr, weight_decay=weight_decay)()
        
        self.use_scheduler = False if scheduler_step is None else True
        if self.use_scheduler:
            self.scheduler = th.optim.lr_scheduler.StepLR(self.opt,
                                step_size=scheduler_step, gamme=scheduler_gamme)
    
    def loss(self, *data):
        loss = self.model.loss(*data) if self.__loss is None else self.__loss(*data)
        return loss
    
    def train_step(self, *data):
        lhs_pos, rhs_pos, lhs_neg, rhg_neg = [th.from_numpy(x).long().to(self.device) for x in data]
        loss = self.loss(lhs_pos, rhs_pos, lhs_neg, rhg_neg)
        loss = loss.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
    
    def train_iter_per_epoch(self):
        
        try:
            with tqdm(self.dataloader(), ncols=100) as bar:
                avg_loss = []
                for batch_data in bar:
                    epoch, loss = yield batch_data
                    bar.set_description(desc=f'Epoch {epoch+1:<3d}')
                    avg_loss.append(loss)
                    bar.set_postfix(BatchLoss=f'{loss:<.4f}', AvgLoss=f'{np.mean(avg_loss):<.4f}')
        
        except KeyboardInterrupt:
            bar.close()
            raise
        bar.close()

    def __train_iter(self, num_epoch, valid_step=None, valid_function=None):
        tmp = -1
        best_i = -1
        for i in range(num_epoch):
            train_consume = self.train_iter_per_epoch()
            data = next(train_consume)
            for _ in range(self.num_batch):
                try:
                    data = train_consume.send((i, self.train_step(*data)))
                except StopIteration:
                    break
            if self.use_scheduler:
                self.scheduler.step()
            if valid_step is not None and valid_function is not None and (i + 1) % valid_step == 0:
                H_at_10 = valid_function()
                save_checkpoint(self.model, epoch=i, dataname=self.dataloader.name)
                if H_at_10 > tmp:
                    best_i = i
                    save_checkpoint(self.model, dataname=self.dataloader.name)
                print(f'Save the model, and the best model in epoch {best_i}')
    
    def fit(self, num_epoch, loss_function=None, valid_step=None, valid_function=None):
        self.__loss = loss_function(self.model) if loss_function is not None else self.model.loss
        return self.__train_iter(num_epoch, valid_step=None, valid_function=None)
    

class Train(TrainBase):
    def set_loss_function(self, loss_function):
        self.__loss = loss_function(self.model)
        








