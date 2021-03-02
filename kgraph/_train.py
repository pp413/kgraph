#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2021/02/27 23:33:40
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
import os, arrow
import numpy as np
import torch
from tqdm import tqdm
from torch.optim import SGD, Adagrad, Adam
from .eval import calculate_ranks
from .eval import calculate_n2n_ranks

def save_checkpoint(model, epoch=None, dataname=None):
    root = os.getcwd()
    
    filename = f'{model.__class__.__name__}'
    filename += f'_{epoch}' if epoch is not None else ''
    filename += f'_{dataname}' if dataname is not None else ''
    filename += '.tgz'
    
    path = os.path.join(root, f'{model.__class__.__name__}', 'checkpoint')
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, filename))


class Opt:
    def __init__(self, model, opt=None, lr=1e-4, weight_decay=0, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v
        if isinstance(opt, str):
            if opt.lower() == 'adam':
                self.opt = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
            elif opt.lower() == 'sgd':
                self.opt = SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)
            elif opt.lower() == 'adagrad':
                self.opt = Adagrad(params=model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                print(f'{opt} is can\'t find')
        else:
            self.opt = opt(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    
    def __call__(self):
        return self.opt


class TrainBase():
    
    def __init__(self, DataLoader, model, opt='Adam', lr=1e-4, weight_decay=0,
                 scheduler_step=None, scheduler_gamme=0.75, device='cpu'):
        self.dataloader = DataLoader
        self.__loss = None
        self.num_batch = DataLoader.num_batchs
        self.device = device
        self.model = model.to(self.device)
        print(device)
        self.opt = Opt(self.model, opt=opt, lr=lr, weight_decay=weight_decay)()
        
        self.use_scheduler = False if scheduler_step is None else True
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt,
                                step_size=scheduler_step, gamme=scheduler_gamme)
    
    def loss(self, *data):
        loss = self.model.loss(*data) if self.__loss is None else self.__loss(*data)
        return loss
    
    def train_step(self, *data):
        data = [x.long().to(self.device) for x in data]
        loss = self.loss(*data)
        loss = loss.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
    
    def train_iter_per_epoch(self):
        
        try:
            num_batchs = self.dataloader.num_batchs
            with tqdm(range(num_batchs), ncols=100) as bar:
                avg_loss = []
                for _, batch_data in zip(bar, self.dataloader):
                    epoch, loss = yield batch_data
                    bar.set_description(desc=f'Epoch {epoch+1:<3d}')
                    avg_loss.append(loss)
                    bar.set_postfix(BatchLoss=f'{loss:<.4f}', AvgLoss=f'{np.mean(avg_loss):<.4f}')
        
        except KeyboardInterrupt:
            bar.close()
            raise

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


def fit(model, data_iter, num_epoch, opt='adam', lr=1e-4, weight_decay=0, scheduler_step=None,
        scheduler_gamme=0.75, valid_step=None, valid_function=None, device='cpu'):
    model.device = device
    process = Train(data_iter, model, opt, lr, weight_decay, scheduler_step, scheduler_gamme, device)
    process.fit(num_epoch, None, valid_step, valid_function)
    
    model_name = model.__class__.__name__
    dataset_name = data_iter.name
    now = arrow.now().format('YYYY-MM-DD')
    save_path = os.path.join(os.getcwd(), model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f'{dataset_name}_{now}.tgz'
    model.save_checkpoint(os.path.join(save_path, file_name))


def initial_graph_model(data_iter):
    
    def initial(cls):
        
        def add_link_prediction(self, batch_size=None, for_test=True, constraint=None):
            table = calculate_ranks(self.predict, data_iter.dataset, for_test=for_test,
                                batch_size=batch_size, device=self.device, constraint=constraint)
            return table
        
        def add_link_n2n_prediction(self, batch_size=None, constraint=None):
            table = calculate_n2n_ranks(self.predict, data_iter.dataset, batch_size, self.device, constraint)
            return table
        
        @property
        def name(self):
            return self.__class__.__name__
        
        @property
        def device(self):
            return self._device
        
        @device.setter
        def device(self, d='cpu'):
            self.to(d)
            self._device = d
        
        def add_fit(self, num_epoch, opt='adam', lr=1e-4, weight_decay=0, scheduler_step=None,
                    scheduler_gamme=0.75, valid_step=None, valid_function=None, device='cpu'):
            fit(self, data_iter,num_epoch, opt=opt, lr=lr, weight_decay=weight_decay, scheduler_step=scheduler_step,
                scheduler_gamme=scheduler_gamme, valid_step=valid_step, valid_function=valid_function, device=device)
        
        def save_checkpoint(self, path):
            torch.save(self.state_dict(), path)
        
        def load_checkpoint(self, path):
            self.load_state_dict(torch.load(os.path.join(path)))
            self.eval()
        
        def pred_train_from(self, path):
            self.load_checkpoint(path)
            self.train()
        
        cls._device = 'cpu'
        cls.device = device
        cls.link_prediction = add_link_prediction
        cls.link_n2n_prediction = add_link_n2n_prediction
        cls.name = name
        cls.fit = add_fit
        cls.save_checkpoint = save_checkpoint
        cls.load_checkpoint = load_checkpoint
        cls.pred_train_from = pred_train_from
        return cls
    return initial
    
