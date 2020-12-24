#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/27 17:11:57
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
import os
import torch
import torch.nn as nn

from abc import abstractmethod, abstractstaticmethod
from .datasets.data_loader import DataLoader
from .train.training import Train
from .eval.function import link_prediction
from .eval.function import link_n2n_prediction
from .eval.function import classification

class Module(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(Module, self).__init__()
        self.__temp = nn.Parameter(torch.FloatTensor([0.1]))
        self.__temp.requires_grad = False
    
    @abstractstaticmethod
    def name():
        return 'Module'
    
    @abstractmethod
    def loss(self, lhs_pos, rhs_pos, lhs_neg, rhs_neg):
        pass
    
    @abstractmethod
    def predict(self, triple):
        pass
    
    def dataloader(self, dataset, batch_size=200, generate_negative_rate=1):
        return DataLoader(dataset, batch_size, generate_negative_rate)
    
    def fit(self, dataset, num_epoch, batch_size, generate_negative_rate=1, opt='Adam',
            lr=1e-4, weight_decay=0, scheduler_step=None, scheduler_gamme=0.75,
            save_checkpoint_path='checkpoint', valid_step=None, valid_function=None):
        
        device = self.__temp.device
        
        dataloader = self.dataloader(dataset, batch_size, generate_negative_rate)
        process = Train(dataloader, self, opt, lr, weight_decay, scheduler_step,
                        scheduler_gamme, device)
        process.fit(num_epoch, None, valid_step, valid_function)
        
        dataset_name = dataset.name
        model_name = self.name()
        save_path = os.path.join(os.getcwd(), model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = f'{model_name} on {dataset_name}.tgz'
        self.save_checkpoint(os.path.join(save_path, file_name))
    
    def link_prediction(self, dataset, test_flag='test', batch_eval_size=2048,
                        flags='original', predtrain_model_file=None):
        dataset_name = dataset.name
        model_name = self.name()
        if predtrain_model_file is None:
            predtrain_model_file = os.path.join(os.getcwd(), model_name)
            fl = f'{model_name} on {dataset_name}.tgz'
            predtrain_model_file = os.path.join(predtrain_model_file, fl)
        
        self.load_checkpoint(predtrain_model_file)
        
        print()
        print('Link prediction')
        
        device = self.__temp.device
        link_prediction(self.predict, self.name(), dataset, dataset.entity_total,
                        dataset.relation_total, test_flag, None, batch_eval_size,
                        flags, device)
    
    def link_n2n_prediction(self, dataset, batch_eval_size=2048, flags='original',
                            predtrain_model_file=None):
        dataset_name = dataset.name
        model_name = self.name()
        if predtrain_model_file is None:
            predtrain_model_file = os.path.join(os.getcwd(), model_name)
            fl = f'{model_name} on {dataset_name}.tgz'
            predtrain_model_file = os.path.join(predtrain_model_file, fl)
        self.load_checkpoint(predtrain_model_file)
        
        print()
        
        device = self.__temp.device
        link_n2n_prediction(self.predict, self.name(), dataset, dataset.entity_total,
                            dataset.relation_total, batch_eval_size, flags, device)
    
    # def classification(self, dataset, batch_size=1000, test_flag='test',
    #                    predtrain_model_file=None):
    #     dataset_name = dataset.name
    #     model_name = self.name()
    #     if predtrain_model_file is None:
    #         predtrain_model_file = os.path.join(os.getcwd(), model_name)
    #         fl = f'{model_name} on {dataset_name}.tgz'
    #         predtrain_model_file = os.path.join(predtrain_model_file, fl)
    #     self.load_checkpoint(predtrain_model_file)
    #     device = self.__temp.device
    #     classification(self.predict, self.name(), dataset, batch_size, test_flag, device)
    
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
    
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()
    
    def pred_train_from(self, path):
        self.load_state_dict(path)
        self.train()
        