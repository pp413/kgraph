#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/27 10:54:42
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
from torch.optim import SGD, Adagrad, Adam


class Opt():
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
            self.opt = opt(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def __call__(self):
        return self.opt