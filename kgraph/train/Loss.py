#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/26 17:59:05
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class KG():
    def __init__(self, model, loss_function, opt):
        self.model = model
        self.opt = opt
    
    def loss(self, lhs_pos, rhs_pos, lhs_neg, rhs_neg):
        pass



