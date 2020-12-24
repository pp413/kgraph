#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/26 19:14:46
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
import os
import torch as th

def get_parameters(model, mode='numpy', param_dict=None):
    all_param_dict = model.state_dict()
    if param_dict == None:
        param_dict = all_param_dict.keys()
    res = {}
    for param in param_dict:
        if mode == "numpy":
            res[param] = all_param_dict[param].cpu().numpy()
        elif mode == "list":
            res[param] = all_param_dict[param].cpu().numpy().tolist()
        else:
            res[param] = all_param_dict[param]
    return res

def save_checkpoint(model, epoch=None, dataname=None):
    root = os.getcwd()
    
    filename = f'{model.__class__.__name__}'
    filename += f'_{epoch}' if epoch is not None else ''
    filename += f'_{dataname}' if dataname is not None else ''
    filename += '.tgz'
    
    path = os.path.join(root, f'{model.__class__.__name__}', 'checkpoint')
    if not os.path.exists(path):
        os.makedirs(path)
    th.save(model.state_dict(), os.path.join(path, filename))
