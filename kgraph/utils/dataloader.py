#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2021/02/26 15:43:57
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import torch
from tqdm.std import tqdm

from .sampler import Sampler

def wrapper(temp_data, batch_size, function, i):
    j = i * batch_size
    batch_data = temp_data[j: j+batch_size, :]
    _batch_size = len(batch_data)
    return function(batch_data, _batch_size)

def _get_batch_data(temp_data, batch_size, function):
    
    return partial(wrapper, temp_data, batch_size, function)

def _get_data_iter(function, num_batchs, num_workers, use_tqdm=True):
    
    NUM_BATCH = num_batchs
    if num_workers < 2:
        for i in range(NUM_BATCH):
            yield function(i)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            _batchs = range(NUM_BATCH)
            chunksize, _ = divmod(NUM_BATCH, executor._max_workers * 4)
            chunksize = chunksize if chunksize > 0 else 1
            
            if use_tqdm:
                with tqdm(_batchs) as par:
                    for _, batch_data in zip(par, executor.map(function, _batchs, chunksize=chunksize)):
                        yield batch_data
            else:
                for batch_data in executor.map(function, _batchs, chunksize=chunksize):
                    yield batch_data

class DataIter(object):
    
    def __init__(self, dataset, batch_size, batch_sampler=None, shuffle=True, neg_ratio=None,
                 num_workers=0, use_selecting_src_rate=True, flags='train', device='cpu'):
        self.num_entities = dataset.entity_total
        self.dataset = dataset
        
        if flags == 'train':
            neg_ratio = 1 if neg_ratio is None else neg_ratio
            self.batch_sampler = Sampler(neg_ratio) if batch_sampler is None else batch_sampler
            self.batch_sampler.all_triples = dataset.get_all_triples()
            self.batch_sampler.num_entities = self.num_entities
            self.shuffle = shuffle
            
            if use_selecting_src_rate and 'select_src' in self.batch_sampler.__class__.__name__.lower():
                self.batch_sampler.select_src_rate = dataset.select_src_rate

            self.temp = dataset.train
        elif flags == 'valid':
            self.temp = dataset.valid
        else:
            self.temp = dataset.test
        self.flags = flags
        self._size = len(self.temp)
        self.batch_size = batch_size
        self.num_batchs = self._size // self.batch_size + 1
        
        self.num_workers = num_workers
        self.name = dataset.name
        self.device = device if flags == 'train' else 'cpu'
    
    def __iter__(self):
        
        if self.flags == 'train':
            temp = self.temp.copy()[np.random.permutation(self._size)]
            
            for batch_data in _get_data_iter(_get_batch_data(temp, self.batch_size, self.batch_sampler),
                                             self.num_batchs, self.num_workers, False):
                yield [torch.from_numpy(i).long().to(self.device) for i in batch_data]
        
        else:
            for i_data in self.temp:
                yield torch.from_numpy(i_data).long().to(self.device)
    
    def __call__(self):
        return self.__iter__()


