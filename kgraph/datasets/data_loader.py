#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/26 17:52:00
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
import numpy as np
from ..function import generate_negative_sample


class SetSample():
    def __init__(self, function, **kwargs):
        def f(*data):
            return function(*data, **kwargs)
        
        self.function = f    
    def __call__(self, *data):
        return self.function(*data)


class DataLoader():
    def __init__(self, DataSets, batch_size=1000, generate_negative_rate=1):
        self.name = DataSets.__class__.__name__
        self.batch_size = batch_size
        self.len = DataSets.len()
        self.num_batch = self.len // batch_size + 1
        self.train_data = DataSets.data['train']
        self.valid_data = DataSets.data['valid']
        self.test_data = DataSets.data['test']
        self.__data = DataSets
        self.generate_negative_rate = generate_negative_rate
        self.num_ent = DataSets.entity_total
        self.select_src_rate = DataSets.get_select_src_rate()
        self.all_triples = DataSets.get_all_triples()
    
    @property
    def data_1_1(self):
        return self.__data.load_1_1()
    
    @property
    def data_1_n(self):
        return self.__data.load_1_n()
    
    @property
    def data_n_1(self):
        return self.__data.load_n_1()
    
    @property
    def data_n_n(self):
        return self.__data.load_n_n()
    
    def __len__(self):
        return self.len

    def with_negative(self):
        train_data = np.random.permutation(self.train_data)
        for i in range(self.num_batch):
            j = i * self.batch_size
            data = train_data[j: j + self.batch_size]
            yield generate_negative_sample(pos_sample=data, 
                    generate_negative_rate=self.generate_negative_rate,
                    num_ent=self.num_ent, select_src_rate=self.select_src_rate,
                    all_triples=self.all_triples)
    
    def without_negative(self):
        train_data = np.random.permutation(self.train_data)
        lhs_batch_size = self.batch_size // 2
        for i in range(self.num_batch):
            j = i * self.batch_size
            data = train_data[j: j + self.batch_size]
            yield [data[:lhs_batch_size, :], data[lhs_batch_size:, :], [], []]
    
    def __call__(self, batch_size=None, negative_rate=None):
        if batch_size is not None:
            self.batch_size = batch_size
        if negative_rate is not None:
            self.generate_negative_rate = negative_rate
        if self.generate_negative_rate > 0:
            return self.with_negative()
        else:
            return self.without_negative()






