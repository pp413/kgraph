#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2021/02/27 21:10:21
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
from .functions import calculate_ranks, calculate_n2n_ranks

            
def add_link_prediction(dataset):
    def _add_link_prediction(cls):
        def wrapper(self, batch_size=None, for_test=True, constraint=None):
            table = calculate_ranks(self.predict, dataset, for_test=for_test,
                                batch_size=batch_size, device=self.device, constraint=constraint)
            return table
        
        @property
        def name(self):
            return self.__class__.__name__
        
        cls.name = name
        cls.link_prediction = wrapper
        return cls
    return _add_link_prediction

            
def add_link_n2n_prediction(dataset):
    def _add_link_prediction(cls):
        def wrapper(self, batch_size=None, constraint=None):
            table = calculate_n2n_ranks(self.predict, dataset, batch_size, self.device, constraint)
            return table
        
        cls.link_n2n_prediction = wrapper
        return cls
    return _add_link_prediction

