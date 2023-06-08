import numpy as np
from kgraph import FB15k237, FB13, WN11
from kgraph import FB15k, WN18, WN18RR

def runner(data_list):
    
    for data_class in data_list:
        data = data_class()
        print(data.__class__.__name__)
        data.num_neg = 10
        # data.smooth_lambda = 0.1
        for batch in data:
            print(batch)
            break


data_list = [FB15k237, FB13, FB15k, WN18, WN18RR, WN11]

runner(data_list)