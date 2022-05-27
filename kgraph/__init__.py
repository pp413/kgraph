import os
import numpy as np
from .utils import load_data
from .utils.read import DataSet
from .utils.sample import Sample
from .utils.tools import generateN2N, load_triple_original_file

from .utils.evaluation import calculate_ranks_on_valid_via_triple
from .utils.evaluation import calculate_ranks_on_valid_via_pair
from .utils.evaluation import calculate_ranks_on_test_via_triple
from .utils.evaluation import calculate_ranks_on_test_via_pair
from .utils.classification import run_triple_classification
from .log import get_result_table, log_N2N

np.set_printoptions(precision=4)

__all__ = ['Data', 'FB15k', 'FB15k237', 'WN18', 'Wn18RR', 'DataIter', 'Predict']

class Data(DataSet):
    
    def __init__(self, num_ent: int=0, num_rel: int=0):
        super(Data, self).__init__(num_ent, num_rel)
    
    def load(self, path: str, no_sort: bool=True) -> None:
        path += '/'
        
        if isinstance(path, str):
            path = path.encode('utf-8')
        if isinstance(path, bytes):
            super(Data, self).load(path, int(no_sort))
        else:
            print('Can not find the dataset.')
        self.path = path.decode('utf-8')
        # print('ok')
        if not os.path.exists(os.path.join(self.path, 'constraint.txt')):
            generateN2N(self.train, self.valid, self.test, self.path)
    
    def clean(self):
        clean_files = ['1-1.txt',
                       '1-n.txt',
                       'n-1.txt',
                       'n-n.txt',
                       'constraint.txt',
                       'train2id.txt',
                       'valid2id.txt',
                       'test2id.txt',
                       'statistics.txt']
        
        for f in clean_files:
            if os.path.exists(os.path.join(self.path, f)):
                os.remove(os.path.join(self.path, f))

    def __load_triple_from_file__(self, file: str):
        return np.array(load_triple_original_file(os.path.join(self.path, file)), dtype=np.int32)
    
    @property
    def one2one(self):
        return self.__load_triple_from_file__('1-1.txt')
    
    @property
    def one2multi(self):
        return self.__load_triple_from_file__('1-n.txt')
    
    @property
    def multi2one(self):
        return self.__load_triple_from_file__('n-1.txt')
    
    @property
    def multi2multi(self):
        return self.__load_triple_from_file__('n-n.txt')
        
        
class FB15k(Data):
    def __init__(self, path: str=None, no_sort: bool=True) -> None:
        path = 'data/' if path is None else path
        url='https://raw.githubusercontent.com/pp413/Knowledge_embedding_benchmark_datasets/main/FB15k.zip'
        super(FB15k, self).__init__()
        super(FB15k, self).load(*load_data(url, path, no_sort=no_sort))

class FB15k237(Data):
    def __init__(self, path: str=None, no_sort: bool=True) -> None:
        path = 'data/' if path is None else path
        url='https://raw.githubusercontent.com/pp413/Knowledge_embedding_benchmark_datasets/main/FB15k-237.zip'
        super(FB15k237, self).__init__()
        super(FB15k237, self).load(*load_data(url, path, no_sort=no_sort))

class WN18(Data):
    def __init__(self, path: str=None, no_sort: bool=True) -> None:
        path = 'data/' if path is None else path
        url='https://raw.githubusercontent.com/pp413/Knowledge_embedding_benchmark_datasets/main/WN18.zip'
        super(WN18, self).__init__()
        super(WN18, self).load(*load_data(url, path, no_sort=no_sort))

class WN18RR(Data):
    def __init__(self, path: str=None, no_sort: bool=True) -> None:
        path = 'data/' if path is None else path
        url='https://raw.githubusercontent.com/pp413/Knowledge_embedding_benchmark_datasets/main/WN18RR.zip'
        super(WN18RR, self).__init__()
        super(WN18RR, self).load(*load_data(url, path, no_sort=no_sort))

class DataIter(Sample):
    
    def __init__(self, data: Data, batch_size: int=128, shuffle: bool=False, num_threads: int=2,
                 smooth_lambda: float=0.1, num_neg: int=1, mode: str='all', bern_flag: bool=False,
                 seed: int=41504, element_type: str='triple') -> None:
        '''
        Args:
            num_ent: number of entities
            num_rel: number of relations
            batch_size: batch size
            shuffle: whether to shuffle the data, default: False.
            num_threads: number of threads
            smooth_lambda: float, the smooth lambda for the labels in sampling process, default 0.1.
            num_neg: int, the number of negative samples for each positive sample, default 1.
            mode: str, the corrupting on xxx, choice from ['all', 'head', 'tail', 'head_tail', 'normal', 'cross'], default 'all'.
            bern_flag: bool, whether to use bernoulli sampling, default False.
            seed: random seed, default 41504.
        '''
        
        assert mode.lower() in ['all', 'head', 'tail', 'head_tail', 'normal', 'cross'], 'mode must be one of "all", "head", "tail", "head_tail"'
        modes = {'all': 0, 'tail': -1, 'head': 1, 'head_tail': 0, 'normal': 0, 'cross': 2}
        assert element_type.lower() in ['triple', 'pair'], 'element_type must be triple or pair'
        element_types = {'triple': 0, 'pair': 1}
        
        num_ent = data.num_ent
        num_rel = data.num_rel
        
        super(DataIter, self).__init__(data, batch_size, num_threads, smooth_lambda, int(shuffle),
                                       num_neg, modes[mode.lower()], int(bern_flag), seed, element_types[element_type.lower()])

    def generate_triple_with_negative(self):
        return super(DataIter, self).generate_triple_with_negative()
    
    def generate_triple_with_negative_on_random(self):
        return super(DataIter, self).generate_triple_with_negative_on_random()
    
    def generate_pair(self):
        return super(DataIter, self).generate_pair()
    
    def generate_pair_on_random(self):
        return super(DataIter, self).generate_pair_on_random()

    
class Predict:
    
    def __init__(self, data: Data, element_type: str='triple') -> None:
        assert element_type.lower() in ['triple', 'pair'], 'element_type must be one of "triple", "pair"'
        
        self.data = data
        self.num_ent = data.num_ent
        self.num_rel = data.num_rel
        self.data_name = data.__class__.__name__
        self.element_type = element_type.lower()
        
        if self.element_type == 'triple':
            self.__predict_test = calculate_ranks_on_test_via_triple
            self.__predict_valid = calculate_ranks_on_valid_via_triple
        else:
            self.__predict_test = calculate_ranks_on_test_via_pair
            self.__predict_valid = calculate_ranks_on_valid_via_pair
    
    def predict_test(self, function, batch_size: int=256):
        results = self.__predict_test(function, self.data, batch_size)
        return get_result_table(*results, data_name=self.data_name, flags='Test')
    
    def predict_valid(self, function, batch_size: int=256):
        results = self.__predict_valid(function, self.data, batch_size)
        return get_result_table(*results, data_name=self.data_name, flags='Valid')
    
    def predict_N2N(self, function, batch_size: int=256, show: bool=False):
        
        results = {'1to1': [], '1toN': [], 'Nto1': [], 'NtoN': []}
        data = self.data
        
        # 1 to 1
        self.data.test = data.one2one
        print('1 to 1')
        results['1to1'] += [i for i in self.__predict_test(function, self.data, batch_size)]
        
        # 1 to n
        self.data.test = data.one2multi
        print('1 to n')
        results['1toN'] += [i for i in self.__predict_test(function, self.data, batch_size)]
        
        # n to 1
        self.data.test = data.multi2one
        print('n to 1')
        results['Nto1'] += [i for i in self.__predict_test(function, self.data, batch_size)]
        
        # n to n
        self.data.test = data.multi2multi
        print('n to n')
        results['NtoN'] += [i for i in self.__predict_test(function, self.data, batch_size)]
        
        return log_N2N(results, data_name=self.data_name)

    def calculate_classification_accuracy(self, function, batch_size=1000, threshold=None):

        def _tmp_element_triple_function(x, y):
            return function(x)
        
        def _tmp_element_pair_function(x, y):
            
            corrupt_tail = x[y==1, :]
            corrupt_head = x[y==0, :]
            corrupt_head = corrupt_head[:, [2, 1, 0]]
            corrupt_head[:, 1] = corrupt_head[:, 1] + self.num_rel
            
            data = np.concatenate((corrupt_tail, corrupt_head), axis=0)
            
            col = data[:, 2]
            row = np.arange(data.shape[0])
            
            return function(data[:, [0, 1]])[row, col]
        
        threshold = -1.0 if threshold is None else threshold
        
        tmp_function = _tmp_element_triple_function if self.element_type == 'triple' else _tmp_element_pair_function
        
        accuracy, threshold = run_triple_classification(tmp_function, self.data, batch_size, threshold)
        return accuracy, threshold
        


