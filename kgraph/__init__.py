from .utils import load_data
from ._utils.read import DataSet
from ._utils.sample import Sample

from ._utils.test import calculate_ranks_on_valid_via_triple
from ._utils.test import calculate_ranks_on_valid_via_pair
from ._utils.test import calculate_ranks_on_test_via_triple
from ._utils.test import calculate_ranks_on_test_via_pair


__all__ = ['Data', 'FB15k', 'FB15k237', 'WN18', 'Wn18RR', 'DataIter', 'Predict']

class Data(DataSet):
    
    def __init__(self, path: str) -> None:
        path += '/'
        
        if isinstance(path, str):
            path = path.encode('utf-8')
        if isinstance(path, bytes):
            super(Data, self).__init__(path)
        else:
            print('Can not find the dataset.')
        
        
class FB15k(Data):
    def __init__(self, path: str=None, no_sort: bool=True) -> None:
        path = 'data/' if path is None else path
        url='https://raw.githubusercontent.com/pp413/Knowledge_embedding_benchmark_datasets/main/FB15k.zip'
        super(FB15k, self).__init__(load_data(url, path, no_sort=no_sort))   

class FB15k237(Data):
    def __init__(self, path: str=None, no_sort: bool=True) -> None:
        path = 'data/' if path is None else path
        url='https://raw.githubusercontent.com/pp413/Knowledge_embedding_benchmark_datasets/main/FB15k-237.zip'
        super(FB15k237, self).__init__(load_data(url, path, no_sort=no_sort))


class WN18(Data):
    def __init__(self, path: str=None, no_sort: bool=True) -> None:
        path = 'data/' if path is None else path
        url='https://raw.githubusercontent.com/pp413/Knowledge_embedding_benchmark_datasets/main/WN18.zip'
        super(WN18, self).__init__(load_data(url, path, no_sort=no_sort))

class WN18RR(Data):
    def __init__(self, path: str=None, no_sort: bool=True) -> None:
        path = 'data/' if path is None else path
        url='https://raw.githubusercontent.com/pp413/Knowledge_embedding_benchmark_datasets/main/WN18RR.zip'
        super(WN18RR, self).__init__(load_data(url, path, no_sort=no_sort))

class DataIter(Sample):
    
    def __init__(self, num_ent: int, num_rel: int, batch_size: int=128, num_threads: int=2,
                 smooth_lambda: float=0.1, num_neg: int=1, mode: str='all', bern_flag: bool=False,
                 seed: int=1) -> None:
        '''
        Args:
            num_ent: number of entities
            num_rel: number of relations
            batch_size: batch size
            num_threads: number of threads
            smooth_lambda: lambda for smooth
            num_neg: number of negative samples
            mode: mode must be one of "all", "head", "tail", "head_tail", default: "all"
            ber_flag: whether to use bernoulli sampling
            seed: random seed
        '''
        
        assert mode in ['all', 'head', 'tail', 'head_tail'], 'mode must be one of "all", "head", "tail", "head_tail"'
        modes = {'all': 0, 'head': -1, 'tail': 1, 'head_tail': 0}
        
        super(DataIter, self).__init__(num_ent, num_rel, batch_size, num_threads, smooth_lambda,
                                         num_neg, modes[mode], int(bern_flag), seed)

    def generate_triple_with_negative(self):
        return super(DataIter, self).generate_triple_with_negative()
    
    def generate_triple_with_negative_on_random(self):
        return super(DataIter, self).generate_triple_with_negative_on_random()
    
    def generate_pair(self):
        return super(DataIter, self).generate_pair()
    
    def generate_pair_on_random(self):
        return super(DataIter, self).generate_pair_on_random()

    
class Predict:
    
    def __init__(self, element_type: str='triple') -> None:
        assert element_type in ['triple', 'pair'], 'element_type must be one of "triple", "pair"'
        
        if element_type == 'triple':
            self.__predict_test = calculate_ranks_on_test_via_triple
            self.__predict_valid = calculate_ranks_on_valid_via_triple
        else:
            self.__predict_test = calculate_ranks_on_test_via_pair
            self.__predict_valid = calculate_ranks_on_valid_via_pair
    
    def predict_test(self, function, num_ent: int, num_rel: int, batch_size: int=256):
        return self.__predict_test(function, num_ent, num_rel, batch_size)
    
    def predict_valid(self, function, num_ent: int, num_rel: int, batch_size: int=256):
        return self.__predict_valid(function, num_ent, num_rel, batch_size)

