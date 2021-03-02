#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/26 17:52:08
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#

import os
import pickle as pkl
import sys

import numpy as np

from ._utils import *


KG_BENCHMARK_DATASETS = ['WN18', 'WN18RR', 'FB15k', 'FB15k-237']


def _load_data(data_name, data_sha1, path=None, original=False):
    url = get_from_aigraph_url(data_name)
    path = set_download_dir(path)
    
    taz_path = download(url, path)
    data_dir = os.path.join(os.environ['KG_DIR'], data_name)
    
    extract_archive(taz_path, data_dir)
    
    data = load_and_check_original_data(data_dir, data_name, data_sha1)
    
    data = clean_data(data)
    
    data, original_data, total_entities, total_relations = str_to_idx(data)
    if original:
        return pprint(original_data, total_entities, total_relations, data_name)

    return pprint(data, total_entities, total_relations, data_name)


def load_fb15k(path=None, original=False):
    """Load the FB15k dataset

    .. warning::
        The dataset includes a large number of inverse relations that spilled to the test set, and its use in
        experiments has been deprecated. Use FB15k-237 instead.

    FB15k is a split of Freebase, first proposed by :cite:`bordes2013translating`.

    The FB15k dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    The dataset is divided in three splits:
    
    - ``train``
    - ``valid``
    - ``test``

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    FB15K     483,142   50,000  59,071  14,951        1,345
    ========= ========= ======= ======= ============ ===========
    Cleaned   483,142   50,000  59,071  14,951        1,345
    ========= ========= ======= ======= ============ ===========

    """
    data_name = 'FB15k'
    
    data_sha1 = {
        'train': '578bdb6b4311d22d4baf7da30aaadf03d687c84d',
        'test': '00d340728878df4f0b318fd1f488855e9b770425',
        'valid': '2694fe891109dea3470bd975dd55eeb12ef30cbd'
    }
    
    return _load_data(data_name, data_sha1, path, original)

def load_fb15k237(path=None, original=False):
    """Load the FB15k-237 dataset

    FB15k-237 is a reduced version of FB15K. It was first proposed by :cite:`toutanova2015representing`.

    The FB15k-237 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    The dataset is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    FB15K-237 272,115   17,535  20,466  14,541        237
    ========= ========= ======= ======= ============ ===========
    Cleaned   272,115   17,516  20,438  14,505        237
    ========= ========= ======= ======= ============ ===========

    return data, total_entities, total_relations
    """

    data_name = 'FB15k-237'
    
    data_sha1 = {
        'train': '1448a31a528da315217960edeca97f68209f8254',
        'test': '263a7bd582cf1d27961fc4143ca2bee474bf03fc',
        'valid': '732eb032787161e61b7bcc7ab3965b7569b405ce'
    }
    
    return _load_data(data_name, data_sha1, path, original)


def load_wn18(path=None, original=False):
    """Load the WN18 dataset

    .. warning::
        The dataset includes a large number of inverse relations that spilled to the test set, and its use in
        experiments has been deprecated. Use WN18RR instead.

    WN18 is a subset of Wordnet. It was first presented by :cite:`bordes2013translating`.

    The WN18 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    The dataset is divided in three splits:

    - ``train``: 141,442 triples
    - ``valid`` 5,000 triples
    - ``test`` 5,000 triples

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    WN18      141,442   5,000   5,000   40,943        18
    ========= ========= ======= ======= ============ ===========
    Cleaned   141,442   5,000   5,000   40,943        18 
    ========= ========= ======= ======= ============ ===========

    return data, total_entities, total_relations
    """

    data_name = 'WN18'
    
    data_sha1 = {
        'train': 'b78e956440e0b7631517aa1b230818f581281e6d',
        'test': 'e5308598809646ad01da6b4d9cede189f918aa31',
        'valid': '2433f787c8dcf3ac376d70ebd358ef3998313ea3'
    }
    
    return _load_data(data_name, data_sha1, path, original)

def load_wn18rr(path=None, original=False):
    """Load the WN18RR dataset

    The dataset is described in :cite:`DettmersMS018`.

    The WN18RR dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.


    It is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    WN18RR    86,835    3,034   3,134   40,943        11
    ========= ========= ======= ======= ============ ===========
    Cleaned   86,835    3,034   3,134   40,943        11
    ========= ========= ======= ======= ============ ===========

    .. warning:: WN18RR's validation set contains 198 unseen entities over 210 triples.
        The test set has 209 unseen entities, distributed over 210 triples.

    return data, total_entities, total_relations
    """

    data_name = 'WN18RR'
    
    data_sha1 = {
        'train': '078fd2890583f99d75342d0eeea6d0c4e6167c76',
        'test': 'da28a9c1759d66f87873d8e1ecc4884501211f83',
        'valid': '38bbe458b5f8e36310456cac5e0119f05d39cef5'
    }
    
    return _load_data(data_name, data_sha1, path, original)

def load_all_datasets(path=None):
    load_fb15k(path=path)
    load_fb15k237(path=path)
    load_wn18(path=path)
    load_wn18rr(path=path)


class DataBase():
    r"""
    The Base Data Class.
    
    Arguments:
        data_name: Name of Dataset.
        data: a dict type of Dataset, {'train': train_id_array, 'test': test_id_array, 'valid': valid_id_array,
              'entity_total': the number of entities, 'relation_total': the number of relations}
        path: the path of these benchmark datasets in Kgraph, default, '~/.KGDataSets'.

    """
    
    def __init__(self, data_name='DataName', data=None, path=None) -> None:
        if data is None and data_name in KG_BENCHMARK_DATASETS:
            self.__initialized(data_name, path=path)
        else:
            self.data = data
            self._entity_total = data['entity_total']
            self._relation_total = data['relation_total']
            
            self.dir_name = None
            self.fdir = '_None_'

            
    def __initialized(self, data_name, path=None):
        
        kg_benchmark_dataset = set([x.lower() for x in KG_BENCHMARK_DATASETS])
        print(data_name)
        # dir_name = data_name.lower()
        dir_name = data_name
        assert dir_name not in kg_benchmark_dataset, f'{data_name} is not in benchmark datasets.'
        
        self.path = path
        self.dir_name = dir_name
        
        self.filenames = [
            '1-1.txt',
            '1-n.txt',
            'n-1.txt',
            'n-n.txt',
            'constrain.txt',
            'entity2id.txt',
            'relation2id.txt',
            'train.csv',
            'valid.csv',
            'test.csv',
            'test2id_all.txt'
        ]
        
        self.fdir = os.path.join(os.environ['KG_DIR'], data_name)
        if not os.path.exists(os.path.join(os.environ['KG_DIR'], 'statistics.txt')):
            self.data, self._entity_total, self._relation_total = self.load_dataset()
        
        row_data, data_names = load_table(os.path.join(os.environ['KG_DIR'], 'statistics.txt'))
        # data_names = list(data_names)
        # data_names = set([data_names[i].lower() for i in range(len(data_names))])
        
        # if not os.path.exists(self.fdir):
        if dir_name not in data_names:
            self.data, self._entity_total, self._relation_total = self.load_dataset()
        
        else:
            tb = pt.PrettyTable(header=True)
            tb.title = f'The statistics of benchmark datasets.'
            tb.field_names = ['DataName', 'TrainSet', 'ValidSet', 'TestSet', 'Entities', 'Relations']
            if row_data is not None:
                for line in row_data:
                    tb.add_row(line)
            print(tb)

            _row_data = {row_data[i][0]: row_data[i][1:] for i in range(len(row_data))}
            
            self._entity_total = _row_data[dir_name][3]
            self._relation_total = _row_data[dir_name][4]
    
    @property
    def name(self):
        return self.dir_name
    
    @property
    def entity_total(self):
        return int(self._entity_total)
    
    @property
    def relation_total(self):
        return int(self._relation_total)
    
    @property
    def size(self):
        return len(self.data['train'])
    
    def load_dataset(self):
        if self.dir_name == 'fb15k':
            return load_fb15k(path=self.path)
        elif self.dir_name == 'fb15k-237':
            return load_fb15k237(path=self.path)
        elif self.dir_name == 'wn18':
            return load_wn18(path=self.path)
        else:
            return load_wn18rr(path=self.path)
    
    def load_1_1(self):
        if self.dir_name is None:
            return None
        file_path = os.path.join(os.environ['KG_DIR'], self.dir_name, '1-1.txt')
        return load_from_text(file_path=file_path, dtype=np.int64)
    
    def load_1_n(self):
        if self.dir_name is None:
            return None
        file_path = os.path.join(os.environ['KG_DIR'], self.dir_name, '1-n.txt')
        return load_from_text(file_path=file_path, dtype=np.int64)
    
    def load_n_1(self):
        if self.dir_name is None:
            return None
        file_path = os.path.join(os.environ['KG_DIR'], self.dir_name, 'n-1.txt')
        return load_from_text(file_path=file_path, dtype=np.int64)
    
    def load_n_n(self):
        if self.dir_name is None:
            return None
        file_path = os.path.join(os.environ['KG_DIR'], self.dir_name, 'n-n.txt')
        return load_from_text(file_path=file_path, dtype=np.int64)
    
    def load_train(self):
        if not os.path.exists(self.fdir):
            return self.data['train']
        file_path = os.path.join(os.environ['KG_DIR'], self.dir_name, 'train.csv')
        return load_from_csv(file_path=file_path, dtype=np.int64)
    
    def load_test(self):
        if not os.path.exists(self.fdir):
            return self.data['test']
        file_path = os.path.join(os.environ['KG_DIR'], self.dir_name, 'test.csv')
        return load_from_csv(file_path=file_path, dtype=np.int64)
    
    def load_valid(self):
        if not os.path.exists(self.fdir):
            return self.data['valid']
        file_path = os.path.join(os.environ['KG_DIR'], self.dir_name, 'valid.csv')
        return load_from_csv(file_path=file_path, dtype=np.int64)
    
    def load_constraint(self):
        if self.dir_name is None:
            return None
        file_path = os.path.join(os.environ['KG_DIR'], self.dir_name, 'constraint.txt')
        constraint = {'rel_src': {}, 'rel_dst': {}}
        with open(file_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = [int(x) for x in line.strip().split('\t')]
                if i % 2 == 0:
                    constraint['rel_src'][line[0]] = set(line[2:])
                else:
                    constraint['rel_dst'][line[0]] = set(line[2:])
        return constraint
        
    @property
    def train(self):
        return self.load_train()
    
    @property
    def test(self):
        return self.load_test()
    
    @property
    def valid(self):
        return self.load_valid()
    
    @property
    def unique_entities(self):
        return np.unique(np.concatenate((self.train[:, 0], self.train[:, 2])))
    
    @property
    def all_triples(self):
        return np.concatenate((self.train, self.valid, self.test), 0)
    
    # @staticmethod
    # def build_graph(data_array):
    #     return build_graph(data_array)
    
    def get_all_triples(self):
        data = {'train': self.train, 'test': self.test, 'valid': self.valid}
        return get_all_triples(data)
    
    def get_select_src_rate(self):
        data = {'train': self.train, 'test': self.test, 'valid': self.valid}
        return get_select_src_rate(data)

    @property
    def select_src_rate(self):
        return self.get_select_src_rate()
    
    

class FB15k(DataBase):
    def __init__(self, path=None):
        super(FB15k, self).__init__(data_name='FB15k', path=path)

class FB15k237(DataBase):
    def __init__(self, path=None):
        super(FB15k237, self).__init__(data_name='FB15k-237', path=path)

class WN18(DataBase):
    def __init__(self, path=None):
        super(WN18, self).__init__(data_name='WN18', path=path)

class WN18RR(DataBase):
    def __init__(self, path=None):
        super(WN18RR, self).__init__(data_name='WN18RR', path=path)           






