# coding:utf-8
import os
import ctypes
import numpy as np

from .read import load_wn18, load_wn18rr, load_wn11
from .read import load_fb15k, load_fb15k237, load_fb13
from .read import load_yago3_10

datasets_name = {'WN18': 0, 'WN18RR': 1, 'FB15k': 2,
                 'FB15k-237': 3, 'YAGO3-10': 4, 'wordnet11': 5, 'freebase13': 6}
all_data_file_name = ['wn18', 'wn18RR', 'fb15k',
                      'fb15k-237', 'YAGO3-10', 'wordnet11', 'freebase13']
datasets = [load_wn18, load_wn18rr, load_fb15k, load_fb15k237, load_yago3_10]


class TrainDataSampler(object):

    def __init__(self, nbatches, datasampler):
        self.nbatches = nbatches
        self.datasampler = datasampler
        self.batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.batch += 1
        if self.batch > self.nbatches:
            raise StopIteration()
        return self.datasampler()

    def __len__(self):
        return self.nbatches


class TrainDataLoader(object):

    def __init__(self,
                 dataset_name="FB15k-237",
                 tri_file=None,
                 ent_file=None,
                 rel_file=None,
                 batch_size=None,
                 nbatches=None,
                 threads=8,
                 sampling_mode="normal",
                 bern_flag=False,
                 filter_flag=True,
                 neg_ent=1,
                 neg_rel=0):
        '''the data loader.
        args:
        dataset_name: the dataset name(string) for training, (default: "FB15k-237").
        tri_file: the train id set file name, (default: "train2id.txt")
        ent_file: the entities id dict file name, (default: "entity2id.txt")
        rel_file: the relations id dict file name, (default: "relation2id.txt")
        batch_size: Batch Size of per batch.
        nbatches: number of batches
        threads: the threads for sampling process.
        sampling_mode: the mode for sampling, (default: "normal", can choice the mode "cross")
        bern_flag:
        filter_flag: bool, filter in sampling, (default: True)
        neg_ent: generate the neg samples rate by corrupt entity (int, default: 1)
        neg_rel: generate the neg samples rate by corrupt relation (int, default: 0)
        '''

        base_file = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../release/Base.so"))
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """argtypes"""
        self.lib.sampling.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64
        ]
        # self.dataset_name = dataset_name
        # print(datasets_name[dataset_name])
        self.dataset_name = os.path.join(os.environ['KG_DIR'], all_data_file_name[datasets_name[dataset_name]]) + '/'
        try:
            load_data = datasets[datasets_name[dataset_name]]
            if not os.path.exists(os.path.join(os.environ['KG_DIR'], all_data_file_name[datasets_name[dataset_name]])):
                load_data()
        except:
            return
        self.tri_file = tri_file
        self.ent_file = ent_file
        self.rel_file = rel_file
        if self.dataset_name != None:
            self.tri_file = self.dataset_name + "train2id.txt"
            self.ent_file = self.dataset_name + "entity2id.txt"
            self.rel_file = self.dataset_name + "relation2id.txt"
        """set essential parameters"""
        self.work_threads = threads
        self.nbatches = nbatches
        self.batch_size = batch_size
        self.bern = bern_flag
        self.filter = filter_flag
        self.negative_ent = neg_ent
        self.negative_rel = neg_rel
        self.sampling_mode = sampling_mode
        self.cross_sampling_flag = 0
        self.read()

    def read(self):
        # set the dataset dir
        if self.dataset_name != None:
            self.lib.setInPath(ctypes.create_string_buffer(
                self.dataset_name.encode(), len(self.dataset_name) * 2))
        else:
            self.lib.setTrainPath(ctypes.create_string_buffer(
                self.tri_file.encode(), len(self.tri_file) * 2))
            self.lib.setEntPath(ctypes.create_string_buffer(
                self.ent_file.encode(), len(self.ent_file) * 2))
            self.lib.setRelPath(ctypes.create_string_buffer(
                self.rel_file.encode(), len(self.rel_file) * 2))

        self.lib.setBern(self.bern)
        self.lib.setWorkThreads(self.work_threads)
        self.lib.randReset()
        if os.path.exists(self.tri_file):
            self.lib.importTrainFiles()
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.tripleTotal = self.lib.getTrainTotal()

        if self.batch_size == None:
            self.batch_size = self.tripleTotal // self.nbatches
        if self.nbatches == None:
            self.nbatches = self.tripleTotal // self.batch_size
        self.batch_seq_size = self.batch_size * \
            (1 + self.negative_ent + self.negative_rel)

        self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
        self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
        self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
        self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
        self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

    def sampling(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            0,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h,
            "batch_t": self.batch_t,
            "batch_r": self.batch_r,
            "batch_y": self.batch_y,
            "mode": "normal"
        }

    def sampling_head(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            -1,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h,
            "batch_t": self.batch_t[:self.batch_size],
            "batch_r": self.batch_r[:self.batch_size],
            "batch_y": self.batch_y,
            "mode": "head_batch"
        }

    def sampling_tail(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
            1,
            self.filter,
            0,
            0
        )
        return {
            "batch_h": self.batch_h[:self.batch_size],
            "batch_t": self.batch_t,
            "batch_r": self.batch_r[:self.batch_size],
            "batch_y": self.batch_y,
            "mode": "tail_batch"
        }

    def cross_sampling(self):
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            return self.sampling_head()
        else:
            return self.sampling_tail()

    """interfaces to set essential parameters"""

    def set_work_threads(self, work_threads):
        self.work_threads = work_threads

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.nbatches = self.tripleTotal // self.batch_size

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_bern_flag(self, bern):
        self.bern = bern

    def set_filter_flag(self, filter):
        self.filter = filter

    """interfaces to get essential parameters"""

    def get_batch_size(self):
        return self.batch_size

    def get_ent_tot(self):
        return self.entTotal

    def get_rel_tot(self):
        return self.relTotal

    def get_triple_tot(self):
        return self.tripleTotal

    def __iter__(self):
        if self.sampling_mode == "normal":
            return TrainDataSampler(self.nbatches, self.sampling)
        else:
            return TrainDataSampler(self.nbatches, self.cross_sampling)

    def __len__(self):
        return self.nbatches
