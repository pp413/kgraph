# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as np
from cython cimport long, int, float, sizeof
from cython cimport boundscheck, wraparound
from cython.parallel cimport prange, parallel
from libc.stdio cimport printf
from .mem cimport Pool

from .cache_data cimport global_mem
from .random_int64 cimport setRandSeed
from .random_int64 cimport randReset
from .random_int64 cimport rand_max
from .random_int64 cimport rand64
from .random_int64 cimport _rand64
from .read cimport train_data
from .corrupt cimport find_target_id
from .corrupt cimport corrupt_tail_c
from .corrupt cimport corrupt_head_c

ctypedef struct Kwargs:
    int batch_size, negative_sample_size, num_threads
    int ent_num, rel_num, mode, start
    bint bern_flag
    float smooth_lambda
    int *TID
    int num_per_thread
    int normal_or_cross

cdef void shuffle(int *ptr, int length):
    cdef int tmp
    cdef int i, j
    for i in range(length):
        j = rand64(i, length)
        tmp = ptr[i]
        ptr[i] = ptr[j]
        ptr[j] = tmp

cdef void initializeKwargs(Kwargs *kwargs):
    kwargs.batch_size = 10
    kwargs.negative_sample_size = 1
    kwargs.num_threads = 2
    kwargs.ent_num = 0
    kwargs.rel_num = 0
    kwargs.mode = 0
    kwargs.start = 0
    kwargs.bern_flag = 0
    kwargs.smooth_lambda = 0.0
    kwargs.TID = NULL
    kwargs.num_per_thread = 0
    kwargs.normal_or_cross = 0

@boundscheck(False)
@wraparound(False)
cdef void generate_triple_with_negative_on_random(long[:, ::1] batch_data, float[:, ::1] batch_labels, Kwargs *kwargs) nogil:
    cdef:
        int i, j, tmp, tId
        int lef, rig
        float prob, p
    
    for tId in prange(kwargs.num_threads, nogil=True, schedule='static', num_threads=kwargs.num_threads):
        lef = tId * kwargs.num_per_thread
        rig = min((tId + 1) * kwargs.num_per_thread, kwargs.batch_size)
        for i in range(lef, rig):
            tmp = rand_max(tId, train_data.data_size)
            batch_data[i, 0] = train_data.data[tmp].head
            batch_data[i, 1] = train_data.data[tmp].rel
            batch_data[i, 2] = train_data.data[tmp].tail
            batch_labels[i, 0] = 0.
            for j in range(1, kwargs.negative_sample_size + 1):
                tmp = i + j * kwargs.batch_size
                if kwargs.mode == 0 and kwargs.normal_or_cross == 0:
                    if kwargs.bern_flag:
                        p = (train_data.rig_mean[batch_data[i, 1]] + train_data.lef_mean[batch_data[i, 1]])
                        prob = 1000. * train_data.rig_mean[batch_data[i, 1]]
                    else:
                        p = 1.
                        prob = 500.
                    
                    if (_rand64(tId) % 1000) * p < prob:
                        batch_data[tmp, 0] = batch_data[i, 0]
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = corrupt_tail_c(tId, batch_data[i, 0], batch_data[i, 1], kwargs.ent_num)
                        batch_labels[tmp, 0] = -1.
                    else:
                        batch_data[tmp, 0] = corrupt_head_c(tId, batch_data[i, 2], batch_data[i, 1], kwargs.ent_num)
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = batch_data[i, 2]
                        batch_labels[tmp, 0] = 1.
                else:
                    if kwargs.normal_or_cross == 1:
                        kwargs.mode = 0 - kwargs.mode
                    if kwargs.mode == -1:
                        batch_data[tmp, 0] = corrupt_head_c(tId, batch_data[i, 2], batch_data[i, 1], kwargs.ent_num)
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = batch_data[i, 2]
                        batch_labels[tmp, 0] = 1.
                    else:
                        batch_data[tmp, 0] = batch_data[i, 0]
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = corrupt_tail_c(tId, batch_data[i, 0], batch_data[i, 1], kwargs.ent_num)
                        batch_labels[tmp, 0] = -1.

@boundscheck(False)
@wraparound(False)
cdef void generate_triple_with_negative(long[:, ::1] batch_data, float[:, ::1] batch_labels, Kwargs *kwargs):
    cdef:
        int i, j, tmp, tId
        int lef, rig
        int start
        float prob, p
    
    if kwargs.TID == NULL:
        kwargs.TID = <int*>global_mem.alloc(train_data.data_size, sizeof(int))
        for i in range(train_data.data_size):
            kwargs.TID[i] = i
        kwargs.start = 0
    
    start = kwargs.start
    if start == 0:
        shuffle(kwargs.TID, train_data.data_size)
    
    for tId in prange(kwargs.num_threads, nogil=True, schedule='static', num_threads=kwargs.num_threads):
        lef = tId * kwargs.num_per_thread
        rig = min((tId + 1) * kwargs.num_per_thread, kwargs.batch_size)
        for i in range(lef, rig):
            tmp =  kwargs.TID[start + i]
            batch_data[i, 0] = train_data.data[tmp].head
            batch_data[i, 1] = train_data.data[tmp].rel
            batch_data[i, 2] = train_data.data[tmp].tail
            batch_labels[i, 0] = 0.
            for j in range(1, kwargs.negative_sample_size + 1):
                tmp = i + j * kwargs.batch_size
                if kwargs.mode == 0 and kwargs.normal_or_cross == 0:
                    if kwargs.bern_flag:
                        p = (train_data.rig_mean[batch_data[i, 1]] + train_data.lef_mean[batch_data[i, 1]])
                        prob = 1000. * train_data.rig_mean[batch_data[i, 1]]
                    else:
                        p = 1.
                        prob = 500.
                    if _rand64(tId) % 1000 * p < prob:
                        batch_data[tmp, 0] = batch_data[i, 0]
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = corrupt_tail_c(tId, batch_data[i, 0], batch_data[i, 1], kwargs.ent_num)
                        batch_labels[tmp, 0] = -1.
                    else:
                        batch_data[tmp, 0] = corrupt_head_c(tId, batch_data[i, 2], batch_data[i, 1], kwargs.ent_num)
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = batch_data[i, 2]
                        batch_labels[tmp, 0] = 1.
                else:
                    if kwargs.normal_or_cross == 1:
                        kwargs.mode = 0 - kwargs.mode
                    if kwargs.mode == -1:
                        batch_data[tmp, 0] = corrupt_head_c(tId, batch_data[i, 2], batch_data[i, 1], kwargs.ent_num)
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = batch_data[i, 2]
                        batch_labels[tmp, 0] = 1.
                    else:
                        batch_data[tmp, 0] = batch_data[i, 0]
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = corrupt_tail_c(tId, batch_data[i, 0], batch_data[i, 1], kwargs.ent_num)
                        batch_labels[tmp, 0] = -1.
    kwargs.start += kwargs.batch_size

@boundscheck(False)
@wraparound(False)
cdef void generate_pair_on_random(long[:, ::1] batch_data, float[:, ::1] batch_labels, Kwargs *kwargs):
    cdef:
        int i, j, tmp, tId, lef, rig, lef_id, rig_id
        float y_label
        int size = train_data.lef_pair_num + train_data.rig_pair_num
    
    if kwargs.smooth_lambda > 0.:
        y_label = 1. - kwargs.smooth_lambda + 1.0 / kwargs.ent_num
        batch_labels[...] = 1.0 / kwargs.ent_num
    else:
        y_label = 1.
        batch_labels[...] = 0.

    for tId in prange(kwargs.num_threads, nogil=True, schedule='static', num_threads=kwargs.num_threads):
        lef = tId * kwargs.num_per_thread
        rig = min((tId + 1) * kwargs.num_per_thread, kwargs.batch_size)
        for i in range(lef, rig):
            if rand_max(tId, size) < 1. * train_data.lef_pair_num:
                tmp = rand_max(tId, train_data.lef_pair_num)
                batch_data[i, 0] = train_data.pair_tail_idx[tmp].ent
                batch_data[i, 1] = train_data.pair_tail_idx[tmp].rel
                lef_id, rig_id = find_target_id(train_data.pair_tail_idx, train_data.pair_lef_head, train_data.pair_rig_head, batch_data[i, 0], batch_data[i, 1])
                for j in range(lef_id, rig_id + 1):
                    batch_labels[i, train_data.data_head[j].tail] = y_label
            else:
                tmp = rand_max(tId, train_data.rig_pair_num)
                batch_data[i, 0] = train_data.pair_head_idx[tmp].ent
                batch_data[i, 1] = train_data.pair_head_idx[tmp].rel + kwargs.rel_num
                lef_id, rig_id = find_target_id(train_data.pair_head_idx, train_data.pair_lef_tail, train_data.pair_rig_tail, batch_data[i, 0], batch_data[i, 1]-kwargs.rel_num)
                for j in range(lef_id, rig_id + 1):
                    batch_labels[i, train_data.data_tail[j].head] = y_label

@boundscheck(False)
@wraparound(False)
cdef void generate_pair(long[:, ::1] batch_data, float[:, ::1] batch_labels, Kwargs *kwargs):
    cdef:
        int i, j, tmp, tId, lef, rig, lef_id, rig_id, start
        float y_label
        int size = train_data.lef_pair_num + train_data.rig_pair_num
    
    if kwargs.smooth_lambda > 0.:
        y_label = 1. - kwargs.smooth_lambda + 1.0 / kwargs.ent_num
        batch_labels[...] = 1.0 / kwargs.ent_num
    else:
        y_label = 1.
        batch_labels[...] = 0.

    if kwargs.TID == NULL:
        kwargs.TID = <int*>global_mem.alloc(size, sizeof(int))
        for i in range(size):
            kwargs.TID[i] = i
        kwargs.start = 0
    
    start = kwargs.start
    if start == 0:
        shuffle(kwargs.TID, size)
    
    for tId in prange(kwargs.num_threads, nogil=True, schedule='static', num_threads=kwargs.num_threads):
        lef = tId * kwargs.num_per_thread
        rig = min((tId + 1) * kwargs.num_per_thread, kwargs.batch_size)
        for i in range(lef, rig):
            tmp = kwargs.TID[start + i]
            if tmp < train_data.lef_pair_num:
                batch_data[i, 0] = train_data.pair_tail_idx[tmp].ent
                batch_data[i, 1] = train_data.pair_tail_idx[tmp].rel
                lef_id, rig_id = find_target_id(train_data.pair_tail_idx, train_data.pair_lef_head, train_data.pair_rig_head, batch_data[i, 0], batch_data[i, 1])
                for j in range(lef_id, rig_id + 1):
                    batch_labels[i, train_data.data_head[j].tail] = y_label
            else:
                tmp = tmp - train_data.lef_pair_num
                batch_data[i, 0] = train_data.pair_head_idx[tmp].ent
                batch_data[i, 1] = train_data.pair_head_idx[tmp].rel + kwargs.rel_num
                lef_id, rig_id = find_target_id(train_data.pair_head_idx, train_data.pair_lef_tail, train_data.pair_rig_tail, batch_data[i, 0], batch_data[i, 1]-kwargs.rel_num)
                for j in range(lef_id, rig_id + 1):
                    batch_labels[i, train_data.data_tail[j].head] = y_label
    kwargs.start += kwargs.batch_size


cdef class Sample:
    cdef:
        int _num_batch   # number of batches in per epoch
        int _batch_size  # batch size
        int _residue
        int _j
        int _element_type
        bint _have_initialized
        Kwargs kwargs
    
    def __init__(self, ent_num: int, rel_num: int, batch_size: int=128, num_threads: int=2, smooth_lambda: float=0.1,
                 negative_rate: int=1, mode: int=0, bern_flag: bint=0, seed: int=2357, element_type: int=0):
        setRandSeed(seed)
        if num_threads < 1:
            num_threads = 1
        randReset(num_threads)
        
        initializeKwargs(&self.kwargs)
        self._batch_size = batch_size
        self.kwargs.num_threads = num_threads
        self.kwargs.smooth_lambda = smooth_lambda
        self.kwargs.ent_num = ent_num
        self.kwargs.rel_num = rel_num
        self.kwargs.negative_sample_size = negative_rate
        self.kwargs.bern_flag = bern_flag

        self._num_batch = 0
        self._have_initialized = True
        self._residue = 0
        self._j = batch_size
        self._element_type = 0

        if mode < 2:
            self.kwargs.mode = mode
            self.kwargs.normal_or_cross = 0
        else:
            self.kwargs.mode = 1
            self.kwargs.normal_or_cross = 1

        self._element_type = element_type
        
        if element_type == 0:
            self._init_training_hyparameter(flags=True, num_batch=self._num_batch, batch_size=batch_size)
        else:
            self._init_training_hyparameter(flags=True, num_batch=self._num_batch, batch_size=batch_size)

        # printf('thread: %d\n', self.kwargs.num_threads)
    
    def _init_training_hyparameter(self, flags: bool=True, num_batch: int=0, batch_size: int=128):
        cdef:
            int tmp_num_batch, tmp_batch_size
            int size
        
        if not flags:
            self._num_batch = num_batch
            self._batch_size = batch_size
            return

        if self._element_type == 0:
            size = train_data.data_size
        else:
            size = train_data.lef_pair_num + train_data.rig_pair_num

        if num_batch == 0:
            self._num_batch = size // batch_size
            if size % batch_size != 0:
                self._num_batch += 1
                self._residue = size % batch_size
            self._batch_size = batch_size
        else:
            
            if size % num_batch == 0:
                self._batch_size = size // num_batch
            
            else:
                if size % (num_batch - 1) == 0:
                    self._batch_size = size // (num_batch - 1)
                    self._num_batch = num_batch - 1
                    return
                
                if size % (num_batch + 1) == 0:
                    self._batch_size = size // (num_batch + 1)
                    self._num_batch = num_batch + 1
                
                else:
                    if batch_size == 128:
                        self._batch_size = size // num_batch
                        self._num_batch = num_batch + 1
                        self._residue = size - num_batch * self._batch_size
                    else:
                        self._num_batch = size // batch_size
                        if size % batch_size != 0:
                            self._num_batch += 1
                            self._residue = size % batch_size
                        self._batch_size = batch_size
    
    def reset_hyparameters(self, flags: bool=True, num_batch: int=0, batch_size: int=128):
        self._init_training_hyparameter(flags=flags, num_batch=num_batch, batch_size=batch_size)
        self._have_initialized = True
    

    def get_size(self, flags='triple'):
        assert flags in ['triple', 'pair'], 'flags should be triple or pair'
        if flags == 'triple':
            return train_data.data_size
        else:
            return train_data.lef_pair_num + train_data.rig_pair_num

    
    def get_batch_size(self):
        return self._j
    
    def _set_num_per_thread(self):
        cdef int n, m
        if self._batch_size > self.kwargs.num_threads:
            n = self._batch_size // self.kwargs.num_threads
            m = self._batch_size % self.kwargs.num_threads
            self.kwargs.num_per_thread = n if m == 0 else n + 1
        else:
            self.kwargs.num_threads = self._batch_size
            self.kwargs.num_per_thread = 1
    
    def generate_triple_with_negative(self):
        cdef:
            bint flag
            int i, j, tmp
            long[:, ::1] batch_data = np.empty(((self._batch_size+1) * (1 + self.kwargs.negative_sample_size), 3), dtype=np.int64)
            float[:, ::1] batch_labels = np.empty(((self._batch_size+1) * (1 + self.kwargs.negative_sample_size), 1), dtype=np.float32)
        if self._have_initialized:
            
            self._set_num_per_thread()
            self._have_initialized = False

        for i in range(self._num_batch):
            if i == self._num_batch - 1:
                j = self._residue
            else:
                j = self._batch_size
            
            self.kwargs.batch_size = j
            self._j = j
            j *= (self.kwargs.negative_sample_size + 1)
            generate_triple_with_negative(batch_data[:j, :], batch_labels[:j, :], &self.kwargs)
            yield np.array(batch_data[:j, :], copy=False), np.array(batch_labels[:j, :], copy=False)
        else:
            self.kwargs.start = 0
    
    def generate_triple_with_negative_on_random(self):
        if self._have_initialized:            
            self._set_num_per_thread()
            self._have_initialized = False
        
        cdef:
            int i
            long[:, ::1] batch_data = np.empty((self._batch_size * (1 + self.kwargs.negative_sample_size), 3), dtype=np.int64)
            float[:, ::1] batch_labels = np.empty((self._batch_size * (1 + self.kwargs.negative_sample_size), 1), dtype=np.float32)
        
        self.kwargs.batch_size = self._batch_size
        self._j = self._batch_size

        for i in range(self._num_batch):
            generate_triple_with_negative_on_random(batch_data, batch_labels, &self.kwargs)
            yield np.array(batch_data), np.array(batch_labels)
        
    def generate_pair(self):
        cdef:
            bint flag
            int i, j, tmp
            long[:, ::1] batch_data = np.empty(((self._batch_size+1), 2), dtype=np.int64)
            float[:, ::1] batch_labels = np.empty(((self._batch_size+1), self.kwargs.ent_num), dtype=np.float32)
        if self._have_initialized:            
            self._set_num_per_thread()
            self._have_initialized = False

        for i in range(self._num_batch):
            if i == self._num_batch - 1:
                j = self._residue
            else:
                j = self._batch_size
            
            self._j = j
            self.kwargs.batch_size = j
            generate_pair(batch_data[:j, :], batch_labels[:j, :], &self.kwargs)
            yield np.array(batch_data[:j, :], copy=False), np.array(batch_labels[:j, :], copy=False)
        else:
            self.kwargs.start = 0
    
    def generate_pair_on_random(self):
        
        if self._have_initialized:            
            self._set_num_per_thread()
            self._have_initialized = False
        
        cdef:
            int i
            long[:, ::1] batch_data = np.empty((self._batch_size, 2), dtype=np.int64)
            float[:, ::1] batch_labels = np.empty((self._batch_size, self.kwargs.ent_num), dtype=np.float32)

        self.kwargs.batch_size = self._batch_size
        self._j = self._batch_size

        for i in range(self._num_batch):
            generate_pair_on_random(batch_data, batch_labels, &self.kwargs)
            yield np.array(batch_data), np.array(batch_labels)
    
    property num_batch:
        def __get__(self):
            return self._num_batch
        def __set__(self, int value):
            self._num_batch = value
    
    property batch_size:
        def __get__(self):
            return self.get_batch_size()
        def __set__(self, int value):
            self._batch_size = value

def find(head: int, rel: int, tail: int):
    print('...........................')

    cdef int i, lef, rig, n, m

    # n = train_data.pair_lef_head[<int>head]
    # m = train_data.pair_rig_head[<int>head]
    # printf('target ent: %d, rel: %d\n', head, rel)
    # printf('find target lef id: %d, ent: %ld, rel: %ld\n', n, train_data.pair_tail_idx[n].ent, train_data.pair_tail_idx[n].rel)
    # printf('find target rig id: %d, ent: %ld, rel: %ld\n', m, train_data.pair_tail_idx[m].ent, train_data.pair_tail_idx[m].rel)

    # for i in range(n, m+1):
    #     printf('find target id: %d, ent: %ld, rel: %ld\n', i, train_data.pair_tail_idx[i].ent, train_data.pair_tail_idx[i].rel)
    #     if i > 20:
    #         break

    lef, rig = find_target_id(train_data.pair_tail_idx, train_data.pair_lef_head, train_data.pair_rig_head, head, rel)

    printf('lef_id: %d, rig_id: %d\n', lef, rig)

    for i in range(lef, rig+1):
        yield train_data.data_head[i].head, train_data.data_head[i].rel, train_data.data_head[i].tail




