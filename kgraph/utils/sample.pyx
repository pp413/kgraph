# cython: language_level=3
# distutils: language=c++
import numpy as np
cimport numpy as np
from cython cimport int, float, sizeof
from cython cimport boundscheck, wraparound
from cython.parallel cimport prange, parallel
from libc.stdio cimport printf
# from memory cimport Pool

from .memory cimport setRandSeed
from .memory cimport randReset
from .memory cimport rand_max
from .memory cimport rand64
from .memory cimport _rand64
from .memory cimport DataStruct
from .memory cimport MemoryPool
from .read cimport DataSet
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
    int use_shuffle
    int train_data_size

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
    kwargs.use_shuffle = 1
    kwargs.train_data_size = 0

@boundscheck(False)
@wraparound(False)
cdef void generate_triple_with_negative_on_random(DataStruct *train_data_ptr, int[:, ::1] batch_data, float[:, ::1] batch_labels, Kwargs *kwargs) nogil:
    cdef:
        int i, j, tmp, tId
        int lef, rig
        float prob, p
        DataStruct* data_ptr = train_data_ptr
    
    for tId in prange(kwargs.num_threads, nogil=True, schedule='static', num_threads=kwargs.num_threads):
        lef = tId * kwargs.num_per_thread
        rig = min((tId + 1) * kwargs.num_per_thread, kwargs.batch_size)
        for i in range(lef, rig):
            tmp = rand_max(tId, data_ptr.data_size, 1)
            batch_data[i, 0] = data_ptr.data[tmp].head
            batch_data[i, 1] = data_ptr.data[tmp].rel
            batch_data[i, 2] = data_ptr.data[tmp].tail
            batch_labels[i, 0] = 0.
            if kwargs.bern_flag:
                p = (data_ptr.rig_mean[batch_data[i, 1]] + data_ptr.lef_mean[batch_data[i, 1]])
                prob = 1000. * data_ptr.rig_mean[batch_data[i, 1]]
            else:
                p = 1.
                prob = 500.
            for j in range(1, kwargs.negative_sample_size + 1):
                tmp = i + j * kwargs.batch_size
                if kwargs.mode == 0 and kwargs.normal_or_cross == 0:
                    if (_rand64(tId) % 1000) * p < prob:
                        batch_data[tmp, 0] = batch_data[i, 0]
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = corrupt_tail_c(data_ptr, tId, batch_data[i, 0], batch_data[i, 1], kwargs.ent_num, 1)
                        batch_labels[tmp, 0] = -1.
                    else:
                        batch_data[tmp, 0] = corrupt_head_c(data_ptr, tId, batch_data[i, 2], batch_data[i, 1], kwargs.ent_num, 1)
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = batch_data[i, 2]
                        batch_labels[tmp, 0] = 1.
                else:
                    if kwargs.normal_or_cross == 1:
                        kwargs.mode = 0 - kwargs.mode
                    if kwargs.mode == 1:
                        batch_data[tmp, 0] = corrupt_head_c(data_ptr, tId, batch_data[i, 2], batch_data[i, 1], kwargs.ent_num, 1)
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = batch_data[i, 2]
                        batch_labels[tmp, 0] = 1.
                    else:
                        batch_data[tmp, 0] = batch_data[i, 0]
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = corrupt_tail_c(data_ptr, tId, batch_data[i, 0], batch_data[i, 1], kwargs.ent_num, 1)
                        batch_labels[tmp, 0] = -1.

@boundscheck(False)
@wraparound(False)
cdef void generate_triple_with_negative(DataStruct *train_data, int[:, ::1] batch_data, float[:, ::1] batch_labels, Kwargs *kwargs, MemoryPool tmp_memory_pool):
    cdef:
        int i, j, tmp, tId
        int lef, rig
        int start
        float prob, p
        DataStruct* data_ptr = train_data
        # MemoryPool tmp_memory_pool
    
    # tmp_memory_pool = MemoryPool()
    
    if kwargs.TID == NULL:
        kwargs.TID = <int*>tmp_memory_pool.alloc(data_ptr.data_size, sizeof(int))
        for i in range(data_ptr.data_size):
            kwargs.TID[i] = i
        kwargs.start = 0
    
    start = kwargs.start
    if start == 0:
        shuffle(kwargs.TID, data_ptr.data_size)
    
    for tId in prange(kwargs.num_threads, nogil=True, schedule='static', num_threads=kwargs.num_threads):
        lef = tId * kwargs.num_per_thread
        rig = min((tId + 1) * kwargs.num_per_thread, kwargs.batch_size)
        for i in range(lef, rig):
            tmp =  kwargs.TID[start + i]
            batch_data[i, 0] = data_ptr.data[tmp].head
            batch_data[i, 1] = data_ptr.data[tmp].rel
            batch_data[i, 2] = data_ptr.data[tmp].tail
            batch_labels[i, 0] = 0.
            if kwargs.bern_flag:
                p = (data_ptr.rig_mean[batch_data[i, 1]] + data_ptr.lef_mean[batch_data[i, 1]])
                prob = 1000. * data_ptr.rig_mean[batch_data[i, 1]]
            else:
                p = 1.
                prob = 500.
            for j in range(1, kwargs.negative_sample_size + 1):
                tmp = i + j * kwargs.batch_size
                if kwargs.mode == 0 and kwargs.normal_or_cross == 0:
                    if _rand64(tId) % 1000 * p < prob:
                        batch_data[tmp, 0] = batch_data[i, 0]
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = corrupt_tail_c(data_ptr, tId, batch_data[i, 0], batch_data[i, 1], kwargs.ent_num, 1)
                        batch_labels[tmp, 0] = -1.
                    else:
                        batch_data[tmp, 0] = corrupt_head_c(data_ptr, tId, batch_data[i, 2], batch_data[i, 1], kwargs.ent_num, 1)
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = batch_data[i, 2]
                        batch_labels[tmp, 0] = 1.
                else:
                    if kwargs.normal_or_cross == 1:
                        kwargs.mode = 0 - kwargs.mode
                    if kwargs.mode == 1:
                        batch_data[tmp, 0] = corrupt_head_c(data_ptr, tId, batch_data[i, 2], batch_data[i, 1], kwargs.ent_num, 1)
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = batch_data[i, 2]
                        batch_labels[tmp, 0] = 1.
                    else:
                        batch_data[tmp, 0] = batch_data[i, 0]
                        batch_data[tmp, 1] = batch_data[i, 1]
                        batch_data[tmp, 2] = corrupt_tail_c(data_ptr, tId, batch_data[i, 0], batch_data[i, 1], kwargs.ent_num, 1)
                        batch_labels[tmp, 0] = -1.
    kwargs.start += kwargs.batch_size


@boundscheck(False)
@wraparound(False)
cdef void generate_pair_on_random(DataStruct *train_data_ptr, int[:, ::1] batch_data, float[:, ::1] batch_labels, Kwargs *kwargs):
    cdef:
        int i, j, tmp, tId, lef, rig, lef_id, rig_id
        float y_label
        DataStruct* data_ptr = train_data_ptr
        int size = kwargs.train_data_size
    
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
            if rand_max(tId, size, 1) < 1. * data_ptr.lef_pair_num:
                tmp = rand_max(tId, data_ptr.lef_pair_num, 1)
                batch_data[i, 0] = data_ptr.pair_tail_idx[tmp].ent
                batch_data[i, 1] = data_ptr.pair_tail_idx[tmp].rel
                lef_id, rig_id = find_target_id(data_ptr.pair_tail_idx, data_ptr.pair_lef_head, data_ptr.pair_rig_head, batch_data[i, 0], batch_data[i, 1])
                for j in range(lef_id, rig_id + 1):
                    batch_labels[i, data_ptr.data_head[j].tail] = y_label
            else:
                tmp = rand_max(tId, data_ptr.rig_pair_num, 1)
                batch_data[i, 0] = data_ptr.pair_head_idx[tmp].ent
                batch_data[i, 1] = data_ptr.pair_head_idx[tmp].rel + kwargs.rel_num
                lef_id, rig_id = find_target_id(data_ptr.pair_head_idx, data_ptr.pair_lef_tail, data_ptr.pair_rig_tail, batch_data[i, 0], batch_data[i, 1]-kwargs.rel_num)
                for j in range(lef_id, rig_id + 1):
                    batch_labels[i, data_ptr.data_tail[j].head] = y_label

@boundscheck(False)
@wraparound(False)
cdef void generate_pair(DataStruct *train_data_ptr, int[:, ::1] batch_data, float[:, ::1] batch_labels, Kwargs *kwargs, MemoryPool tmp_memory_pool):
    cdef:
        int i, j, tmp, tId, lef, rig, lef_id, rig_id, start
        float y_label
        DataStruct* data_ptr = train_data_ptr
        int size = kwargs.train_data_size
        # MemoryPool tmp_memory_pool
    
    # tmp_memory_pool = MemoryPool()
    
    if kwargs.smooth_lambda > 0.:
        y_label = 1. - kwargs.smooth_lambda + 1.0 / kwargs.ent_num
        batch_labels[...] = 1.0 / kwargs.ent_num
    else:
        y_label = 1.
        batch_labels[...] = 0.

    # printf('size: %d\n', size)
    if kwargs.TID == NULL:
        kwargs.TID = <int*>tmp_memory_pool.alloc(size, sizeof(int))
        for i in range(size):
            kwargs.TID[i] = i
        kwargs.start = 0
    
    start = kwargs.start
    if start == 0:
        shuffle(kwargs.TID, size)

    # printf('start: %d\n', start)
    
    for tId in prange(kwargs.num_threads, nogil=True, schedule='static', num_threads=kwargs.num_threads):
        lef = tId * kwargs.num_per_thread
        rig = min((tId + 1) * kwargs.num_per_thread, kwargs.batch_size)
        # printf('lef: %d, rig: %d\n', lef, rig)
        for i in range(lef, rig):
            tmp = kwargs.TID[start + i]
            # printf('start+i: %d, tmp: %d\n', start + i, tmp)
            if tmp < data_ptr.lef_pair_num:
                batch_data[i, 0] = data_ptr.pair_tail_idx[tmp].ent
                batch_data[i, 1] = data_ptr.pair_tail_idx[tmp].rel
                # printf('1\n')
                lef_id, rig_id = find_target_id(data_ptr.pair_tail_idx, data_ptr.pair_lef_head, data_ptr.pair_rig_head, batch_data[i, 0], batch_data[i, 1])
                for j in range(lef_id, rig_id + 1):
                    batch_labels[i, data_ptr.data_head[j].tail] = y_label
                # printf('11\n')
            else:
                tmp = tmp - data_ptr.lef_pair_num
                batch_data[i, 0] = data_ptr.pair_head_idx[tmp].ent
                batch_data[i, 1] = data_ptr.pair_head_idx[tmp].rel + kwargs.rel_num
                # printf('2\n')
                lef_id, rig_id = find_target_id(data_ptr.pair_head_idx, data_ptr.pair_lef_tail, data_ptr.pair_rig_tail, batch_data[i, 0], data_ptr.pair_head_idx[tmp].rel)
                for j in range(lef_id, rig_id + 1):
                    batch_labels[i, data_ptr.data_tail[j].head] = y_label
                # printf('22\n')
            # printf('4\n')
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
        DataStruct * data_ptr
        MemoryPool tmp_memory_pool
    
    def __init__(self, data: DataSet, batch_size: int=128, num_threads: int=2, smooth_lambda: float=0.1, use_shuffle: int=1,
                 negative_rate: int=1, mode: int=0, bern_flag: bint=0, seed: int=41504, element_type: int=0):
        setRandSeed(seed)
        if num_threads < 1:
            num_threads = 1
        randReset(num_threads)
        
        initializeKwargs(&self.kwargs)
        self._batch_size = batch_size
        self.kwargs.num_threads = num_threads
        self.kwargs.smooth_lambda = smooth_lambda
        self.kwargs.ent_num = data.num_ent
        self.kwargs.rel_num = data.num_rel
        self.kwargs.negative_sample_size = negative_rate
        self.kwargs.bern_flag = bern_flag
        self.kwargs.use_shuffle = use_shuffle

        self.tmp_memory_pool = MemoryPool()

        self._num_batch = 0
        self._have_initialized = True
        self._residue = 0
        self._j = batch_size
        self.data_ptr = <DataStruct*>data.getTrainDataPtr()
        self.kwargs.train_data_size = self.data_ptr.lef_pair_num + self.data_ptr.rig_pair_num

        if mode < 2:
            self.kwargs.mode = mode
            self.kwargs.normal_or_cross = 0
        else:
            self.kwargs.mode = 1
            self.kwargs.normal_or_cross = 1

        self._element_type = element_type
        
        self._init_training_hyparameter(flags=True, num_batch=self._num_batch, batch_size=batch_size)

    
    cdef _init_training_hyparameter(self, flags: bool=True, num_batch: int=0, batch_size: int=128):
        cdef:
            int tmp_num_batch, tmp_batch_size
            int size
        
        if not flags:
            self._num_batch = num_batch
            self._batch_size = batch_size
            return

        if self._element_type == 0:
            size = self.data_ptr.data_size
        else:
            size = self.data_ptr.lef_pair_num + self.data_ptr.rig_pair_num

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
            return self.data_ptr.data_size
        else:
            return self.data_ptr.lef_pair_num + self.data_ptr.rig_pair_num

    
    cdef get_batch_size(self):
        return self._j
    
    cdef _set_num_per_thread(self):
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
            int tmp_batch_size = (self._batch_size+1) * (1 + self.kwargs.negative_sample_size)
            int[:, ::1] batch_data = <int[:tmp_batch_size, :3]>(<int *>self.tmp_memory_pool.alloc(tmp_batch_size * 3, sizeof(int)))
            float[:, ::1] batch_labels = <float[:tmp_batch_size, :1]>(<float *>self.tmp_memory_pool.alloc(tmp_batch_size, sizeof(float)))
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
            generate_triple_with_negative(self.data_ptr, batch_data[:j, :], batch_labels[:j, :], &self.kwargs, self.tmp_memory_pool)
            yield np.array(batch_data[:j, :], dtype=np.int32), np.array(batch_labels[:j, :], dtype=np.float32)
        else:
            self.kwargs.start = 0
    
    def generate_triple_with_negative_on_random(self):
        if self._have_initialized:            
            self._set_num_per_thread()
            self._have_initialized = False
        
        cdef:
            int i
            int tmp_batch_size = self._batch_size * (1 + self.kwargs.negative_sample_size)
            # int[:, ::1] batch_data = np.empty((tmp_batch_size, 3), dtype=np.int64)
            int[:, ::1] batch_data = <int[:tmp_batch_size, :3]>(<int *>self.tmp_memory_pool.alloc(tmp_batch_size * 3, sizeof(int)))
            # float[:, ::1] batch_labels = np.empty((tmp_batch_size, 1), dtype=np.float32)
            float[:, ::1] batch_labels = <float[:tmp_batch_size, :1]>(<float *>self.tmp_memory_pool.alloc(tmp_batch_size, sizeof(float)))
        
        self.kwargs.batch_size = self._batch_size
        self._j = self._batch_size

        for i in range(self._num_batch):
            generate_triple_with_negative_on_random(self.data_ptr, batch_data, batch_labels, &self.kwargs)
            yield np.array(batch_data, dtype=np.int32), np.array(batch_labels, dtype=np.float32)
        
    def generate_pair(self):
        cdef:
            bint flag
            int i, j, tmp
            int tmp_batch_size = self._batch_size+1
            int[:, ::1] batch_data = <int[:tmp_batch_size, :2]>(<int *>self.tmp_memory_pool.alloc(tmp_batch_size * 2, sizeof(int)))
            float[:, ::1] batch_labels = <float[:tmp_batch_size, :self.kwargs.ent_num]>(<float *>self.tmp_memory_pool.alloc(tmp_batch_size * self.kwargs.ent_num, sizeof(float)))
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
            # print('batch_size: ', self.kwargs.batch_size, i)
            generate_pair(self.data_ptr, batch_data[:j, :], batch_labels[:j, :], &self.kwargs, self.tmp_memory_pool)
            yield np.array(batch_data[:j, :], dtype=np.int32), np.array(batch_labels[:j, :], dtype=np.float32)
        else:
            self.kwargs.start = 0
    
    def generate_pair_on_random(self):
        
        if self._have_initialized:            
            self._set_num_per_thread()
            self._have_initialized = False
        
        cdef:
            int i
            int tmp_batch_size = self._batch_size
            # int[:, ::1] batch_data = np.empty((self._batch_size, 2), dtype=np.int64)
            int[:, ::1] batch_data = <int[:tmp_batch_size, :2]>(<int *>self.tmp_memory_pool.alloc(tmp_batch_size * 2, sizeof(int)))
            # float[:, ::1] batch_labels = np.empty((self._batch_size, self.kwargs.ent_num), dtype=np.float32)
            float[:, ::1] batch_labels = <float[:tmp_batch_size, :self.kwargs.ent_num]>(<float *>self.tmp_memory_pool.alloc(tmp_batch_size * self.kwargs.ent_num, sizeof(float)))

        self.kwargs.batch_size = self._batch_size
        self._j = self._batch_size

        for i in range(self._num_batch):
            generate_pair_on_random(self.data_ptr, batch_data, batch_labels, &self.kwargs)
            yield np.array(batch_data, dtype=np.int32), np.array(batch_labels, dtype=np.float32)
    
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

