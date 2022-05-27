# cython: language_level = 3
# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp.algorithm cimport sort

from cython cimport sizeof

from libc.stdio cimport FILE, fopen
from libc.stdio cimport fscanf, printf
from libc.stdlib cimport exit, EXIT_FAILURE
from libc.string cimport memset
from libc.string cimport memcpy

from .memory cimport MemoryPool
from .memory cimport Constrain
from .memory cimport DataStruct, Triple, Pair, quick_sort
from .memory cimport cmp_head, cmp_tail, cmp_rel2, cmp_rel3
from .memory cimport set_int_ptr, set_float_ptr, set_pair_ptr, set_triple_ptr

# cdef DataStruct all_triples
# cdef DataStruct train_data    # train data
# cdef DataStruct test_data     # test data
# cdef DataStruct valid_data   # valid data
cdef Constrain *type_constrain

cdef (int*, int*, int*, int*, Pair*, Pair*, int*, int*, int*, int*, int, int) _generate_index(
    Triple *dataHead, Triple *dataTail, int num, int num_ent, MemoryPool tmp_memory_pool)

cdef void putTrainInCache_c(DataStruct *data_ptr, int[:, ::1] data_array, int entityTotal, int relationTotal, MemoryPool tmp_memory_pool)

cdef void _putTestInCache(DataStruct *_test_data, int[:, ::1] data, int entityTotal, int relationTotal, MemoryPool tmp_memory_pool)

cdef void putValidInCache_c(DataStruct *valid_data, int[:, ::1] valid_data_array, int entityTotal, int relationTotal, MemoryPool tmp_memory_pool)

cdef void putTestInCache_c(DataStruct *test_data, int[:, ::1] test_data_array, int entityTotal, int relationTotal, MemoryPool tmp_memory_pool)

cdef void putAllInCache_c(DataStruct *all_triples, int[:, ::1] train_data_array, int[:, ::1] valid_data_array, int[:, ::1] test_data_array, int entityTotal, int relationTotal, MemoryPool tmp_memory_pool)

cdef void get_constrain(Constrain **ptr, DataStruct *data_ptr, int relationTotal, MemoryPool tmp_memory_pool)

cdef np.ndarray[int, ndim=2] getDataFromCache_c(DataStruct *ptr)

cdef class DataSet:
    cdef:
        DataStruct train_data_ptr
        DataStruct valid_data_ptr
        DataStruct test_data_ptr
        DataStruct all_triples_ptr

        int train_data_size
        int bern_flag, mode, normal_or_cross, num_neg
        float _smooth_lambda
        MemoryPool tmp_memory_pool

    cdef readonly:
        int num_ent
        int num_rel
        int element_type
    
    cdef DataStruct * getTrainDataPtr(self)
    
    cdef DataStruct * getValidDataPtr(self)
    
    cdef DataStruct * getTestDataPtr(self)
    
    cdef DataStruct * getAllTriplesPtr(self)
