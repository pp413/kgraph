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
from .memory cimport global_memory_pool, Constrain
from .memory cimport DataStruct, Triple, Pair, quick_sort
from .memory cimport cmp_head, cmp_tail, cmp_rel2, cmp_rel3, initializeData
from .memory cimport set_int_ptr, set_float_ptr, set_pair_ptr, set_triple_ptr
from .memory cimport load_triple_from_numpy, Data

cdef DataStruct all_triples
cdef DataStruct train_data    # train data
cdef DataStruct test_data     # test data
cdef DataStruct valid_data   # valid data
cdef Constrain *type_constrain

cdef (int*, int*, int*, int*, Pair*, Pair*, int*, int*, int*, int*, int, int) _generate_index(
    Triple *dataHead, Triple *dataTail, int num, int num_ent)

cdef void putTrainInCache_c(long[:, ::1] data_array, int entityTotal, int relationTotal)

cdef void _putTestInCache(DataStruct *_test_data, long[:, ::1] data, int entityTotal, int relationTotal)

cdef void putValidAndTestInCache_c(long[:, ::1] valid_data_array, long[:, ::1] test_data_array, int entityTotal, int relationTotal)

cdef void putAllInCache_c(long[:, ::1] train_data_array, long[:, ::1] valid_data_array, long[:, ::1] test_data_array, int entityTotal, int relationTotal)

cdef void get_constrain(Constrain **ptr, DataStruct *data_ptr, int relationTotal)

cdef np.ndarray[long, ndim=2] getDataFromCache_c(DataStruct *ptr)
