# cython: language_level = 3
# distutils: language=c++
cimport numpy as np
from libcpp.algorithm cimport sort

from cython cimport sizeof

from libc.stdio cimport FILE, fopen
from libc.stdio cimport fscanf, printf
from libc.stdlib cimport exit, EXIT_FAILURE
from libc.string cimport memset
from libc.string cimport memcpy

from .mem cimport Pool
from .cache_data cimport global_mem, Constrain
from .cache_data cimport Data, Triple, Pair, global_mem, quick_sort
from .cache_data cimport cmp_head, cmp_tail, cmp_rel2, cmp_rel3, initializeData
from .cache_data cimport set_int_ptr, set_float_ptr, set_pair_ptr, set_triple_ptr
from .cache_data cimport load_triple_from_numpy

cdef Data all_triples
cdef Data train_data    # train data
cdef Data test_data     # test data
cdef Data valid_data    # valid data
cdef Constrain *type_constrain

cdef (int*, int*, int*, int*, Pair*, Pair*, int*, int*, int*, int*, int, int) _generate_index(
    Triple *dataHead, Triple *dataTail, int num, int num_ent)

cdef void putTrainInCache_c(long[:, ::1] data_array, int entityTotal, int relationTotal)

cdef void _putTestInCache(Data *_test_data, long[:, ::1] data, int entityTotal, int relationTotal)

cdef void putValidAndTestInCache_c(long[:, ::1] valid_data_array, long[:, ::1] test_data_array, int entityTotal, int relationTotal)

cdef void putAllInCache_c(long[:, ::1] train_data_array, long[:, ::1] valid_data_array, long[:, ::1] test_data_array, int entityTotal, int relationTotal)

cdef void get_constrain(Constrain **ptr, Data *data_ptr, int relationTotal)

cdef np.ndarray[long, ndim=2] getDataFromCache_c(Data *ptr)
