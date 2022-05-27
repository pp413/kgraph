# cython: language_level = 3
# distutils: language = c++

from .memory cimport rand64, rand_max
from .memory cimport Pair, Triple, DataStruct
from .memory cimport Constrain

# from .read cimport train_data

cdef (int, int) find_target_id(Pair *ptr, int *pair_lef, int *pair_rig, int ent, int rel) nogil

cdef int corrupt_tail_c(DataStruct* train_data, int tId, int head, int rel, int entityTotal, int flag) nogil

cdef int corrupt_head_c(DataStruct* train_data, int tId, int tail, int rel, int entityTotal, int flag) nogil

cdef bint find(DataStruct *ptr, int head, int rel, int tail) nogil

cdef int corrupt_head_with_constrain(int tId, DataStruct *ptr, Constrain *constrain, int head, int rel, int entityTotal) nogil

cdef int corrupt_tail_with_constrain(int tId, DataStruct *ptr, Constrain *constrain, int tail, int rel, int entityTotal) nogil
