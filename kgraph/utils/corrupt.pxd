# cython: language_level = 3
# distutils: language = c++
from .cache_data cimport Pair, Triple, Data
from .cache_data cimport Constrain
from .read cimport train_data
from .random_int64 cimport rand_max
from .random_int64 cimport rand64

cdef (int, int) find_target_id(Pair *ptr, int *pair_lef, int *pair_rig, long ent, long rel) nogil

cdef long corrupt_tail_c(int tId, long head, long rel, int entityTotal) nogil

cdef long corrupt_head_c(int tId, long tail, long rel, int entityTotal) nogil

cdef bint find(Data *ptr, long head, long rel, long tail) nogil

cdef long corrupt_head_with_constrain(int tId, Data *ptr, Constrain *constrain, long head, long rel, int entityTotal) nogil

cdef long corrupt_tail_with_constrain(int tId, Data *ptr, Constrain *constrain, long tail, long rel, int entityTotal) nogil
