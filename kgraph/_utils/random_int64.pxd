# cython: language_level = 3
# distutils: language=c++
from .mem cimport Pool
from libc.stdio cimport printf
from libc.stdlib cimport rand, srand

cdef unsigned long long *next_random
cdef Pool rand_mem

cdef void setThreadNumberAndRandSeed(const int thread_number, const int seed)

cdef void setRandSeed(int seed)

cdef void randReset(int thread_number)

cdef unsigned long long _rand64(int tId) nogil

cdef long rand_max(const int tId, const long x) nogil

cdef long rand64(const long a, const long b) nogil
