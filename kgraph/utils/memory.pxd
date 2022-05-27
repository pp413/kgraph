# cython: language_level = 3
# distutils: language=c++


from libcpp.vector cimport vector
from cython cimport long


# from cython cimport int, float


import warnings

from libc.stdio cimport printf

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size, int(*compar)(const_void *, const_void *)) nogil


cdef bint WARN_ZERO_ALLOC = False

cdef struct IndexedElement:
    int index
    float value

ctypedef struct Pair:
    int ent, rel
    int lef_id, rig_id

ctypedef struct Triple:
    int head, rel, tail  


ctypedef struct EntTotal_data:
    int lef_num, rig_num


ctypedef bint (*cmp_type)(Triple a, Triple b) nogil


ctypedef struct DataStruct:
    Triple *data            # data
    Triple *data_head       # data based on the head sort
    Triple *data_tail       # data based on the tail sort
    int* freqEnt            # the frequency of entity in data set
    int* freqRel            # the frequency of relation in data set
    int* lefHead            # the left Id of head in data_head
    int* rigHead            # the right Id of head in data_head
    int* lefTail            # the left Id of tail in data_tail
    int* rigTail            # the right Id of tail in data_tail
    int* headList          # the list of head
    int* tailList          # the list of tail
    float* lef_mean         # the mean of left entity(head) of relation in data set
    float* rig_mean         # the mean of right entity(tail) of relation in data set
    Pair* pair_head_idx     # (tail, relation) --> (leftId, rightId)
    Pair* pair_tail_idx     # (head, relation) --> (leftId, rightId)
    int* pair_lef_head        # leftId of head of pair_head_idx for finding.
    int* pair_rig_head        # rightId of head of pair_head_idx for finding.
    int* pair_lef_tail        # lefId of tail of pair_tail_idx for finding.
    int* pair_rig_tail       # rightId of tail of pair_tail_idx for finding.
    EntTotal_data ent_total_data   # the number of entities in data set
    int lef_pair_num, rig_pair_num  # the number of left and right pairs in data set
    int data_size            # the size of data set 

ctypedef struct Constrain:
    int *left_id_of_heads_of_relation
    int *right_id_of_heads_of_relation
    int *left_id_of_tails_of_relation
    int *right_id_of_tails_of_relation


cdef class Memory:
    
    cdef int* data


cdef class Data:

    cdef Py_ssize_t ncols
    cdef Py_ssize_t shape[2]
    cdef Py_ssize_t strides[2]
    cdef vector[int] v
    cdef int view_flag
    cdef Py_ssize_t start_i, end_i

    cdef add_row(self, int* new_row)


ctypedef void* (*malloc_t)(size_t n)
ctypedef void (*free_t)(void *p)


cdef class PyMalloc:
    cdef malloc_t malloc_func
    cdef void _set(self, malloc_t malloc)

cdef class PyFree:
    cdef free_t free_func
    cdef void _set(self, free_t free)

cdef PyMalloc WrapMalloc(malloc_t malloc)

cdef PyFree WrapFree(free_t free)

cdef class MemoryPool:

    cdef readonly Py_ssize_t size
    cdef readonly dict addresses
    cdef readonly list refs 
    cdef readonly PyMalloc pymalloc
    cdef readonly PyFree pyfree
    
    cdef void* alloc(self, Py_ssize_t number, Py_ssize_t elem_size) except NULL
    
    cdef void* realloc(self, void* p, Py_ssize_t new_size) except NULL
    
    cdef void free(self, void* p) except *

cdef void initializeData(DataStruct *ptr)

cdef MemoryPool global_memory_pool

cdef void set_int_ptr(int **ptr, int num, int flags)

cdef void set_float_ptr(float **ptr, int num)

cdef void set_pair_ptr(Pair **ptr, int num)

cdef void set_triple_ptr(Triple **ptr, int num)

cdef void load_triple_from_numpy(Triple* ptr, long[:, ::1] data)

cdef bint cmp_head(Triple a, Triple b) nogil

cdef bint cmp_tail(Triple a, Triple b) nogil

cdef bint cmp_rel(Triple a, Triple b) nogil

cdef bint cmp_rel2(Triple a, Triple b) nogil

cdef bint cmp_rel3(Triple a, Triple b) nogil

cdef void quick_sort(Triple *ptr, int num, cmp_type cmp)


cdef unsigned long long *next_random
cdef MemoryPool global_random_memory_pool


cdef void setThreadNumberAndRandSeed(const int thread_number, const int seed)

cdef void setRandSeed(int seed)

cdef void randReset(int thread_number)

cdef unsigned long long _rand64(int tId) nogil

cdef long rand_max(const int tId, const long x) nogil

cdef long rand64(const long a, const long b) nogil

cdef int _compare(const_void *a, const_void *b)