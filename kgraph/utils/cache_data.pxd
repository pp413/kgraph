# cython: language_level = 3
# distutils: language=c++
cimport numpy as np
from .mem cimport Pool
from libc.string cimport memset
from libcpp.algorithm cimport sort
from cython cimport long, int, float, sizeof

from libc.stdio cimport printf

cdef Pool global_mem

ctypedef struct Pair:
    long ent, rel
    int lef_id, rig_id

ctypedef struct Triple:
    long head, rel, tail

ctypedef struct EntTotal_data:
    int lef_num, rig_num

ctypedef bint (*cmp_type)(Triple a, Triple b) nogil

ctypedef struct Data:
    Triple *data            # data
    Triple *data_head       # data based on the head sort
    Triple *data_tail       # data based on the tail sort
    int* freqEnt            # the frequency of entity in data set
    int* freqRel            # the frequency of relation in data set
    int* lefHead            # the left Id of head in data_head
    int* rigHead            # the right Id of head in data_head
    int* lefTail            # the left Id of tail in data_tail
    int* rigTail            # the right Id of tail in data_tail
    long* headList          # the list of head
    long* tailList          # the list of tail
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
    long *left_id_of_heads_of_relation
    long *right_id_of_heads_of_relation
    long *left_id_of_tails_of_relation
    long *right_id_of_tails_of_relation

cdef void initializeData(Data *ptr)

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

cdef void quick_sort(Triple* ptr, int num, cmp_type cmp)
