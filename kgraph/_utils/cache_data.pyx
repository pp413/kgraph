# cython: language_level = 3
# distutils: language=c++

cdef void initializeData(Data *ptr):
    ptr.data = NULL
    ptr.data_head = NULL
    ptr.data_tail = NULL
    ptr.freqEnt = NULL
    ptr.freqRel = NULL
    ptr.lefHead = NULL
    ptr.rigHead = NULL
    ptr.lefTail = NULL
    ptr.rigTail = NULL
    ptr.headList = NULL
    ptr.tailList = NULL
    ptr.lef_mean = NULL
    ptr.rig_mean = NULL
    ptr.pair_head_idx = NULL
    ptr.pair_tail_idx = NULL
    ptr.ent_total_data.rig_num = 0
    ptr.ent_total_data.lef_num = 0
    ptr.lef_pair_num = 0
    ptr.rig_pair_num = 0
    ptr.data_size = 0

cdef void set_int_ptr(int **ptr, int num, int flags):
    '''
    flags in [0, 1], 1: every element of data is -1.
    '''
    cdef int *tmp_ptr = ptr[0]
    ptr[0] = <int*>global_mem.alloc(num, sizeof(int))
    if tmp_ptr != NULL:
        global_mem.free(tmp_ptr)
    if flags != 0:
        memset(ptr[0], -1, num * sizeof(int))

cdef void set_float_ptr(float **ptr, int num):
    cdef float *tmp_ptr = ptr[0]
    ptr[0] = <float*>global_mem.alloc(num, sizeof(float))
    if tmp_ptr != NULL:
        global_mem.free(tmp_ptr)

cdef void set_pair_ptr(Pair **ptr, int num):
    cdef Pair *tmp_ptr = ptr[0]
    cdef Pair *_ptr = <Pair*>global_mem.alloc(num, sizeof(Pair))
    if tmp_ptr != NULL:
        global_mem.free(tmp_ptr)
    cdef int i
    for i in range(num):
        _ptr[i].lef_id = -1
        _ptr[i].rig_id = -1
    ptr[0] = _ptr

cdef void set_triple_ptr(Triple **ptr, int num):
    cdef Triple *tmp_ptr = ptr[0]
    ptr[0] = <Triple*>global_mem.alloc(num, sizeof(Triple))
    if tmp_ptr != NULL:
        global_mem.free(tmp_ptr)

cdef void load_triple_from_numpy(Triple* ptr, long[:, ::1] data):
    cdef int i
    cdef int n  = data.shape[0]
    if ptr == NULL:
        printf('ptr is NULL!\n')
    for i in range(n):
        # printf('i: %d, data_i: %d', i, data[i, 0])
        ptr[i].head = data[i, 0]
        ptr[i].rel = data[i, 1]
        ptr[i].tail = data[i, 2]

cdef bint cmp_head(Triple a, Triple b) nogil:
    return (a.head < b.head) or (a.head == b.head and a.rel < b.rel) or (a.head == b.head and a.rel == b.rel and a.tail < b.tail)

cdef bint cmp_tail(Triple a, Triple b) nogil:
    return (a.tail < b.tail) or (a.tail == b.tail and a.rel < b.rel) or (a.tail == b.tail and a.rel == b.rel and a.head < b.head)

cdef bint cmp_rel(Triple a, Triple b) nogil:
    return (a.head < b.head) or (a.head == b.head and a.tail < b.tail) or (a.head == b.head and a.tail == b.tail and a.rel < b.rel)

cdef bint cmp_rel2(Triple a, Triple b) nogil:
    return (a.rel < b.rel) or (a.rel == b.rel and a.head < b.head) or (a.rel == b.rel and a.head == b.head and a.tail < b.tail)

cdef bint cmp_rel3(Triple a, Triple b) nogil:
    return (a.rel < b.rel) or (a.rel == b.rel and a.tail < b.tail) or (a.rel == b.rel and a.tail == b.tail and a.head < b.head)

cdef void quick_sort(Triple *ptr, int num, cmp_type cmp):
    sort(ptr, ptr + num, cmp)


