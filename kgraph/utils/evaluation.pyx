# cython: language_level = 3
# distutils: language = c++
import numpy as np
from tqdm import tqdm, trange

cimport numpy as np
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cython cimport long, int, float, sizeof
# from libcpp.algorithm cimport sort

from .memory cimport MemoryPool
from .memory cimport Triple, DataStruct
from .memory cimport IndexedElement
from .memory cimport _compare, qsort

from .read cimport DataSet
from .corrupt cimport find_target_id


np.import_array()


# cdef extern from "stdlib.h":
#     ctypedef void const_void "const void"
#     void qsort(void *base, int nmemb, int size, int(*compar)(const_void *, const_void *)) nogil

cdef void argsort(int* index, float[::1] array) except *:
    '''花费时间更加低'''
    cdef:
        int tmp_i
        int length = array.shape[0]
        long long[::1] index_array = np.asarray(array).argsort()
    
    for tmp_i in range(length):
        index[tmp_i] = <int>index_array[tmp_i]


# cdef void argsort(int* index, float[::1] array) except *:
#     cdef int i
#     cdef int n = array.shape[0]
#     cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
#     for i in range(n):
#         order_struct[i].index = i
#         order_struct[i].value = array[i]
#     qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
#     for i in range(n):
#         index[i] = order_struct[i].index
#     free(order_struct)


def calculate_ranks_on_valid_via_triple(function, data: DataSet, batch_size: int):
    cdef:
        int i, j, k, n, m, tmp
        int tmp_i, tmp_j, lef_i, rig_i
        int filter_lef_i, filter_rig_i
        DataStruct *data_ptr = data.getValidDataPtr()
        DataStruct* train_data = data.getTrainDataPtr()
        int lef_num = data_ptr.lef_pair_num
        int rig_num = data_ptr.rig_pair_num
        int num_ent = data.num_ent
        int num_rel = data.num_rel

        MemoryPool lmem = MemoryPool()
        int *_triple = <int*>lmem.alloc(3 * num_ent, sizeof(int))
        int[:, ::1] triple = <int[:num_ent, :3]>_triple
        float *_scores = <float*>lmem.alloc(num_ent, sizeof(float))
        float[::1] scores = <float[:num_ent]>_scores
        np.ndarray[float, ndim=1] tmp_scores

        float *rhs_ranks = <float*>lmem.alloc(data_ptr.data_size, sizeof(float))
        float *rhs_franks = <float*>lmem.alloc(data_ptr.data_size, sizeof(float))
        float *lhs_ranks = <float*>lmem.alloc(data_ptr.data_size, sizeof(float))
        float *lhs_franks = <float*>lmem.alloc(data_ptr.data_size, sizeof(float))
        
        int *targets = <int*>lmem.alloc(num_ent, sizeof(int))
        int *targets_tmp = <int*>lmem.alloc(num_ent, sizeof(int))
    
    memset(_triple, -1, 3 * num_ent * sizeof(int))
    memset(rhs_ranks, -1, data_ptr.data_size * sizeof(float))
    memset(rhs_franks, -1, data_ptr.data_size * sizeof(float))
    memset(lhs_ranks, -1, data_ptr.data_size * sizeof(float))
    memset(lhs_franks, -1, data_ptr.data_size * sizeof(float))

    t_kwargs = dict(desc="Valid Evaluating:", unit="pair", ncols=80, initial=1, total=lef_num + rig_num)
    tmp_i = 0
    tmp_j = 0
    n = num_ent // batch_size if num_ent % batch_size == 0 else num_ent // batch_size + 1

    try:
        with trange(1, lef_num + rig_num + 1, **t_kwargs) as tbar:
            for i in tbar:
                if i <= lef_num:
                    triple[:, 0] = data_ptr.pair_tail_idx[i-1].ent
                    triple[:, 1] = data_ptr.pair_tail_idx[i-1].rel
                    for j in range(num_ent):
                        triple[j, 2] = j
                    
                    for m in range(0, n):
                        k = m * batch_size
                        if k + batch_size <= num_ent:
                            tmp_scores = function(np.array(triple[k:k+batch_size, :]))
                            for j in range(batch_size):
                                scores[k+j] = tmp_scores[j]
                        else:
                            tmp_scores = function(np.array(triple[k:num_ent, :]))
                            for j in range(num_ent-k):
                                scores[k+j] = tmp_scores[j]
                    
                    lef_i = data_ptr.pair_tail_idx[i-1].lef_id
                    rig_i = data_ptr.pair_tail_idx[i-1].rig_id
                    filter_lef_i, filter_rig_i = find_target_id(train_data.pair_tail_idx, train_data.pair_lef_head,
                                                                train_data.pair_rig_head,
                                                                data_ptr.pair_tail_idx[i-1].ent,
                                                                data_ptr.pair_tail_idx[i-1].rel)
                    # printf('\n %d, %d, %d, %d\n', data_ptr.pair_tail_idx[i-1].ent,
                    #                             data_ptr.pair_tail_idx[i-1].rel,
                    #                             filter_rig_i,
                    #                             filter_lef_i)
                    cal_tail_rank_c(0, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data_ptr.data_head, scores, train_data)
                    tmp_i += rig_i - lef_i + 1
                else:
                    triple[:, 2] = data_ptr.pair_head_idx[i-lef_num-1].ent
                    triple[:, 1] = data_ptr.pair_head_idx[i-lef_num-1].rel
                    for j in range(num_ent):
                        triple[j, 0] = j
                    
                    for m in range(0, n):
                        k = m * batch_size
                        if k + batch_size <= num_ent:
                            tmp_scores = function(np.array(triple[k:k+batch_size, :]))
                            for j in range(batch_size):
                                scores[k+j] = tmp_scores[j]
                        else:
                            tmp_scores = function(np.array(triple[k:num_ent, :]))
                            for j in range(num_ent-k):
                                scores[k+j] = tmp_scores[j]
                    
                    lef_i = data_ptr.pair_head_idx[i-lef_num-1].lef_id
                    rig_i = data_ptr.pair_head_idx[i-lef_num-1].rig_id
                    filter_lef_i, filter_rig_i = find_target_id(train_data.pair_head_idx, train_data.pair_lef_tail,
                                                                train_data.pair_rig_tail,
                                                                data_ptr.pair_head_idx[i-lef_num-1].ent,
                                                                data_ptr.pair_head_idx[i-lef_num-1].rel)
                    cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data_ptr.data_tail, scores, train_data)
                    tmp_j += rig_i - lef_i + 1
    except KeyboardInterrupt:
        tbar.close()
        raise
    
    return (np.array(<float[:data_ptr.data_size]>rhs_ranks), np.array(<float[:data_ptr.data_size]>rhs_franks),
            np.array(<float[:data_ptr.data_size]>lhs_ranks), np.array(<float[:data_ptr.data_size]>lhs_franks))


def calculate_ranks_on_test_via_triple(function, data: DataSet, batch_size: int):
    cdef:
        int i, j, k, n, m, tmp, tmp_num
        int tmp_i, tmp_j, lef_i, rig_i
        int filter_lef_i, filter_rig_i
        DataStruct* data_ptr = data.getTestDataPtr()
        DataStruct* all_triples = data.getAllTriplesPtr()
        int lef_num = data_ptr.lef_pair_num
        int rig_num = data_ptr.rig_pair_num
        int num_ent = data.num_ent
        int num_rel = data.num_rel

        MemoryPool lmem = MemoryPool()
        int *_triple = <int*>lmem.alloc(3 * num_ent, sizeof(int))
        int[:, ::1] triple = <int[:num_ent, :3]>_triple
        float *_scores = <float*>lmem.alloc(num_ent, sizeof(float))
        float[::1] scores = <float[:num_ent]>_scores
        np.ndarray[float, ndim=1] tmp_scores

        float *rhs_ranks = <float*>lmem.alloc(data_ptr.data_size, sizeof(float))
        float *rhs_franks = <float*>lmem.alloc(data_ptr.data_size, sizeof(float))
        float *lhs_ranks = <float*>lmem.alloc(data_ptr.data_size, sizeof(float))
        float *lhs_franks = <float*>lmem.alloc(data_ptr.data_size, sizeof(float))
        
        int *targets = <int*>lmem.alloc(num_ent, sizeof(int))
        int *targets_tmp = <int*>lmem.alloc(num_ent, sizeof(int))
    
    memset(_triple, -1, 3 * num_ent * sizeof(int))
    memset(rhs_ranks, -1, data_ptr.data_size * sizeof(float))
    memset(rhs_franks, -1, data_ptr.data_size * sizeof(float))
    memset(lhs_ranks, -1, data_ptr.data_size * sizeof(float))
    memset(lhs_franks, -1, data_ptr.data_size * sizeof(float))

    t_kwargs = dict(desc="Test Evaluating:", unit="pair", ncols=80, initial=1, total=lef_num + rig_num)
    tmp_i = 0
    tmp_j = 0
    n = num_ent // batch_size if num_ent % batch_size == 0 else num_ent // batch_size + 1

    tmp_num = 0

    try:
        with trange(1, lef_num + rig_num + 1, **t_kwargs) as tbar:
            for i in tbar:
                if i <= lef_num:
                    triple[:, 0] = data_ptr.pair_tail_idx[i-1].ent
                    triple[:, 1] = data_ptr.pair_tail_idx[i-1].rel
                    for j in range(num_ent):
                        triple[j, 2] = j
                    
                    for m in range(0, n):
                        k = m * batch_size
                        if k + batch_size <= num_ent:
                            tmp_scores = function(np.array(triple[k:k+batch_size, :]))
                            for j in range(batch_size):
                                scores[k+j] = tmp_scores[j]
                        else:
                            tmp_scores = function(np.array(triple[k:num_ent, :]))
                            for j in range(num_ent-k):
                                scores[k+j] = tmp_scores[j]

                        tmp_num += 1
                        #printf("%d, %d %d\n", tmp_num, tmp_scores.shape[0], tmp_scores.shape[1])

                    
                    lef_i = data_ptr.pair_tail_idx[i-1].lef_id
                    rig_i = data_ptr.pair_tail_idx[i-1].rig_id
                    filter_lef_i, filter_rig_i = find_target_id(all_triples.pair_tail_idx, all_triples.pair_lef_head,
                                                                all_triples.pair_rig_head,
                                                                data_ptr.pair_tail_idx[i-1].ent,
                                                                data_ptr.pair_tail_idx[i-1].rel)
                    # printf("--1-----------cal_tail starting ---------------------\n")
                    # printf('%d, %d, %d, %d, %d, %d, %d\n', i-1, rig_i, lef_i, data_ptr.pair_tail_idx[i-1].ent,
                    #                             data_ptr.pair_tail_idx[i-1].rel,
                    #                             filter_rig_i,
                    #                             filter_lef_i)
                    cal_tail_rank_c(data_ptr.pair_tail_idx[i-1].ent + data_ptr.pair_tail_idx[i-1].rel, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data_ptr.data_head, scores, all_triples)
                    tmp_i += rig_i - lef_i + 1
                    #printf("--1---------------------------------------------------\n")
                else:
                    triple[:, 2] = data_ptr.pair_head_idx[i-lef_num-1].ent
                    triple[:, 1] = data_ptr.pair_head_idx[i-lef_num-1].rel
                    for j in range(num_ent):
                        triple[j, 0] = j
                    
                    for m in range(0, n):
                        k = m * batch_size
                        if k + batch_size <= num_ent:
                            tmp_scores = function(np.array(triple[k:k+batch_size, :]))
                            for j in range(batch_size):
                                scores[k+j] = tmp_scores[j]
                        else:
                            tmp_scores = function(np.array(triple[k:num_ent, :]))
                            for j in range(num_ent-k):
                                scores[k+j] = tmp_scores[j]
                        
                        tmp_num += 1
                        #printf("%d, %d %d\n", tmp_num, tmp_scores.shape[0], tmp_scores.shape[1])
                    
                    lef_i = data_ptr.pair_head_idx[i-lef_num-1].lef_id
                    rig_i = data_ptr.pair_head_idx[i-lef_num-1].rig_id
                    filter_lef_i, filter_rig_i = find_target_id(all_triples.pair_head_idx, all_triples.pair_lef_tail,
                                                                all_triples.pair_rig_tail,
                                                                data_ptr.pair_head_idx[i-lef_num-1].ent,
                                                                data_ptr.pair_head_idx[i-lef_num-1].rel)
                    cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data_ptr.data_tail, scores, all_triples)
                    tmp_j += rig_i - lef_i + 1
                    #printf("--2---------------------------------------------------\n")
    except KeyboardInterrupt:
        tbar.close()
        raise
    
    return (np.array(<float[:data_ptr.data_size]>rhs_ranks), np.array(<float[:data_ptr.data_size]>rhs_franks),
            np.array(<float[:data_ptr.data_size]>lhs_ranks), np.array(<float[:data_ptr.data_size]>lhs_franks))


cdef void cal_head_rank_c(int index, int num_ent, int *idx, float *ranks, float *franks, int* targets, int* tmp, int lef_i, int rig_i, int filter_lef_i, int filter_rig_i, Triple *ptr, float[::1] scores, DataStruct *filter_data) except *:
    # cdef int tmp_idx

    # if filter_rig_i - filter_lef_i + 1 <= 0:
        # for tmp_idx in range(rig_i - lef_i +1):
            # ranks[idx[0]+tmp_idx] = 0
            # franks[idx[0]+tmp_idx] = 0
        # printf("%d, %d, %d\n", filter_rig_i, filter_lef_i, filter_rig_i - filter_lef_i + 1)
        # return
    cdef:
        int start = idx[0]
        int i, j, k
        int num = rig_i - lef_i + 1

        MemoryPool mem = MemoryPool()
        float *rank = <float*>mem.alloc(num, sizeof(float))
        float *frank = <float*>mem.alloc(num, sizeof(float))
        int *predict_targets = <int*>mem.alloc(rig_i - lef_i + 1, sizeof(int))
        int *filter_targets = <int*>mem.alloc(filter_rig_i - filter_lef_i + 1, sizeof(int))
        int *_idx = <int*>mem.alloc(num_ent, sizeof(int))
        # DataStruct *filter_data = &all_triples if flags else &train_data
    
    memset(targets, 0, num_ent * sizeof(int))
    memset(tmp, 0, num_ent * sizeof(int))
    
    if filter_lef_i >= 0 and filter_rig_i >=0:
        for i in range(filter_lef_i, filter_rig_i + 1):
            filter_targets[i-filter_lef_i] = filter_data.data_tail[i].head
        for i in range(filter_rig_i - filter_lef_i + 1):
            targets[filter_targets[i]] = 1

    for i in range(lef_i, rig_i + 1):
        predict_targets[i - lef_i] = ptr[i].head
    
    for i in range(rig_i - lef_i + 1):
        targets[predict_targets[i]] = -1

    argsort(_idx, scores)
    for i in range(num_ent):
        tmp[i] = targets[_idx[i]]

    j = 0
    k = 0
    for i in range(num_ent):
        x = tmp[i]
        if x == 1 or x == -1:
            if x == -1:
                frank[k] = i - j + 1
                rank[k] = i + 1
                k += 1
            j += 1
        if j == filter_rig_i - filter_lef_i + 1 + num:
            break
    
    for i in range(num):
        ranks[start + i] = rank[i]
        franks[start + i] = frank[i]


cdef void cal_tail_rank_c(int index, int num_ent, int *idx, float *ranks, float *franks, int* targets, int* tmp, int lef_i, int rig_i, int filter_lef_i, int filter_rig_i, Triple *ptr, float[::1] scores, DataStruct *filter_data) except *:

    cdef:
        int start = idx[0]
        int i, j, k
        int num = rig_i - lef_i + 1

        MemoryPool mem = MemoryPool()
        float *rank = <float*>mem.alloc(num, sizeof(float))
        float *frank = <float*>mem.alloc(num, sizeof(float))
        int *predict_targets = <int*>mem.alloc(num, sizeof(int))
        int *filter_targets = <int*>mem.alloc(filter_rig_i - filter_lef_i + 1, sizeof(int))
        int *_idx = <int*>mem.alloc(num_ent, sizeof(int))
        # DataStruct *filter_data = &all_triples if flags else &train_data

    memset(targets, 0, num_ent * sizeof(int))
    memset(tmp, 0, num_ent * sizeof(int))
    
    #printf('--1-----------------\n')

    if filter_lef_i >= 0 and filter_rig_i >=0:
        for i in range(filter_lef_i, filter_rig_i + 1):
            filter_targets[i-filter_lef_i] = filter_data.data_head[i].tail

        for i in range(filter_rig_i - filter_lef_i + 1):
            targets[filter_targets[i]] = 1
    
    #printf('--2-----------------\n')
    for i in range(lef_i, rig_i + 1):
        predict_targets[i - lef_i] = ptr[i].tail
    
    #printf('--3-----------------\n')
    for i in range(rig_i - lef_i + 1):
        targets[predict_targets[i]] = -1
    
    #printf('--4-----------------\n')
    argsort(_idx, scores)
    #printf('--5-----------------\n')
    for i in range(num_ent):
        tmp[i] = targets[_idx[i]]

    #printf('--6-----------------\n')
    j = 0
    k = 0
    for i in range(num_ent):
        x = tmp[i]
        if x == 1 or x == -1:
            if x == -1:
                frank[k] = <float>(i - j) + 1.
                rank[k] = <float>i + 1.
                k += 1
            j += 1
        if j == filter_rig_i - filter_lef_i + 1 + num:
            break
    
    #printf('--1-----------------\n')
    for i in range(num):
        ranks[start + i] = rank[i]
        franks[start + i] = frank[i]
    
    #printf('-------------------\n')
    #del mem
    #printf('--cal_tail_rank_c--\n')


def calculate_ranks_on_test_via_pair(function, dataset: DataSet, batch_size: int):
    cdef:
        int i, j, n, m, k, v, tmp_i, tmp_j, lef_i, rig_i, filter_lef_i, filter_rig_i

        DataStruct* data = dataset.getTestDataPtr()
        DataStruct* all_triples = dataset.getAllTriplesPtr()

        int num_ent = dataset.num_ent
        int num_rel = dataset.num_rel

        MemoryPool lmem = MemoryPool()

        int lef_num = data.lef_pair_num
        int rig_num = data.rig_pair_num
        int data_size = data.data_size

        float *rhs_ranks = <float*>lmem.alloc(data_size, sizeof(float))
        float *rhs_franks = <float*>lmem.alloc(data_size, sizeof(float))
        float *lhs_ranks = <float*>lmem.alloc(data_size, sizeof(float))
        float *lhs_franks = <float*>lmem.alloc(data_size, sizeof(float))

        int *targets = <int*>lmem.alloc(num_ent, sizeof(int))
        int *targets_tmp = <int*>lmem.alloc(num_ent, sizeof(int))

        int *_pair = <int*>lmem.alloc(2 * batch_size, sizeof(int))
        int[:, ::1] batch_data = <int[:batch_size, :2]>_pair

        np.ndarray[float, ndim=2] scores
        float[::1] score_tmp = <float[:num_ent]>lmem.alloc(num_ent, sizeof(float))
    
    memset(rhs_ranks, -1, data_size * sizeof(float))
    memset(rhs_franks, -1, data_size * sizeof(float))
    memset(lhs_ranks, -1, data_size * sizeof(float))
    memset(lhs_franks, -1, data_size * sizeof(float))

    t_kwargs = dict(desc="Test Evaluating:", unit="pair", ncols=80, initial=1, total=lef_num + rig_num)

    n = lef_num - (lef_num % batch_size)
    m = rig_num - (rig_num % batch_size)
    tmp_i = 0
    tmp_j = 0

    try:
        with trange(1, lef_num + rig_num + 1, **t_kwargs) as tbar:
            for i in tbar:
                if i <= n:
                    j = (i-1) % batch_size
                    batch_data[j, 0] = data.pair_tail_idx[i-1].ent
                    batch_data[j, 1] = data.pair_tail_idx[i-1].rel
                    if j == batch_size - 1:
                        scores = function(np.array(batch_data))
                        for k in range(batch_size):
                            for v in range(num_ent):
                                score_tmp[v] = scores[k, v]
                            lef_i = data.pair_tail_idx[i-batch_size+k].lef_id
                            rig_i = data.pair_tail_idx[i-batch_size+k].rig_id
                            filter_lef_i, filter_rig_i = find_target_id(all_triples.pair_tail_idx, all_triples.pair_lef_head,
                                                                all_triples.pair_rig_head,
                                                                data.pair_tail_idx[i-batch_size+k].ent,
                                                                data.pair_tail_idx[i-batch_size+k].rel)
                            cal_tail_rank_c(i-batch_size+k, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_head, score_tmp, all_triples)
                            tmp_i += rig_i - lef_i + 1

                if n < i <= lef_num:
                    j = (i-1) % batch_size
                    batch_data[j, 0] = data.pair_tail_idx[i-1].ent
                    batch_data[j, 1] = data.pair_tail_idx[i-1].rel
                    if j == (lef_num - n) - 1:
                        scores = function(np.array(batch_data[:j+1, :]))
                        for k in range(lef_num - n):
                            for v in range(num_ent):
                                score_tmp[v] = scores[k, v]
                            lef_i = data.pair_tail_idx[i-(lef_num - n)+k].lef_id
                            rig_i = data.pair_tail_idx[i-(lef_num - n)+k].rig_id
                            filter_lef_i, filter_rig_i = find_target_id(all_triples.pair_tail_idx, all_triples.pair_lef_head,
                                                                all_triples.pair_rig_head,
                                                                data.pair_tail_idx[i-(lef_num - n)+k].ent,
                                                                data.pair_tail_idx[i-(lef_num - n)+k].rel)
                            cal_tail_rank_c(0, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_head, score_tmp, all_triples)
                            tmp_i += rig_i - lef_i + 1
                
                if lef_num < i <= m + lef_num:
                    j = (i-lef_num -1) % batch_size
                    batch_data[j, 0] = data.pair_head_idx[i-lef_num-1].ent
                    batch_data[j, 1] = data.pair_head_idx[i-lef_num-1].rel + num_rel
                    if j == batch_size - 1:
                        scores = function(np.array(batch_data))
                        for k in range(batch_size):
                            for v in range(num_ent):
                                score_tmp[v] = scores[k, v]
                            lef_i = data.pair_head_idx[i-lef_num-batch_size+k].lef_id
                            rig_i = data.pair_head_idx[i-lef_num-batch_size+k].rig_id
                            filter_lef_i, filter_rig_i = find_target_id(all_triples.pair_head_idx, all_triples.pair_lef_tail,
                                                                all_triples.pair_rig_tail,
                                                                data.pair_head_idx[i-lef_num-batch_size+k].ent,
                                                                data.pair_head_idx[i-lef_num-batch_size+k].rel)
                            cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_tail, score_tmp, all_triples)
                            tmp_j += rig_i - lef_i + 1
                
                if m + lef_num < i <= lef_num + rig_num:
                    j = (i-lef_num -1) % batch_size
                    batch_data[j, 0] = data.pair_head_idx[i-lef_num-1].ent
                    batch_data[j, 1] = data.pair_head_idx[i-lef_num-1].rel + num_rel
                    if j == (rig_num - m) - 1:
                        scores = function(np.array(batch_data[:j+1, :]))
                        for k in range(rig_num - m):
                            for v in range(num_ent):
                                score_tmp[v] = scores[k, v]
                            lef_i = data.pair_head_idx[i-lef_num-(rig_num - m)+k].lef_id
                            rig_i = data.pair_head_idx[i-lef_num-(rig_num - m)+k].rig_id
                            filter_lef_i, filter_rig_i = find_target_id(all_triples.pair_head_idx, all_triples.pair_lef_tail,
                                                                all_triples.pair_rig_tail,
                                                                data.pair_head_idx[i-lef_num-(rig_num - m)+k].ent,
                                                                data.pair_head_idx[i-lef_num-(rig_num - m)+k].rel)
                            cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_tail, score_tmp, all_triples)
                            tmp_j += rig_i - lef_i + 1
    
    except KeyboardInterrupt:
        tbar.close()
        raise
    
    return (np.array(<float[:data.data_size]>rhs_ranks), np.array(<float[:data.data_size]>rhs_franks),
            np.array(<float[:data.data_size]>lhs_ranks), np.array(<float[:data.data_size]>lhs_franks))

def calculate_ranks_on_valid_via_pair(function, dataset: DataSet, batch_size: int):
    cdef:
        int i, j, n, m, k, v, tmp_i, tmp_j, lef_i, rig_i, filter_lef_i, filter_rig_i

        DataStruct* data = dataset.getValidDataPtr()
        DataStruct* train_data = dataset.getTrainDataPtr()

        int num_ent = dataset.num_ent
        int num_rel = dataset.num_rel

        MemoryPool lmem = MemoryPool()
        int lef_num = data.lef_pair_num
        int rig_num = data.rig_pair_num
        int data_size = data.data_size

        float *rhs_ranks = <float*>lmem.alloc(data_size, sizeof(float))
        float *rhs_franks = <float*>lmem.alloc(data_size, sizeof(float))
        float *lhs_ranks = <float*>lmem.alloc(data_size, sizeof(float))
        float *lhs_franks = <float*>lmem.alloc(data_size, sizeof(float))

        int *targets = <int*>lmem.alloc(num_ent, sizeof(int))
        int *targets_tmp = <int*>lmem.alloc(num_ent, sizeof(int))

        int *_pair = <int*>lmem.alloc(2 * batch_size, sizeof(int))
        int[:, ::1] batch_data = <int[:batch_size, :2]>_pair

        np.ndarray[float, ndim=2] scores
        float[::1] score_tmp = <float[:num_ent]>lmem.alloc(num_ent, sizeof(float))
    
    memset(rhs_ranks, -1, data_size * sizeof(float))
    memset(rhs_franks, -1, data_size * sizeof(float))
    memset(lhs_ranks, -1, data_size * sizeof(float))
    memset(lhs_franks, -1, data_size * sizeof(float))

    t_kwargs = dict(desc="Valid Evaluating:", unit="pair", ncols=80, initial=1, total=lef_num + rig_num)

    n = lef_num - (lef_num % batch_size)
    m = rig_num - (rig_num % batch_size)
    tmp_i = 0
    tmp_j = 0

    try:
        with trange(1, lef_num + rig_num + 1, **t_kwargs) as tbar:
            for i in tbar:
                if i <= n:
                    j = (i-1) % batch_size
                    batch_data[j, 0] = data.pair_tail_idx[i-1].ent
                    batch_data[j, 1] = data.pair_tail_idx[i-1].rel
                    if j == batch_size - 1:
                        scores = function(np.array(batch_data))
                        for k in range(batch_size):
                            for v in range(num_ent):
                                score_tmp[v] = scores[k, v]
                            lef_i = data.pair_tail_idx[i-batch_size+k].lef_id
                            rig_i = data.pair_tail_idx[i-batch_size+k].rig_id
                            filter_lef_i, filter_rig_i = find_target_id(train_data.pair_tail_idx, train_data.pair_lef_head,
                                                                train_data.pair_rig_head,
                                                                data.pair_tail_idx[i-batch_size+k].ent,
                                                                data.pair_tail_idx[i-batch_size+k].rel)
                            cal_tail_rank_c(0, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_head, score_tmp, train_data)
                            tmp_i += rig_i - lef_i + 1

                if n < i <= lef_num:
                    j = (i-1) % batch_size
                    batch_data[j, 0] = data.pair_tail_idx[i-1].ent
                    batch_data[j, 1] = data.pair_tail_idx[i-1].rel
                    if j == (lef_num - n) - 1:
                        scores = function(np.array(batch_data[:j+1, :]))
                        for k in range(lef_num - n):
                            for v in range(num_ent):
                                score_tmp[v] = scores[k, v]
                            lef_i = data.pair_tail_idx[i-(lef_num - n)+k].lef_id
                            rig_i = data.pair_tail_idx[i-(lef_num - n)+k].rig_id
                            filter_lef_i, filter_rig_i = find_target_id(train_data.pair_tail_idx, train_data.pair_lef_head,
                                                                train_data.pair_rig_head,
                                                                data.pair_tail_idx[i-(lef_num - n)+k].ent,
                                                                data.pair_tail_idx[i-(lef_num - n)+k].rel)
                            cal_tail_rank_c(0, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_head, score_tmp, train_data)
                            tmp_i += rig_i - lef_i + 1
                
                if lef_num < i <= m + lef_num:
                    j = (i-lef_num -1) % batch_size
                    batch_data[j, 0] = data.pair_head_idx[i-lef_num-1].ent
                    batch_data[j, 1] = data.pair_head_idx[i-lef_num-1].rel + num_rel
                    if j == batch_size - 1:
                        scores = function(np.array(batch_data))
                        for k in range(batch_size):
                            for v in range(num_ent):
                                score_tmp[v] = scores[k, v]
                            lef_i = data.pair_head_idx[i-lef_num-batch_size+k].lef_id
                            rig_i = data.pair_head_idx[i-lef_num-batch_size+k].rig_id
                            filter_lef_i, filter_rig_i = find_target_id(train_data.pair_head_idx, train_data.pair_lef_tail,
                                                                train_data.pair_rig_tail,
                                                                data.pair_head_idx[i-lef_num-batch_size+k].ent,
                                                                data.pair_head_idx[i-lef_num-batch_size+k].rel)
                            cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_tail, score_tmp, train_data)
                            tmp_j += rig_i - lef_i + 1
                
                if m + lef_num < i <= lef_num + rig_num:
                    j = (i-lef_num -1) % batch_size
                    batch_data[j, 0] = data.pair_head_idx[i-lef_num-1].ent
                    batch_data[j, 1] = data.pair_head_idx[i-lef_num-1].rel + num_rel
                    if j == (rig_num - m) - 1:
                        scores = function(np.array(batch_data[:j+1, :]))
                        for k in range(rig_num - m):
                            for v in range(num_ent):
                                score_tmp[v] = scores[k, v]
                            lef_i = data.pair_head_idx[i-lef_num-(rig_num - m)+k].lef_id
                            rig_i = data.pair_head_idx[i-lef_num-(rig_num - m)+k].rig_id
                            filter_lef_i, filter_rig_i = find_target_id(train_data.pair_head_idx, train_data.pair_lef_tail,
                                                                train_data.pair_rig_tail,
                                                                data.pair_head_idx[i-lef_num-(rig_num - m)+k].ent,
                                                                data.pair_head_idx[i-lef_num-(rig_num - m)+k].rel)
                            cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_tail, score_tmp, train_data)
                            tmp_j += rig_i - lef_i + 1
    
    except KeyboardInterrupt:
        tbar.close()
        raise
    
    return (np.array(<float[:data.data_size]>rhs_ranks), np.array(<float[:data.data_size]>rhs_franks),
            np.array(<float[:data.data_size]>lhs_ranks), np.array(<float[:data.data_size]>lhs_franks))