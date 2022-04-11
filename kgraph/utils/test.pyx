# cython: language_level = 3
# distutils: language = c++
import numpy as np
from tqdm import tqdm, trange

cimport numpy as np
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from .mem cimport Pool
from libc.string cimport memset
from cython cimport long, int, float, sizeof
from libcpp.algorithm cimport sort

from .cache_data cimport global_mem
from .cache_data cimport Triple, Data

from .read cimport valid_data
from .read cimport test_data
from .read cimport all_triples, train_data
from .corrupt cimport find_target_id


np.import_array()

cdef void argsort(int* index, float[::1] array) except *:
    cdef:
        int tmp_i
        int length = array.shape[0]
        long[::1] index_array = np.asarray(array).argsort()
    
    for tmp_i in range(length):
        index[tmp_i] = <int>index_array[tmp_i] 


def calculate_ranks_on_valid_via_triple(function, num_ent: int, num_rel: int, batch_size: int):
    cdef:
        int i, j, k, n, m, tmp
        int tmp_i, tmp_j, lef_i, rig_i
        int filter_lef_i, filter_rig_i
        int lef_num = valid_data.lef_pair_num
        int rig_num = valid_data.rig_pair_num

        Pool lmem = Pool()
        long *_triple = <long*>lmem.alloc(3 * num_ent, sizeof(long))
        long[:, ::1] triple = <long[:num_ent, :3]>_triple
        float *_scores = <float*>lmem.alloc(num_ent, sizeof(float))
        float[::1] scores = <float[:num_ent]>_scores
        np.ndarray[float, ndim=1] tmp_scores

        float *rhs_ranks = <float*>lmem.alloc(valid_data.data_size, sizeof(float))
        float *rhs_franks = <float*>lmem.alloc(valid_data.data_size, sizeof(float))
        float *lhs_ranks = <float*>lmem.alloc(valid_data.data_size, sizeof(float))
        float *lhs_franks = <float*>lmem.alloc(valid_data.data_size, sizeof(float))
        
        long *targets = <long*>lmem.alloc(num_ent, sizeof(long))
        long *targets_tmp = <long*>lmem.alloc(num_ent, sizeof(long))
    
    memset(_triple, -1, 3 * num_ent * sizeof(long))
    memset(rhs_ranks, -1, valid_data.data_size * sizeof(float))
    memset(rhs_franks, -1, valid_data.data_size * sizeof(float))
    memset(lhs_ranks, -1, valid_data.data_size * sizeof(float))
    memset(lhs_franks, -1, valid_data.data_size * sizeof(float))

    t_kwargs = dict(desc="Valid Evaluating:", unit="pair", ncols=80, initial=1, total=lef_num + rig_num)
    tmp_i = 0
    tmp_j = 0
    n = num_ent // batch_size if num_ent % batch_size == 0 else num_ent // batch_size + 1

    try:
        with trange(1, lef_num + rig_num + 1, **t_kwargs) as tbar:
            for i in tbar:
                if i <= lef_num:
                    triple[:, 0] = valid_data.pair_tail_idx[i-1].ent
                    triple[:, 1] = valid_data.pair_tail_idx[i-1].rel
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
                    
                    lef_i = valid_data.pair_tail_idx[i-1].lef_id
                    rig_i = valid_data.pair_tail_idx[i-1].rig_id
                    filter_lef_i, filter_rig_i = find_target_id(train_data.pair_tail_idx, train_data.pair_lef_head,
                                                                train_data.pair_rig_head,
                                                                valid_data.pair_tail_idx[i-1].ent,
                                                                valid_data.pair_tail_idx[i-1].rel)
                    # printf('\n %d, %d, %d, %d\n', valid_data.pair_tail_idx[i-1].ent,
                    #                             valid_data.pair_tail_idx[i-1].rel,
                    #                             filter_rig_i,
                    #                             filter_lef_i)
                    cal_tail_rank_c(0, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, valid_data.data_head, scores, 0)
                    tmp_i += rig_i - lef_i + 1
                else:
                    triple[:, 2] = valid_data.pair_head_idx[i-lef_num-1].ent
                    triple[:, 1] = valid_data.pair_head_idx[i-lef_num-1].rel
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
                    
                    lef_i = valid_data.pair_head_idx[i-lef_num-1].lef_id
                    rig_i = valid_data.pair_head_idx[i-lef_num-1].rig_id
                    filter_lef_i, filter_rig_i = find_target_id(train_data.pair_head_idx, train_data.pair_lef_tail,
                                                                train_data.pair_rig_tail,
                                                                valid_data.pair_head_idx[i-lef_num-1].ent,
                                                                valid_data.pair_head_idx[i-lef_num-1].rel)
                    cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, valid_data.data_tail, scores, 0)
                    tmp_j += rig_i - lef_i + 1
    except KeyboardInterrupt:
        tbar.close()
        raise
    
    return (np.array(<float[:valid_data.data_size]>rhs_ranks), np.array(<float[:valid_data.data_size]>rhs_franks),
            np.array(<float[:valid_data.data_size]>lhs_ranks), np.array(<float[:valid_data.data_size]>lhs_franks))


def calculate_ranks_on_test_via_triple(function, num_ent: int, num_rel: int, batch_size: int):
    cdef:
        int i, j, k, n, m, tmp, tmp_num
        int tmp_i, tmp_j, lef_i, rig_i
        int filter_lef_i, filter_rig_i
        int lef_num = test_data.lef_pair_num
        int rig_num = test_data.rig_pair_num

        Pool lmem = Pool()
        long *_triple = <long*>lmem.alloc(3 * num_ent, sizeof(long))
        long[:, ::1] triple = <long[:num_ent, :3]>_triple
        float *_scores = <float*>lmem.alloc(num_ent, sizeof(float))
        float[::1] scores = <float[:num_ent]>_scores
        np.ndarray[float, ndim=1] tmp_scores

        float *rhs_ranks = <float*>lmem.alloc(test_data.data_size, sizeof(float))
        float *rhs_franks = <float*>lmem.alloc(test_data.data_size, sizeof(float))
        float *lhs_ranks = <float*>lmem.alloc(test_data.data_size, sizeof(float))
        float *lhs_franks = <float*>lmem.alloc(test_data.data_size, sizeof(float))
        
        long *targets = <long*>lmem.alloc(num_ent, sizeof(long))
        long *targets_tmp = <long*>lmem.alloc(num_ent, sizeof(long))
    
    memset(_triple, -1, 3 * num_ent * sizeof(long))
    memset(rhs_ranks, -1, test_data.data_size * sizeof(float))
    memset(rhs_franks, -1, test_data.data_size * sizeof(float))
    memset(lhs_ranks, -1, test_data.data_size * sizeof(float))
    memset(lhs_franks, -1, test_data.data_size * sizeof(float))

    t_kwargs = dict(desc="Test Evaluating:", unit="pair", ncols=80, initial=1, total=lef_num + rig_num)
    tmp_i = 0
    tmp_j = 0
    n = num_ent // batch_size if num_ent % batch_size == 0 else num_ent // batch_size + 1

    tmp_num = 0

    try:
        with trange(1, lef_num + rig_num + 1, **t_kwargs) as tbar:
            for i in tbar:
                if i <= lef_num:
                    triple[:, 0] = test_data.pair_tail_idx[i-1].ent
                    triple[:, 1] = test_data.pair_tail_idx[i-1].rel
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

                    
                    lef_i = test_data.pair_tail_idx[i-1].lef_id
                    rig_i = test_data.pair_tail_idx[i-1].rig_id
                    filter_lef_i, filter_rig_i = find_target_id(all_triples.pair_tail_idx, all_triples.pair_lef_head,
                                                                all_triples.pair_rig_head,
                                                                test_data.pair_tail_idx[i-1].ent,
                                                                test_data.pair_tail_idx[i-1].rel)
                    # printf("--1-----------cal_tail starting ---------------------\n")
                    # printf('%d, %d, %d, %d, %d, %d, %d\n', i-1, rig_i, lef_i, test_data.pair_tail_idx[i-1].ent,
                    #                             test_data.pair_tail_idx[i-1].rel,
                    #                             filter_rig_i,
                    #                             filter_lef_i)
                    cal_tail_rank_c(test_data.pair_tail_idx[i-1].ent + test_data.pair_tail_idx[i-1].rel, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, test_data.data_head, scores, 1)
                    tmp_i += rig_i - lef_i + 1
                    #printf("--1---------------------------------------------------\n")
                else:
                    triple[:, 2] = test_data.pair_head_idx[i-lef_num-1].ent
                    triple[:, 1] = test_data.pair_head_idx[i-lef_num-1].rel
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
                    
                    lef_i = test_data.pair_head_idx[i-lef_num-1].lef_id
                    rig_i = test_data.pair_head_idx[i-lef_num-1].rig_id
                    filter_lef_i, filter_rig_i = find_target_id(all_triples.pair_head_idx, all_triples.pair_lef_tail,
                                                                all_triples.pair_rig_tail,
                                                                test_data.pair_head_idx[i-lef_num-1].ent,
                                                                test_data.pair_head_idx[i-lef_num-1].rel)
                    cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, test_data.data_tail, scores, 1)
                    tmp_j += rig_i - lef_i + 1
                    #printf("--2---------------------------------------------------\n")
    except KeyboardInterrupt:
        tbar.close()
        raise
    
    return (np.array(<float[:test_data.data_size]>rhs_ranks), np.array(<float[:test_data.data_size]>rhs_franks),
            np.array(<float[:test_data.data_size]>lhs_ranks), np.array(<float[:test_data.data_size]>lhs_franks))


cdef void cal_head_rank_c(int index, int num_ent, int *idx, float *ranks, float *franks, long* targets, long* tmp, int lef_i, int rig_i, int filter_lef_i, int filter_rig_i, Triple *ptr, float[::1] scores, bint flags):
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

        Pool mem = Pool()
        float *rank = <float*>mem.alloc(num, sizeof(float))
        float *frank = <float*>mem.alloc(num, sizeof(float))
        long *predict_targets = <long*>mem.alloc(rig_i - lef_i + 1, sizeof(long))
        long *filter_targets = <long*>mem.alloc(filter_rig_i - filter_lef_i + 1, sizeof(long))
        int *_idx = <int*>mem.alloc(num_ent, sizeof(int))
        Data *filter_data = &all_triples if flags else &train_data
    
    memset(targets, 0, num_ent * sizeof(long))
    memset(tmp, 0, num_ent * sizeof(long))
    
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


cdef void cal_tail_rank_c(int index, int num_ent, int *idx, float *ranks, float *franks, long* targets, long* tmp, int lef_i, int rig_i, int filter_lef_i, int filter_rig_i, Triple *ptr, float[::1] scores, bint flags) except *:
    # cdef int tmp_idx, target_num

    #target_num = filter_rig_i - filter_lef_i + 1
    #if target_num <= 0:
        #target_num = rig_i - lef_i + 1
        #for tmp_idx in range(rig_i - lef_i +1):
            #ranks[idx[0]+tmp_idx] = 0
            #franks[idx[0]+tmp_idx] = 0
        #printf("%d, %d, %d, %d\n", index, filter_rig_i, filter_lef_i, target_num)
        #return

    cdef:
        int start = idx[0]
        int i, j, k
        int num = rig_i - lef_i + 1

        Pool mem = Pool()
        float *rank = <float*>mem.alloc(num, sizeof(float))
        float *frank = <float*>mem.alloc(num, sizeof(float))
        long *predict_targets = <long*>mem.alloc(num, sizeof(long))
        long *filter_targets = <long*>mem.alloc(filter_rig_i - filter_lef_i + 1, sizeof(long))
        int *_idx = <int*>mem.alloc(num_ent, sizeof(int))
        Data *filter_data = &all_triples if flags else &train_data

    memset(targets, 0, num_ent * sizeof(long))
    memset(tmp, 0, num_ent * sizeof(long))
    
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


def calculate_ranks_on_test_via_pair(function, num_ent: int, num_rel: int, batch_size: int):
    cdef:
        int i, j, n, m, k, v, tmp_i, tmp_j, lef_i, rig_i, filter_lef_i, filter_rig_i

        int lef_num = test_data.lef_pair_num
        int rig_num = test_data.rig_pair_num
        int data_size = test_data.data_size

        Data *data = &test_data
        Pool lmem = Pool()

        float *rhs_ranks = <float*>lmem.alloc(data_size, sizeof(float))
        float *rhs_franks = <float*>lmem.alloc(data_size, sizeof(float))
        float *lhs_ranks = <float*>lmem.alloc(data_size, sizeof(float))
        float *lhs_franks = <float*>lmem.alloc(data_size, sizeof(float))

        long *targets = <long*>lmem.alloc(num_ent, sizeof(long))
        long *targets_tmp = <long*>lmem.alloc(num_ent, sizeof(long))

        long *_pair = <long*>lmem.alloc(2 * batch_size, sizeof(long))
        long[:, ::1] batch_data = <long[:batch_size, :2]>_pair

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
                            cal_tail_rank_c(i-batch_size+k, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_head, score_tmp, 1)
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
                            cal_tail_rank_c(0, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_head, score_tmp, 1)
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
                            cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_tail, score_tmp, 1)
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
                            cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_tail, score_tmp, 1)
                            tmp_j += rig_i - lef_i + 1
    
    except KeyboardInterrupt:
        tbar.close()
        raise
    
    return (np.array(<float[:data.data_size]>rhs_ranks), np.array(<float[:data.data_size]>rhs_franks),
            np.array(<float[:data.data_size]>lhs_ranks), np.array(<float[:data.data_size]>lhs_franks))

def calculate_ranks_on_valid_via_pair(function, num_ent: int, num_rel: int, batch_size: int):
    cdef:
        int i, j, n, m, k, v, tmp_i, tmp_j, lef_i, rig_i, filter_lef_i, filter_rig_i

        int lef_num = valid_data.lef_pair_num
        int rig_num = valid_data.rig_pair_num
        int data_size = valid_data.data_size

        Data *data = &valid_data
        Pool lmem = Pool()

        float *rhs_ranks = <float*>lmem.alloc(data_size, sizeof(float))
        float *rhs_franks = <float*>lmem.alloc(data_size, sizeof(float))
        float *lhs_ranks = <float*>lmem.alloc(data_size, sizeof(float))
        float *lhs_franks = <float*>lmem.alloc(data_size, sizeof(float))

        long *targets = <long*>lmem.alloc(num_ent, sizeof(long))
        long *targets_tmp = <long*>lmem.alloc(num_ent, sizeof(long))

        long *_pair = <long*>lmem.alloc(2 * batch_size, sizeof(long))
        long[:, ::1] batch_data = <long[:batch_size, :2]>_pair

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
                            cal_tail_rank_c(0, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_head, score_tmp, 0)
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
                            cal_tail_rank_c(0, num_ent, &tmp_i, rhs_ranks, rhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_head, score_tmp, 0)
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
                            cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_tail, score_tmp, 0)
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
                            cal_head_rank_c(0, num_ent, &tmp_j, lhs_ranks, lhs_franks, targets, targets_tmp, lef_i, rig_i, filter_lef_i, filter_rig_i, data.data_tail, score_tmp, 0)
                            tmp_j += rig_i - lef_i + 1
    
    except KeyboardInterrupt:
        tbar.close()
        raise
    
    return (np.array(<float[:data.data_size]>rhs_ranks), np.array(<float[:data.data_size]>rhs_franks),
            np.array(<float[:data.data_size]>lhs_ranks), np.array(<float[:data.data_size]>lhs_franks))