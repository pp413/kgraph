# cython: language_level = 3
# distutils: language=c++
import numpy as np
cimport numpy as np

import torch

from cpython cimport array
from cython cimport boundscheck, wraparound, cdivision

from .memory cimport initializeData
from .memory cimport setRandMemory
from .memory cimport reset_random_on_id
from .memory cimport random_for_prob
from .memory cimport MemoryPool
from .memory cimport load_triple_from_numpy
from .corrupt cimport find_target_id
from .corrupt cimport corrupt_tail_c
from .corrupt cimport corrupt_head_c

# #######################################################################################
# load the data from data2id.txt
@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef int[:, ::1] loadTripleIdFile_c(char* path, MemoryPool tmp_memory_pool):
    cdef:
        int i, num, tmp
        int n
        FILE *fin

    fin = fopen(path, 'r')
    if not fin:
        exit(EXIT_FAILURE)
    
    tmp = fscanf(fin, '%d', &num)
    cdef int * ptr = <int *> tmp_memory_pool.alloc(3 * num, sizeof(int))
    
    for i in range(num):
        tmp = fscanf(fin, '%d%d%d', &ptr[3 * i], &ptr[3 * i + 1], &ptr[3 * i + 2])
    return <int[:num, :3]>ptr

cdef int getTotal_c(char* path):
    cdef:
        int num
        FILE *fin
    
    fin = fopen(path, 'r')
    if not fin:
        printf('The path of file is error!\n')
        exit(EXIT_FAILURE)
    fscanf(fin, '%d', &num)
    return num

# #########################################################################################
# cache_data tools
cdef (int*, int*, int*, int*, Pair*, Pair*, int*, int*, int*, int*, int, int) _generate_index(
    Triple *dataHead, Triple *dataTail, int num, int num_ent, MemoryPool tmp_memory_pool):

    cdef:
        int i, hr_j, rt_j, k, tmp_r
        int *lefHead_data
        int *rigHead_data
        int *lefTail_data
        int *rigTail_data
    
    lefHead_data = NULL
    rigHead_data = NULL
    lefTail_data = NULL
    rigTail_data = NULL

    set_int_ptr(&lefHead_data, num_ent, 0, tmp_memory_pool)
    set_int_ptr(&rigHead_data, num_ent, -1, tmp_memory_pool)
    set_int_ptr(&lefTail_data, num_ent, 0, tmp_memory_pool)
    set_int_ptr(&rigTail_data, num_ent, -1, tmp_memory_pool)
    
    for i in range(1, num):
        if dataHead[i].head != dataHead[i - 1].head:
            lefHead_data[dataHead[i].head] = i
            rigHead_data[dataHead[i - 1].head] = i - 1
        
        if dataTail[i].tail != dataTail[i - 1].tail:
            lefTail_data[dataTail[i].tail] = i
            rigTail_data[dataTail[i - 1].tail] = i - 1
    
    lefHead_data[dataHead[0].head] = 0
    rigHead_data[dataHead[num - 1].head] = num - 1
    lefTail_data[dataTail[0].tail] = 0
    rigTail_data[dataTail[num - 1].tail] = num - 1

    hr_j = 0
    rt_j = 0
    for i in range(num_ent):
        if lefHead_data[i] != -1 or rigHead_data[i] != -1:
            tmp_r = -1
            for k in range(lefHead_data[i], rigHead_data[i] + 1):
                if dataHead[k].rel != tmp_r:
                    tmp_r = dataHead[k].rel
                    hr_j += 1
        
        if lefTail_data[i] != -1 or rigTail_data[i] != -1:
            tmp_r = -1
            for k in range(lefTail_data[i], rigTail_data[i] + 1):
                if dataTail[k].rel != tmp_r:
                    tmp_r = dataTail[k].rel
                    rt_j += 1
    
    cdef:
        Pair *pair_tail_idx = <Pair*>tmp_memory_pool.alloc(1, sizeof(Pair))
        Pair *pair_head_idx = <Pair*>tmp_memory_pool.alloc(1, sizeof(Pair))
        
    set_pair_ptr(&pair_tail_idx, hr_j, tmp_memory_pool)
    set_pair_ptr(&pair_head_idx, rt_j, tmp_memory_pool)
    
    hr_j = 0
    rt_j = 0
    for i in range(num_ent):
        if rigHead_data[i] != -1:
            tmp_r = -1
            for k in range(lefHead_data[i], rigHead_data[i] + 1):
                
                if dataHead[k].rel != tmp_r:
                    pair_tail_idx[hr_j].ent = dataHead[k].head
                    pair_tail_idx[hr_j].rel = dataHead[k].rel
                    pair_tail_idx[hr_j].lef_id = k
                    if (hr_j - 1 >= 0) and (pair_tail_idx[hr_j - 1].rig_id == -1):
                        pair_tail_idx[hr_j - 1].rig_id = k - 1
                    tmp_r = dataHead[k].rel
                    hr_j += 1
            pair_tail_idx[hr_j - 1].rig_id = rigHead_data[i]

        if rigTail_data[i] != -1:
            tmp_r = -1
            for k in range(lefTail_data[i], rigTail_data[i] + 1):
                
                if dataTail[k].rel != tmp_r:
                    pair_head_idx[rt_j].ent = dataTail[k].tail
                    pair_head_idx[rt_j].rel = dataTail[k].rel
                    pair_head_idx[rt_j].lef_id = k
                    if (rt_j - 1 >= 0) and (pair_head_idx[rt_j - 1].rig_id == -1):
                        pair_head_idx[rt_j - 1].rig_id = k - 1
                    tmp_r = dataTail[k].rel
                    rt_j += 1
            pair_head_idx[rt_j - 1].rig_id = rigTail_data[i]
    
    cdef:
        int *pair_lef_head = <int*>tmp_memory_pool.alloc(1, sizeof(int))
        int *pair_rig_head = <int*>tmp_memory_pool.alloc(1, sizeof(int))
        int *pair_lef_tail = <int*>tmp_memory_pool.alloc(1, sizeof(int))
        int *pair_rig_tail = <int*>tmp_memory_pool.alloc(1, sizeof(int))
    set_int_ptr(&pair_lef_head, num_ent, -1, tmp_memory_pool)
    set_int_ptr(&pair_rig_head, num_ent, -1, tmp_memory_pool)
    set_int_ptr(&pair_lef_tail, num_ent, -1, tmp_memory_pool)
    set_int_ptr(&pair_rig_tail, num_ent, -1, tmp_memory_pool)

    for i in range(1, hr_j):
        if pair_tail_idx[i].ent != pair_tail_idx[i - 1].ent:
            pair_lef_head[pair_tail_idx[i].ent] = i
            pair_rig_head[pair_tail_idx[i - 1].ent] = i - 1
    
    for i in range(1, rt_j):
        if pair_head_idx[i].ent != pair_head_idx[i - 1].ent:
            pair_lef_tail[pair_head_idx[i].ent] = i
            pair_rig_tail[pair_head_idx[i - 1].ent] = i - 1
    
    pair_lef_head[pair_tail_idx[0].ent] = 0
    pair_rig_head[pair_tail_idx[hr_j - 1].ent] = hr_j - 1
    pair_lef_tail[pair_head_idx[0].ent] = 0
    pair_rig_tail[pair_head_idx[rt_j - 1].ent] = rt_j - 1

    return lefHead_data, rigHead_data, lefTail_data, rigTail_data, pair_head_idx, pair_tail_idx, pair_lef_head, pair_rig_head, pair_lef_tail, pair_rig_tail, hr_j, rt_j


cdef void putTrainInCache_c(DataStruct *data_ptr, int[:, ::1] data_array, int entityTotal, int relationTotal, MemoryPool tmp_memory_pool):
    cdef int num = data_array.shape[0]

    cdef:
        int i, j, n
        MemoryPool mem = MemoryPool()
        int *_headList = <int*>mem.alloc(entityTotal, sizeof(int))
        int *_tailList = <int*>mem.alloc(entityTotal, sizeof(int))
    memset(_headList, -1, sizeof(int) * entityTotal)
    memset(_tailList, -1, sizeof(int) * entityTotal)

    data_ptr.data_size = num

    set_triple_ptr(&(data_ptr.data), num, tmp_memory_pool)
    set_triple_ptr(&(data_ptr.data_head), num, tmp_memory_pool)
    set_triple_ptr(&(data_ptr.data_tail),num, tmp_memory_pool)

    load_triple_from_numpy(data_ptr.data, data_array)
    load_triple_from_numpy(data_ptr.data_head, data_array)
    load_triple_from_numpy(data_ptr.data_tail, data_array)

    quick_sort(data_ptr.data, num, cmp_head)
    quick_sort(data_ptr.data_head, num, cmp_head)
    quick_sort(data_ptr.data_tail, num, cmp_tail)

    data_ptr.lefHead, data_ptr.rigHead, data_ptr.lefTail, data_ptr.rigTail, data_ptr.pair_head_idx, data_ptr.pair_tail_idx, data_ptr.pair_lef_head, data_ptr.pair_rig_head, data_ptr.pair_lef_tail, data_ptr.pair_rig_tail, data_ptr.lef_pair_num, data_ptr.rig_pair_num = _generate_index(
        data_ptr.data_head, data_ptr.data_tail, num, entityTotal, tmp_memory_pool)

    set_int_ptr(&(data_ptr.freqEnt), entityTotal, 0, tmp_memory_pool)
    set_int_ptr(&(data_ptr.freqRel), relationTotal, 0, tmp_memory_pool)
    set_float_ptr(&(data_ptr.lef_mean), relationTotal, tmp_memory_pool)
    set_float_ptr(&(data_ptr.rig_mean), relationTotal, tmp_memory_pool)

    data_ptr.ent_total_data.rig_num = 1
    data_ptr.ent_total_data.lef_num = 1

    data_ptr.freqEnt[data_ptr.data[0].head] += 1
    data_ptr.freqEnt[data_ptr.data[0].tail] += 1
    data_ptr.freqRel[data_ptr.data[0].rel] += 1

    for i in range(1, num):
        data_ptr.freqEnt[data_ptr.data[i].head] += 1
        data_ptr.freqEnt[data_ptr.data[i].tail] += 1
        data_ptr.freqRel[data_ptr.data[i].rel] += 1

        if data_ptr.data_tail[i].tail != data_ptr.data_tail[i-1].tail:
            if _tailList[data_ptr.data_tail[i].tail] == -1:
                _tailList[data_ptr.data_tail[i].tail] = data_ptr.data_tail[i].tail
                data_ptr.ent_total_data.rig_num += 1
        
        if data_ptr.data_head[i].head != data_ptr.data_head[i-1].head:
            if _headList[data_ptr.data_head[i].head] == -1:
                _headList[data_ptr.data_head[i].head] = data_ptr.data_head[i].head
                data_ptr.ent_total_data.lef_num += 1
    
    data_ptr.headList = <int*>tmp_memory_pool.alloc(data_ptr.ent_total_data.lef_num, sizeof(int))
    data_ptr.tailList = <int*>tmp_memory_pool.alloc(data_ptr.ent_total_data.rig_num, sizeof(int))

    j = 0
    n = 0
    for i in range(entityTotal):
        if _headList[i] != -1:
            data_ptr.headList[j] = _headList[i]
            j += 1
        if _tailList[i] != -1:
            data_ptr.tailList[n] = _tailList[i]
            n += 1
    
    sort(data_ptr.headList, data_ptr.headList + data_ptr.ent_total_data.lef_num)
    sort(data_ptr.tailList, data_ptr.tailList + data_ptr.ent_total_data.rig_num)

    for i in range(entityTotal):
        for j in range(data_ptr.lefHead[i] + 1, data_ptr.rigHead[i]+1):
            if data_ptr.data_head[j].rel != data_ptr.data_head[j - 1].rel:
                data_ptr.lef_mean[data_ptr.data_head[j].rel] += 1

        if data_ptr.lefHead[i] <= data_ptr.rigHead[i]:
            data_ptr.lef_mean[data_ptr.data_head[data_ptr.lefHead[i]].rel] += 1
        
        for j in range(data_ptr.lefTail[i] + 1, data_ptr.rigTail[i]+1):
            if data_ptr.data_tail[j].rel != data_ptr.data_tail[j - 1].rel:
                data_ptr.rig_mean[data_ptr.data_tail[j].rel] += 1

        if data_ptr.lefTail[i] <= data_ptr.rigTail[i]:
            data_ptr.rig_mean[data_ptr.data_tail[data_ptr.lefTail[i]].rel] += 1
    
    for i in range(relationTotal):
        if data_ptr.lef_mean[i] > 0.:
            data_ptr.lef_mean[i] = data_ptr.freqRel[i] / data_ptr.lef_mean[i]
        if data_ptr.rig_mean[i] > 0.:
            data_ptr.rig_mean[i] = data_ptr.freqRel[i] / data_ptr.rig_mean[i]

cdef void _putTestInCache(DataStruct *_test_data, int[:, ::1] data, int entityTotal, int relationTotal, MemoryPool tmp_memory_pool):
    cdef int num = data.shape[0]
    _test_data.data_size = num

    set_triple_ptr(&_test_data.data, num, tmp_memory_pool)
    set_triple_ptr(&_test_data.data_head, num, tmp_memory_pool)
    set_triple_ptr(&_test_data.data_tail, num, tmp_memory_pool)

    load_triple_from_numpy(_test_data.data, data)
    load_triple_from_numpy(_test_data.data_head, data)
    load_triple_from_numpy(_test_data.data_tail, data)

    quick_sort(_test_data.data, num, cmp_head)
    quick_sort(_test_data.data_head, num, cmp_head)
    quick_sort(_test_data.data_tail, num, cmp_tail)

    _test_data.lefHead, _test_data.rigHead, _test_data.lefTail, _test_data.rigTail, _test_data.pair_head_idx, _test_data.pair_tail_idx, _test_data.pair_lef_head, _test_data.pair_rig_head, _test_data.pair_lef_tail, _test_data.pair_rig_tail, _test_data.lef_pair_num, _test_data.rig_pair_num = _generate_index(
        _test_data.data_head, _test_data.data_tail, num, entityTotal, tmp_memory_pool)

cdef void putValidInCache_c(DataStruct *valid_data, int[:, ::1] valid_data_array, int entityTotal, int relationTotal, MemoryPool tmp_memory_pool):
    _putTestInCache(valid_data, valid_data_array, entityTotal, relationTotal, tmp_memory_pool)

cdef void putTestInCache_c(DataStruct *test_data, int[:, ::1] test_data_array, int entityTotal, int relationTotal, MemoryPool tmp_memory_pool):
    _putTestInCache(test_data, test_data_array, entityTotal, relationTotal, tmp_memory_pool)

cdef void putAllInCache_c(DataStruct *all_triples, int[:, ::1] train_data_array, int[:, ::1] valid_data_array, int[:, ::1] test_data_array, int entityTotal, int relationTotal, MemoryPool tmp_memory_pool):

    cdef:
        int i, _
        Pair *_p
        int train_num = train_data_array.shape[0]
        int valid_num = valid_data_array.shape[0]
    
    all_triples.data_size = train_num + valid_num

    # putTrainInCache_c(train_data, train_data_array, entityTotal, relationTotal)
    # putValidInCache_c(valid_data, valid_data_array, entityTotal, relationTotal)
    # putTestInCache_c(test_data, test_data_array, entityTotal, relationTotal)

    set_triple_ptr(&(all_triples.data_head), train_num + valid_num, tmp_memory_pool)
    set_triple_ptr(&(all_triples.data_tail), train_num + valid_num, tmp_memory_pool)

    for i in range(train_num):
        all_triples.data_head[i].head = <int>train_data_array[i, 0]
        all_triples.data_head[i].rel = <int>train_data_array[i, 1]
        all_triples.data_head[i].tail = <int>train_data_array[i, 2]

        all_triples.data_tail[i].head = <int>train_data_array[i, 0]
        all_triples.data_tail[i].rel = <int>train_data_array[i, 1]
        all_triples.data_tail[i].tail = <int>train_data_array[i, 2]
    
    for i in range(valid_num):
        all_triples.data_head[i + train_num].head = <int>valid_data_array[i, 0]
        all_triples.data_head[i + train_num].rel = <int>valid_data_array[i, 1]
        all_triples.data_head[i + train_num].tail = <int>valid_data_array[i, 2]

        all_triples.data_tail[i + train_num].head = <int>valid_data_array[i, 0]
        all_triples.data_tail[i + train_num].rel = <int>valid_data_array[i, 1]
        all_triples.data_tail[i + train_num].tail = <int>valid_data_array[i, 2]
    
    quick_sort(all_triples.data_head, train_num + valid_num, cmp_head)
    quick_sort(all_triples.data_tail, train_num + valid_num, cmp_tail)
    
    all_triples.lefHead, all_triples.rigHead, all_triples.lefTail, all_triples.rigTail, all_triples.pair_head_idx, all_triples.pair_tail_idx, all_triples.pair_lef_head, all_triples.pair_rig_head, all_triples.pair_lef_tail, all_triples.pair_rig_tail, _, _ = _generate_index(
        all_triples.data_head, all_triples.data_tail, train_num + valid_num, entityTotal, tmp_memory_pool)


cdef void get_constrain(Constrain **ptr, DataStruct *data_ptr, int relationTotal, MemoryPool tmp_memory_pool):
    cdef:
        int i
        MemoryPool tmp_mem = MemoryPool()
        Triple *data_rel_1 = <Triple*>tmp_mem.alloc(data_ptr.data_size, sizeof(Triple))
        Triple *data_rel_2 = <Triple*>tmp_mem.alloc(data_ptr.data_size, sizeof(Triple))
        Constrain constrain

    constrain.left_id_of_heads_of_relation = <int*>tmp_memory_pool.alloc(relationTotal, sizeof(int))
    constrain.right_id_of_heads_of_relation = <int*>tmp_memory_pool.alloc(relationTotal, sizeof(int))
    constrain.left_id_of_tails_of_relation = <int*>tmp_memory_pool.alloc(relationTotal, sizeof(int))
    constrain.right_id_of_tails_of_relation = <int*>tmp_memory_pool.alloc(relationTotal, sizeof(int))

    memset(constrain.left_id_of_heads_of_relation, -1, relationTotal * sizeof(int))
    memset(constrain.right_id_of_heads_of_relation, -1, relationTotal * sizeof(int))
    memset(constrain.left_id_of_tails_of_relation, -1, relationTotal * sizeof(int))
    memset(constrain.right_id_of_tails_of_relation, -1, relationTotal * sizeof(int))

    memcpy(data_rel_1, data_ptr.data_head, data_ptr.data_size * sizeof(Triple))
    memcpy(data_rel_2, data_ptr.data_head, data_ptr.data_size * sizeof(Triple))
    quick_sort(data_rel_1, data_ptr.data_size, cmp_rel2)
    quick_sort(data_rel_2, data_ptr.data_size, cmp_rel3)

    for i in range(1, data_ptr.data_size):
        if data_rel_1[i].rel != data_rel_1[i - 1].rel:
            constrain.left_id_of_heads_of_relation[data_rel_1[i].rel] = data_rel_1[i].head
            constrain.right_id_of_heads_of_relation[data_rel_1[i - 1].rel] = data_rel_1[i - 1].head
            constrain.left_id_of_tails_of_relation[data_rel_2[i].rel] = data_rel_2[i].tail
            constrain.right_id_of_tails_of_relation[data_rel_2[i - 1].rel] = data_rel_2[i - 1].tail
    
    constrain.left_id_of_heads_of_relation[data_rel_1[0].rel] = data_rel_1[0].head
    constrain.right_id_of_heads_of_relation[data_rel_1[data_ptr.data_size - 1].rel] = data_rel_1[data_ptr.data_size - 1].head
    constrain.left_id_of_tails_of_relation[data_rel_2[0].rel] = data_rel_2[0].tail
    constrain.right_id_of_tails_of_relation[data_rel_2[data_ptr.data_size - 1].rel] = data_rel_2[data_ptr.data_size - 1].tail

    ptr[0] = &constrain

cdef void generate_per_triple(DataStruct *data_ptr, int num_ent, int num_rel, int[:, ::1] corrupts, int[::1] labels, int tId, int mode, int normal_or_cross, int bern_flag) except *:

    cdef:
        int i, h, r, t
        float prob, p
        int num_neg = corrupts.shape[0]

    h = data_ptr.data[tId].head
    r = data_ptr.data[tId].rel
    t = data_ptr.data[tId].tail

    if bern_flag == 1:
        p = (data_ptr.rig_mean[r] + data_ptr.lef_mean[r])
        prob = 1000. * data_ptr.rig_mean[r]
    else:
        p = 1.
        prob = 500.

    for i in range(num_neg):
        corrupts[i, 0] = h
        corrupts[i, 1] = r
        corrupts[i, 2] = t
        if mode == 0 and normal_or_cross == 0:
            if (random_for_prob(tId) % 1000) * p < prob:
                corrupts[i, 2] = corrupt_tail_c(data_ptr, tId, h, r, num_ent, 0)
                labels[i] = -1
            else:
                corrupts[i, 0] = corrupt_head_c(data_ptr, tId, t, r, num_ent, 0)
                labels[i] = 1
        else:
            if normal_or_cross == 1:
                mode = 0 - mode
            if mode == -1:
                corrupts[i, 0] = corrupt_head_c(data_ptr, tId, t, r, num_ent, 0)
                labels[i] = 1
            else:
                corrupts[i, 2] = corrupt_tail_c(data_ptr, tId, h, r, num_ent, 0)
                labels[i] = -1


cdef void generate_per_pair(DataStruct *data_ptr, int num_ent, int num_rel, int[::1] per_pair, float[::1] per_label, int tId, int corrupt_on_tail, float smooth_lambda):

    cdef:
        int i, lef_id, rig_id
        float y_label
    
    if smooth_lambda > 0. and num_ent > 0.:
        y_label = 1. - smooth_lambda + 1.0 / num_ent
        per_label[...] = 1.0 / num_ent
    else:
        y_label = 1.
        per_label[...] = 0.

    if corrupt_on_tail == 0:
        per_pair[0] = data_ptr.pair_tail_idx[tId].ent
        per_pair[1] = data_ptr.pair_tail_idx[tId].rel

        lef_id, rig_id = find_target_id(data_ptr.pair_tail_idx, data_ptr.pair_lef_head, data_ptr.pair_rig_head,
                                        per_pair[0], per_pair[1])

        for i in range(lef_id, rig_id + 1):
            per_label[data_ptr.data_head[i].tail] = y_label
    
    else:
        per_pair[0] = data_ptr.pair_head_idx[tId].ent
        per_pair[1] = data_ptr.pair_head_idx[tId].rel + num_rel

        lef_id, rig_id = find_target_id(data_ptr.pair_head_idx, data_ptr.pair_lef_tail, data_ptr.pair_rig_tail,
                                        per_pair[0], data_ptr.pair_head_idx[tId].rel)

        for i in range(lef_id, rig_id + 1):
                    per_label[data_ptr.data_tail[i].head] = y_label

cdef np.ndarray[int, ndim=2] getDataFromCache_c(DataStruct *ptr):
    cdef:
        # int[:, ::1] data = np.zeros((ptr.data_size, 3), dtype=np.int32)
        np.ndarray[int, ndim=2] data = np.zeros((ptr.data_size, 3), dtype=np.int32)
        int i
    
    for i in range(ptr.data_size):
        data[i, 0] = (ptr.data_head[i].head)
        data[i, 1] = (ptr.data_head[i].rel)
        data[i, 2] = (ptr.data_head[i].tail)
    
    return np.array(data, copy=False)

# ########################################################################################
# python
def loadTripleIdFile(path_file):
    cdef MemoryPool tmp_memory_pool
    tmp_memory_pool = MemoryPool()
    return loadTripleIdFile_c(<char*>path_file, tmp_memory_pool)

def getTotal(path_file):
    return getTotal_c(<char*>path_file)

def setGlobalPool():
    global tmp_memory_pool
    tmp_memory_pool = MemoryPool()


cdef char* generate_path_c(const unsigned char[:] path, const unsigned char[:] file_name, MemoryPool tmp_memory_pool):
    cdef:
        int i
        char *path_c
        int length_path = path.shape[0]
        int length_name = file_name.shape[0]
    
    if path[-1] != b'/':
        length_path += 1

    path_c = <char*>tmp_memory_pool.alloc(length_path + length_name + 1, sizeof(char))
    for i in range(length_path-1):
        path_c[i] = path[i]
    path_c[length_path-1] = b'/'
    
    for i in range(length_name):
        path_c[i + length_path] = file_name[i]

    path_c[length_path + length_name] = b'\0'
    return path_c


cdef class DataSet:
    
    cdef DataStruct * getTrainDataPtr(self):
        return &(self.train_data_ptr)
    
    cdef DataStruct * getValidDataPtr(self):
        return &(self.valid_data_ptr)
    
    cdef DataStruct * getTestDataPtr(self):
        return &(self.test_data_ptr)
    
    cdef DataStruct * getAllTriplesPtr(self):
        return &(self.all_triples_ptr)
    
    def __init__(self, num_ent: int=0, num_rel: int=0):
        self.tmp_memory_pool = MemoryPool()
        setRandMemory()
        reset_random_on_id()
        self.num_ent = num_ent
        self.num_rel = num_rel

        self.train_data_size = 0
        self.num_neg = 0
        self.element_type = 0
        self._smooth_lambda = 0.1

        self.mode = 0
        self.normal_or_cross = 0
        self.bern_flag = 0

        initializeData(&(self.train_data_ptr))
        initializeData(&(self.valid_data_ptr))
        initializeData(&(self.test_data_ptr))
        initializeData(&(self.all_triples_ptr))

    # def load(self, const unsigned char[:] root_path, int no_sort):
    def load(self, root_path: bytes, no_sort: int):
        cdef:
            char* entity2id_path
            char* relation2id_path
            char* train2id_path = generate_path_c(root_path, b"train2id.txt", self.tmp_memory_pool)
            int[:, ::1] train_data_array = loadTripleIdFile_c(train2id_path, self.tmp_memory_pool)
            char* valid2id_path = generate_path_c(root_path, b"valid2id.txt", self.tmp_memory_pool)
            int[:, ::1] valid_data_array = loadTripleIdFile_c(valid2id_path, self.tmp_memory_pool)
            char* test2id_path = generate_path_c(root_path, b"test2id.txt", self.tmp_memory_pool)
            int[:, ::1] test_data_array = loadTripleIdFile_c(test2id_path, self.tmp_memory_pool)
        
        if no_sort > 0:
            entity2id_path = generate_path_c(root_path, b"entity2id_no_sort.txt", self.tmp_memory_pool)
            relation2id_path = generate_path_c(root_path, b"relation2id_no_sort.txt", self.tmp_memory_pool)
        else:
            entity2id_path = generate_path_c(root_path, b"entity2id_on_sort.txt", self.tmp_memory_pool)
            relation2id_path = generate_path_c(root_path, b"relation2id_on_sort.txt", self.tmp_memory_pool)
        self.num_ent = getTotal_c(entity2id_path)
        self.num_rel = getTotal_c(relation2id_path)

        putTrainInCache_c(&(self.train_data_ptr), train_data_array, self.num_ent, self.num_rel, self.tmp_memory_pool)
        putValidInCache_c(&(self.valid_data_ptr), valid_data_array, self.num_ent, self.num_rel, self.tmp_memory_pool)
        putTestInCache_c(&(self.test_data_ptr), test_data_array, self.num_ent, self.num_rel, self.tmp_memory_pool)
        putAllInCache_c(&(self.all_triples_ptr), train_data_array, valid_data_array, test_data_array, self.num_ent, self.num_rel, self.tmp_memory_pool)

    def getTrain(self):
        # global train_data
        return getDataFromCache_c(&(self.train_data_ptr))
    
    def getValid(self):
        # global valid_data
        return getDataFromCache_c(&(self.valid_data_ptr))
    
    def getTest(self):
        # global test_data
        return getDataFromCache_c(&(self.test_data_ptr))
    
    def getAll(self):
        # global all_triples
        return getDataFromCache_c(&(self.all_triples_ptr))
    
    def initConstraint(self):
        global type_constrain
        # global all_triples
        get_constrain(&type_constrain, &(self.all_triples_ptr), self.num_rel, self.tmp_memory_pool)
    
    def resetAllInCache(self, int[:, ::1] train_data, int[:, ::1] valid_data, int[:, ::1] test_data):

        self.train = train_data
        self.valid = valid_data
        self.test = test_data
        self.update()
    
    def update(self):
        putAllInCache_c(&(self.all_triples_ptr),
                        getDataFromCache_c(&(self.train_data_ptr)),
                        getDataFromCache_c(&(self.valid_data_ptr)),
                        getDataFromCache_c(&(self.test_data_ptr)),
                        self.num_ent, self.num_rel, self.tmp_memory_pool)
    
    property trainArray:
        def __get__(self):
            return getDataFromCache_c(&(self.train_data_ptr))
        
        def __set__(self, int[:, ::1] value):
            putTrainInCache_c(&(self.train_data_ptr), value, self.num_ent, self.num_rel, self.tmp_memory_pool)
            self.__calculate_train_data_size()
    
    property validArray:
        def __get__(self):
            return getDataFromCache_c(&(self.valid_data_ptr))
        
        def __set__(self, int[:, ::1] value):
            putValidInCache_c(&(self.valid_data_ptr), value, self.num_ent, self.num_rel, self.tmp_memory_pool)
    
    property testArray:
        def __get__(self):
            return getDataFromCache_c(&(self.test_data_ptr))
        
        def __set__(self, int[:, ::1] value):
            putTestInCache_c(&(self.test_data_ptr), value, self.num_ent, self.num_rel, self.tmp_memory_pool)
    
    property train:
        def __get__(self):
            return self.trainArray
        
        def __set__(self, int[:, ::1] value):
            self.trainArray = value
    
    property valid:
        def __get__(self):
            return self.validArray
        
        def __set__(self, int[:, ::1] value):
            self.validArray = value

    property test:
        def __get__(self):
            return self.testArray

        def __set__(self, int[:, ::1] value):
            self.testArray = value
    
    def __calculate_train_data_size(self):
        if self.element_type == 1:
            self.train_data_size = self.train_data_ptr.lef_pair_num + self.train_data_ptr.rig_pair_num
        else:
            self.train_data_size = self.train_data_ptr.data_size
    
    def set_trainDataSetPair(self, smooth_lambda: float=0.0):
        '''
        smooth_lambda: float, the smooth lambda for the labels in sampling process, default 0.0.
        return the DataSet for DataLoader in pytorch.
        '''
        self._smooth_lambda = smooth_lambda
        self.element_type = 1
        self.__calculate_train_data_size()
    
    property smooth_lambda:
        def __get__(self):
            return self._smooth_lambda
        
        def __set__(self, value: float):
            self._smooth_lambda = value
            self.element_type = 1
            self.__calculate_train_data_size()

    def get_item_pair(self, index: int):
        cdef np.ndarray[int, ndim=1] index_per_pair = np.zeros(2, dtype=np.int32)
        cdef np.ndarray[float, ndim=1] index_per_label = np.zeros(self.num_ent, dtype=np.float32)

        if index < self.train_data_ptr.lef_pair_num:
            generate_per_pair(&(self.train_data_ptr), self.num_ent, self.num_rel, index_per_pair, index_per_label, index, 0, self._smooth_lambda)
        else:
            index = index - self.train_data_ptr.lef_pair_num
            generate_per_pair(&(self.train_data_ptr), self.num_ent, self.num_rel, index_per_pair, index_per_label, index, 1, self._smooth_lambda)
        return index_per_pair, index_per_label
    
    def set_trainDataSetTriple(self, num_neg:int, mode: str='all', bern_flag: bool=False):
        '''
        num_neg: int, the number of negative samples for each positive sample.
        mode: str, the sampling mode, choice from ['all', 'head', 'tail', 'head_tail', 'normal', 'cross'], default 'all'.
        bern_flag: bool, whether to use bernoulli sampling, default False.
        return the DataSet for DataLoader in pytorch.
        '''
        assert mode.lower() in ['all', 'head', 'tail', 'head_tail', 'normal', 'cross'], 'mode must be one of "all", "head", "tail", "head_tail"'
        modes = {'all': 0, 'head': -1, 'tail': 1, 'head_tail': 0, 'normal': 0, 'cross': 2}
        mode_i = modes[mode.lower()]
        normal_or_cross = 0
        if mode_i == 2:
            mode_i = 1
            normal_or_cross = 1
        
        self.num_neg = num_neg

        self.mode = mode_i
        self.normal_or_cross = normal_or_cross
        self.bern_flag = int(bern_flag)
        
        self.train_data_size = self.train_data_ptr.data_size

        self.element_type = 0
    
    def set_cross_sampling(self):
        self.mode = 1
        self.normal_or_cross = 1
    
    property cross_sampling:
        def __get__(self):
            return self.normal_or_cross == 1
        
        def __set__(self, value: bool):
            if value:
                self.set_cross_sample()
            else:
                self.mode = 0
                self.normal_or_cross = 0
            self.element_type = 0
            self.__calculate_train_data_size()
    
    property bernoulli_sampling:
        def __get__(self):
            return bool(self.bern_flag)
        
        def __set__(self, value: bool):
            self.bern_flag = int(value)
            self.element_type = 0
            self.__calculate_train_data_size()
    
    property num_neg:
        def __get__(self):
            return self.num_neg
        
        def __set__(self, value: int):
            self.num_neg = value

            self.element_type = 0
            self.__calculate_train_data_size()
    
    def get_item_triple(self, index: int):

        cdef np.ndarray[int, ndim=1] index_per_triple = np.zeros(3, dtype=np.int32)
        cdef np.ndarray[int, ndim=2] corrupts = np.zeros((self.num_neg, 3), dtype=np.int32)
        cdef np.ndarray[int, ndim=1] index_per_label = np.zeros(self.num_neg, dtype=np.int32)

        index_per_triple[0] = self.train_data_ptr.data[index].head
        index_per_triple[1] = self.train_data_ptr.data[index].rel
        index_per_triple[2] = self.train_data_ptr.data[index].tail

        generate_per_triple(&(self.train_data_ptr), self.num_ent, self.num_rel, corrupts, index_per_label, index, self.mode, self.normal_or_cross, self.bern_flag)

        return index_per_triple, corrupts, index_per_label
    
    def __getitem__(self, index: int):
        index = index % self.train_data_size

        if self.element_type == 0:
            return self.get_item_triple(index)
        else:
            return self.get_item_pair(index)
    
    def __len__(self):
        return self.train_data_size
