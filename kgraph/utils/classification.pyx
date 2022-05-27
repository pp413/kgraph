# cython: language_level = 3
# distutils: language = c++
import numpy as np
cimport numpy as np
from tqdm import tqdm
from libc.stdlib cimport malloc, free
from .memory cimport Triple, DataStruct
from .memory cimport _rand64, MemoryPool
from .memory cimport IndexedElement
from .memory cimport _compare, qsort

from .corrupt cimport corrupt_tail_c
from .corrupt cimport corrupt_head_c
from .read cimport test_data
cdef MemoryPool load_memory_pool

from libc.stdio cimport printf

cdef int* getNegTest(int[:, ::1] pos, int[:, ::1] neg, int num_ent):
    cdef:
        int i
        int total = test_data.data_size
        Triple *posTestList = test_data.data

        int * label = <int*> load_memory_pool.alloc(total, sizeof(int))
    
    for i in range(total):
        pos[i, 0] = posTestList[i].head
        pos[i, 2] = posTestList[i].tail
        pos[i, 1] = posTestList[i].rel
        neg[i, 0] = posTestList[i].head
        neg[i, 2] = posTestList[i].tail
        neg[i, 1] = posTestList[i].rel
        label[i] = 1

        if _rand64(0) % 1000 < 500:
            neg[i, 2] = corrupt_tail_c(0, neg[i, 0], neg[i, 1], num_ent)
        else:
            neg[i, 0] = corrupt_head_c(0, neg[i, 2], neg[i, 1], num_ent)
            label[i] = 0
    
    return label


cdef void arg_sort(int* index, float* score, int num) except *:
    cdef int i
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(num * sizeof(IndexedElement))
    cdef int* tmp = <int*> malloc(num * sizeof(int))
    
    for i in range(num):
        tmp[i] = index[i]
        order_struct[i].index = i
        order_struct[i].value = score[i]
        
    qsort(<void *> order_struct, num, sizeof(IndexedElement), _compare)
    
    for i in range(num):
        index[i] = tmp[order_struct[i].index]
        score[i] = order_struct[i].value
        
    free(order_struct)
    free(tmp)

def run_triple_classification(function, num_ent: int, batch_size: int=1, threshold: float=-1.0):

    global load_memory_pool
    load_memory_pool = MemoryPool()

    cdef:
        int num_batch
        int i, j, start, end
        float acc
        float total_current

        np.ndarray[float, ndim=1] res_pos, res_neg

        int size = test_data.data_size
        float total_all = <float>(2 * size)
        float total_true = <float>size
        float total_false = <float>size

        float* score = <float*> load_memory_pool.alloc(2 * size, sizeof(float))
        int* ans = <int*> load_memory_pool.alloc(2 * size, sizeof(int))

        int *negTestList_ptr = <int *> load_memory_pool.alloc(3 * size, sizeof(int))
        int *posTestList_ptr = <int *> load_memory_pool.alloc(3 * size, sizeof(int))

        int[:, ::1] neg = <int[:size, :3]>negTestList_ptr
        int[:, ::1] pos = <int[:size, :3]>posTestList_ptr

        int[::1] label = <int[:size]>getNegTest(pos, neg, num_ent)

    num_batch = size / batch_size
    num_batch = num_batch if num_batch * batch_size == size else num_batch + 1

    with tqdm(total=num_batch, ncols=80) as pbar:
        pbar.set_description("Classification:")

        for i in range(num_batch):
            start = i * batch_size
            end = (i + 1) * batch_size
            end = end if end < size else size

            res_pos = function(np.array(pos[start:end, :]), np.array(label[start:end]))
            for j in range(end - start):
                ans[2 * start + j] = 1
                score[2 * start + j] = res_pos[j]

            res_neg = function(np.array(neg[start:end, :]), np.array(label[start:end]))
            for j in range(end - start):
                ans[start + end + j] = 0
                score[start + end + j] = res_neg[j]
            
            pbar.update(1)
    
    arg_sort(ans, score, 2 * size)
    if threshold == -1.0:
        threshold, _ = get_best_threshold(score, ans, size)
    
    total_current = 0.0
    for i in range(2 * size):
        if score[i] > threshold:
            acc = (2 * total_current + total_false - i) / total_all
            break
        else:
            if ans[i] == 1:
                total_current += 1
    
    return acc, threshold


cdef (float, float) get_best_threshold(float* score, int* ans, int n):
    cdef:
        int i
        float threshold, res_current
        float total_current, res_mx
        float total_all = <float>(2 * n)
        float total_true = <float>n
        float total_false = <float>n
    total_current = 0.0
    res_mx = 0.0
    threshold = -1.0
    # arg_sort(ans, score, 2 * n)
    for i in range(2 * n):
        if ans[i] == 1:
            total_current += 1.0
        res_current = (2 * total_current + total_false - i - 1) / total_all
        if res_current > res_mx:
            res_mx = res_current
            threshold = score[i]
    
    return threshold, res_mx




