# cython: language_level = 3
# distutils: language=c++
import numpy as np
from cython cimport boundscheck, wraparound, cdivision

# #######################################################################################
# load the data from data2id.txt
@cdivision(True)
@boundscheck(False)
@wraparound(False)
cdef np.ndarray[np.int64_t, ndim=2] loadTripleIdFile_c(char* path):
    cdef:
        int i, num, tmp
        long n
        FILE *fin

    fin = fopen(path, 'r')
    if not fin:
        exit(EXIT_FAILURE)
    
    tmp = fscanf(fin, '%d', &num)
    printf("the total of triples is %d \n", num)
    cdef np.ndarray[np.int64_t, ndim=2] data = np.empty((num, 3), dtype=np.int64)
    cdef long[:, ::1] d = data
    
    for i in range(num):
        tmp = fscanf(fin, '%ld%ld%ld', &d[i, 0], &d[i, 1], &d[i, 2])
    return data

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
    Triple *dataHead, Triple *dataTail, int num, int num_ent):

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

    set_int_ptr(&lefHead_data, num_ent, 0)
    set_int_ptr(&rigHead_data, num_ent, -1)
    set_int_ptr(&lefTail_data, num_ent, 0)
    set_int_ptr(&rigTail_data, num_ent, -1)
    
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
        Pair *pair_tail_idx = <Pair*>global_mem.alloc(1, sizeof(Pair))
        Pair *pair_head_idx = <Pair*>global_mem.alloc(1, sizeof(Pair))
        
    set_pair_ptr(&pair_tail_idx, hr_j)
    set_pair_ptr(&pair_head_idx, rt_j)
    
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
        int *pair_lef_head = <int*>global_mem.alloc(1, sizeof(int))
        int *pair_rig_head = <int*>global_mem.alloc(1, sizeof(int))
        int *pair_lef_tail = <int*>global_mem.alloc(1, sizeof(int))
        int *pair_rig_tail = <int*>global_mem.alloc(1, sizeof(int))
    set_int_ptr(&pair_lef_head, num_ent, -1)
    set_int_ptr(&pair_rig_head, num_ent, -1)
    set_int_ptr(&pair_lef_tail, num_ent, -1)
    set_int_ptr(&pair_rig_tail, num_ent, -1)

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


cdef void putTrainInCache_c(long[:, ::1] data_array, int entityTotal, int relationTotal):
    global train_data
    cdef int num = data_array.shape[0]

    cdef:
        int i, j, n
        Pool mem = Pool()
        long *_headList = <long*>mem.alloc(entityTotal, sizeof(long))
        long *_tailList = <long*>mem.alloc(entityTotal, sizeof(long))
    memset(_headList, -1, sizeof(long) * entityTotal)
    memset(_tailList, -1, sizeof(long) * entityTotal)

    train_data.data_size = num

    set_triple_ptr(&train_data.data, num)
    set_triple_ptr(&train_data.data_head, num)
    set_triple_ptr(&train_data.data_tail ,num)

    load_triple_from_numpy(train_data.data, data_array)
    load_triple_from_numpy(train_data.data_head, data_array)
    load_triple_from_numpy(train_data.data_tail, data_array)

    quick_sort(train_data.data, num, cmp_head)
    quick_sort(train_data.data_head, num, cmp_head)
    quick_sort(train_data.data_tail, num, cmp_tail)

    train_data.lefHead, train_data.rigHead, train_data.lefTail, train_data.rigTail, train_data.pair_head_idx, train_data.pair_tail_idx, train_data.pair_lef_head, train_data.pair_rig_head, train_data.pair_lef_tail, train_data.pair_rig_tail, train_data.lef_pair_num, train_data.rig_pair_num = _generate_index(
        train_data.data_head, train_data.data_tail, num, entityTotal)

    set_int_ptr(&train_data.freqEnt, entityTotal, 0)
    set_int_ptr(&train_data.freqRel, relationTotal, 0)
    set_float_ptr(&train_data.lef_mean, relationTotal)
    set_float_ptr(&train_data.rig_mean, relationTotal)

    train_data.ent_total_data.rig_num = 1
    train_data.ent_total_data.lef_num = 1

    train_data.freqEnt[train_data.data[0].head] += 1
    train_data.freqEnt[train_data.data[0].tail] += 1
    train_data.freqRel[train_data.data[0].rel] += 1

    for i in range(1, num):
        train_data.freqEnt[train_data.data[i].head] += 1
        train_data.freqEnt[train_data.data[i].tail] += 1
        train_data.freqRel[train_data.data[i].rel] += 1

        if train_data.data_tail[i].tail != train_data.data_tail[i-1].tail:
            if _tailList[train_data.data_tail[i].tail] == -1:
                _tailList[train_data.data_tail[i].tail] = train_data.data_tail[i].tail
                train_data.ent_total_data.rig_num += 1
        
        if train_data.data_head[i].head != train_data.data_head[i-1].head:
            if _headList[train_data.data_head[i].head] == -1:
                _headList[train_data.data_head[i].head] = train_data.data_head[i].head
                train_data.ent_total_data.lef_num += 1
    
    train_data.headList = <long*>global_mem.alloc(train_data.ent_total_data.lef_num, sizeof(long))
    train_data.tailList = <long*>global_mem.alloc(train_data.ent_total_data.rig_num, sizeof(long))

    j = 0
    n = 0
    for i in range(entityTotal):
        if _headList[i] != -1:
            train_data.headList[j] = _headList[i]
            j += 1
        if _tailList[i] != -1:
            train_data.tailList[n] = _tailList[i]
            n += 1
    
    sort(train_data.headList, train_data.headList + train_data.ent_total_data.lef_num)
    sort(train_data.tailList, train_data.tailList + train_data.ent_total_data.rig_num)

    for i in range(entityTotal):
        for j in range(train_data.lefHead[i] + 1, train_data.rigHead[i]+1):
            if train_data.data_head[j].rel != train_data.data_head[j - 1].rel:
                train_data.lef_mean[train_data.data_head[j].rel] += 1

        if train_data.lefHead[i] <= train_data.rigHead[i]:
            train_data.lef_mean[train_data.data_head[train_data.lefHead[i]].rel] += 1
        
        for j in range(train_data.lefTail[i] + 1, train_data.rigTail[i]+1):
            if train_data.data_tail[j].rel != train_data.data_tail[j - 1].rel:
                train_data.rig_mean[train_data.data_tail[j].rel] += 1

        if train_data.lefTail[i] <= train_data.rigTail[i]:
            train_data.rig_mean[train_data.data_tail[train_data.lefTail[i]].rel] += 1
    
    for i in range(relationTotal):
        if train_data.lef_mean[i] > 0.:
            train_data.lef_mean[i] = train_data.freqRel[i] / train_data.lef_mean[i]
        if train_data.rig_mean[i] > 0.:
            train_data.rig_mean[i] = train_data.freqRel[i] / train_data.rig_mean[i]

cdef void _putTestInCache(Data *_test_data, long[:, ::1] data, int entityTotal, int relationTotal):
    cdef int num = data.shape[0]
    _test_data.data_size = num

    set_triple_ptr(&_test_data.data, num)
    set_triple_ptr(&_test_data.data_head, num)
    set_triple_ptr(&_test_data.data_tail, num)

    load_triple_from_numpy(_test_data.data, data)
    load_triple_from_numpy(_test_data.data_head, data)
    load_triple_from_numpy(_test_data.data_tail, data)

    quick_sort(_test_data.data, num, cmp_head)
    quick_sort(_test_data.data_head, num, cmp_head)
    quick_sort(_test_data.data_tail, num, cmp_tail)

    _test_data.lefHead, _test_data.rigHead, _test_data.lefTail, _test_data.rigTail, _test_data.pair_head_idx, _test_data.pair_tail_idx, _test_data.pair_lef_head, _test_data.pair_rig_head, _test_data.pair_lef_tail, _test_data.pair_rig_tail, _test_data.lef_pair_num, _test_data.rig_pair_num = _generate_index(
        _test_data.data_head, _test_data.data_tail, num, entityTotal)

cdef void putValidAndTestInCache_c(long[:, ::1] valid_data_array, long[:, ::1] test_data_array, int entityTotal, int relationTotal):

    global valid_data
    global test_data
    _putTestInCache(&valid_data, valid_data_array, entityTotal, relationTotal)
    _putTestInCache(&test_data, test_data_array, entityTotal, relationTotal)

cdef void putAllInCache_c(long[:, ::1] train_data_array, long[:, ::1] valid_data_array, long[:, ::1] test_data_array, int entityTotal, int relationTotal):
    global all_triples
    cdef:
        int i, _
        Pair *_p
        int train_num = train_data_array.shape[0]
        int valid_num = valid_data_array.shape[0]
    
    all_triples.data_size = train_num + valid_num

    putTrainInCache_c(train_data_array, entityTotal, relationTotal)
    putValidAndTestInCache_c(valid_data_array, test_data_array, entityTotal, relationTotal)

    set_triple_ptr(&all_triples.data_head, train_num + valid_num)
    set_triple_ptr(&all_triples.data_tail, train_num + valid_num)

    for i in range(train_num):
        all_triples.data_head[i].head = train_data_array[i, 0]
        all_triples.data_head[i].rel = train_data_array[i, 1]
        all_triples.data_head[i].tail = train_data_array[i, 2]

        all_triples.data_tail[i].head = train_data_array[i, 0]
        all_triples.data_tail[i].rel = train_data_array[i, 1]
        all_triples.data_tail[i].tail = train_data_array[i, 2]
    
    for i in range(valid_num):
        all_triples.data_head[i + train_num].head = valid_data_array[i, 0]
        all_triples.data_head[i + train_num].rel = valid_data_array[i, 1]
        all_triples.data_head[i + train_num].tail = valid_data_array[i, 2]

        all_triples.data_tail[i + train_num].head = valid_data_array[i, 0]
        all_triples.data_tail[i + train_num].rel = valid_data_array[i, 1]
        all_triples.data_tail[i + train_num].tail = valid_data_array[i, 2]
    
    quick_sort(all_triples.data_head, train_num + valid_num, cmp_head)
    quick_sort(all_triples.data_tail, train_num + valid_num, cmp_tail)
    
    all_triples.lefHead, all_triples.rigHead, all_triples.lefTail, all_triples.rigTail, all_triples.pair_head_idx, all_triples.pair_tail_idx, all_triples.pair_lef_head, all_triples.pair_rig_head, all_triples.pair_lef_tail, all_triples.pair_rig_tail, _, _ = _generate_index(
        all_triples.data_head, all_triples.data_tail, train_num + valid_num, entityTotal)


cdef void get_constrain(Constrain **ptr, Data *data_ptr, int relationTotal):
    global global_mem
    cdef:
        int i
        Pool tmp_mem = Pool()
        Triple *data_rel_1 = <Triple*>tmp_mem.alloc(data_ptr.data_size, sizeof(Triple))
        Triple *data_rel_2 = <Triple*>tmp_mem.alloc(data_ptr.data_size, sizeof(Triple))
        Constrain constrain

    constrain.left_id_of_heads_of_relation = <long*>global_mem.alloc(relationTotal, sizeof(long))
    constrain.right_id_of_heads_of_relation = <long*>global_mem.alloc(relationTotal, sizeof(long))
    constrain.left_id_of_tails_of_relation = <long*>global_mem.alloc(relationTotal, sizeof(long))
    constrain.right_id_of_tails_of_relation = <long*>global_mem.alloc(relationTotal, sizeof(long))

    memset(constrain.left_id_of_heads_of_relation, -1, relationTotal * sizeof(long))
    memset(constrain.right_id_of_heads_of_relation, -1, relationTotal * sizeof(long))
    memset(constrain.left_id_of_tails_of_relation, -1, relationTotal * sizeof(long))
    memset(constrain.right_id_of_tails_of_relation, -1, relationTotal * sizeof(long))

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
    
cdef np.ndarray[long, ndim=2] getDataFromCache_c(Data *ptr):
    cdef:
        long[:, ::1] data = np.zeros((ptr.data_size, 3), dtype=np.int64)
        int i
    
    for i in range(ptr.data_size):
        data[i, 0] = ptr.data_head[i].head
        data[i, 1] = ptr.data_head[i].rel
        data[i, 2] = ptr.data_head[i].tail
    
    return np.array(data, copy=False)

# ########################################################################################
# python
def loadTripleIdFile(path_file):
    return loadTripleIdFile_c(<char*>path_file)

def getTotal(path_file):
    return getTotal_c(<char*>path_file)

def setGlobalPool():
    global global_mem
    global_mem = Pool()

def initializeTrainData():
    global train_data  
    initializeData(&train_data)

def initializeTestData():
    global test_data
    initializeData(&test_data)

def initializeValidData():
    global valid_data
    initializeData(&valid_data)

def initializeAllData():
    global all_triples
    initializeData(&all_triples)

    initializeTrainData()
    initializeValidData()
    initializeTestData()

def putTrainInCache(long[:, ::1] train_data_array, int entityTotal, int relationTotal):
    putTrainInCache_c(train_data_array, entityTotal, relationTotal)

def putValidAndTestInCache(long[:, ::1] valid_data_array,
                           long[:, ::1] test_data_array,
                           int entityTotal, int relationTotal):
    putValidAndTestInCache_c(valid_data_array, test_data_array, entityTotal, relationTotal)

def putAllInCache(long[:, ::1] train_data_array,
                  long[:, ::1] valid_data_array,
                  long[:, ::1] test_data_array,
                  int entityTotal, int relationTotal):
    putAllInCache_c(train_data_array, valid_data_array, test_data_array, entityTotal, relationTotal)

def getTrainFromCache():
    return getDataFromCache_c(&train_data)

def getValidFromCache():
    return getDataFromCache_c(&valid_data)

def getTestFromCache():
    return getDataFromCache_c(&test_data)

def getAllFromCache():
    return getDataFromCache_c(&all_triples)

def loadTypeConstrain(int relationTotal):
    global type_constrain
    get_constrain(&type_constrain, &all_triples, relationTotal)

cdef char* generate_path_c(const unsigned char[:] path, const unsigned char[:] file_name):
    cdef:
        int i
        char *path_c
        int length_path = path.shape[0]
        int length_name = file_name.shape[0]
    
    if path[-1] != b'/':
        length_path += 1

    path_c = <char*>global_mem.alloc(length_path + length_name + 1, sizeof(char))
    for i in range(length_path-1):
        path_c[i] = path[i]
    path_c[length_path-1] = b'/'
    
    for i in range(length_name):
        path_c[i + length_path] = file_name[i]

    path_c[length_path + length_name] = b'\0'
    return path_c


cdef class DataSet:

    cdef readonly:
        int num_ent
        int num_rel

    def __init__(self, const unsigned char[:] root_path):
        global train_data
        global valid_data
        global test_data
        global all_triples
        global global_mem
        global_mem = Pool()
        cdef char* train2id_path = generate_path_c(root_path, b"train2id.txt")
        cdef char* valid2id_path = generate_path_c(root_path, b"valid2id.txt")
        cdef char* test2id_path = generate_path_c(root_path, b"test2id.txt")
        cdef char* entity2id_path = generate_path_c(root_path, b"entity2id.txt")
        cdef char* relation2id_path = generate_path_c(root_path, b"relation2id.txt")

        cdef:
            long[:, ::1] train_data_array = loadTripleIdFile_c(train2id_path)
            long[:, ::1] valid_data_array = loadTripleIdFile_c(valid2id_path)
            long[:, ::1] test_data_array = loadTripleIdFile_c(test2id_path)
        self.num_ent = getTotal_c(entity2id_path)
        self.num_rel = getTotal_c(relation2id_path)

        initializeData(&train_data)
        initializeData(&valid_data)
        initializeData(&test_data)
        initializeData(&all_triples)

        putAllInCache_c(train_data_array, valid_data_array, test_data_array, self.num_ent, self.num_rel)

    def getTrain(self):
        global train_data
        return getDataFromCache_c(&train_data)
    
    def getValid(self):
        global valid_data
        return getDataFromCache_c(&valid_data)
    
    def getTest(self):
        global test_data
        return getDataFromCache_c(&test_data)
    
    def getAll(self):
        global all_triples
        return getDataFromCache_c(&all_triples)
    
    def initConstraint(self):
        global type_constrain
        global all_triples
        get_constrain(&type_constrain, &all_triples, self.num_rel)
    
    def resetAllInCache(self, long[:, ::1] train_data, long[:, ::1] valid_data, long[:, ::1] test_data):
        putAllInCache_c(train_data, valid_data, test_data, self.num_ent, self.num_rel)
    
    property train:
        def __get__(self):
            return self.getTrain()
        
        def __set__(self, long[:, ::1] value):
            putTrainInCache_c(value, self.num_ent, self.num_rel)
