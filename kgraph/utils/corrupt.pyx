# cython: language_level = 3
# distutils: language = c++

cdef (int, int) find_target_id(Pair *ptr, int *pair_lef, int *pair_rig, int ent, int rel) nogil:
    cdef:
        int i, j, k
        int ent_id = <int>ent
    
    i = pair_lef[ent_id]
    j = pair_rig[ent_id]

    while i < j:
        k = i + ((j-i) >> 1)
        if rel == ptr[k].rel:
            i = k
            break
        if rel < ptr[k].rel:
            j = k
        else:
            i = k + 1
    if rel != ptr[i].rel:
        return -1, -1
    
    if ptr[i].rig_id - ptr[i].lef_id < 0:
        return 0, -1

    return ptr[i].lef_id, ptr[i].rig_id

cdef int corrupt_tail_c(int tId, int head, int rel, int entityTotal) nogil:
    cdef:
        int lef, rig, mid, ll, rr
        int tmp
    
    lef, rig = find_target_id(train_data.pair_tail_idx, train_data.pair_lef_head, train_data.pair_rig_head, head, rel)
    tmp = <int>rand_max(tId, <long>(entityTotal - (rig - lef + 1)))

    if tmp < train_data.data_head[lef].tail:
        return tmp
    if tmp + rig - lef + 1 > train_data.data_head[rig].tail:
        return tmp + rig - lef + 1

    ll = lef
    rr = rig
    while ll < rr:
        mid = ll + ((rr - ll) >> 1)
        if tmp == train_data.data_head[mid].tail:
            ll = mid
            break
        if tmp < train_data.data_head[mid].tail:
            rr = mid
        else:
            ll = mid + 1
    if tmp == train_data.data_head[ll].tail:
        return corrupt_tail_c(tId, head, rel, entityTotal)
    else:

        return tmp + ll - lef

cdef int corrupt_head_c(int tId, int tail, int rel, int entityTotal) nogil:
    cdef:
        int lef, rig, mid, ll, rr
        int tmp
    
    lef, rig = find_target_id(train_data.pair_head_idx, train_data.pair_lef_tail, train_data.pair_rig_tail, tail, rel)
    tmp = <int>rand_max(tId, <long>(entityTotal - (rig - lef + 1)))

    if tmp < train_data.data_tail[lef].head:

        return tmp
    if tmp + rig - lef + 1 > train_data.data_tail[rig].head:

        return tmp + rig - lef + 1

    ll = lef
    rr = rig
    while ll < rr:
        mid = ll + ((rr - ll) >> 1)
        if tmp == train_data.data_tail[mid].head:
            ll = mid
            break
        if tmp < train_data.data_tail[mid].head:
            rr = mid
        else:
            ll = mid + 1
    if tmp == train_data.data_tail[ll].head:
        return corrupt_head_c(tId, tail, rel, entityTotal)
    else:
        return tmp + ll - lef

cdef bint find(DataStruct *ptr, int head, int rel, int tail) nogil:
    cdef int l, r, mid
    l = 0
    r = ptr.data_size
    while l < r:
        mid = l + ((r - l) >> 1)
        if (head == ptr.data_head[mid].head) or (head == ptr.data_head[mid].head and rel == ptr.data_head[mid].rel) or(head == ptr.data_head[mid].head and rel == ptr.data_head[mid].rel and tail == ptr.data_head[mid].tail):
            l = mid
            break
        if (head < ptr.data_head[mid].head) or (head == ptr.data_head[mid].head and rel < ptr.data_head[mid].rel) or (head == ptr.data_head[mid].head and rel == ptr.data_head[mid].rel and tail < ptr.data_head[mid].tail):
            r = mid
        else:
            l = mid + 1
    
    if (head == ptr.data_head[l].head and rel == ptr.data_head[l].rel and tail == ptr.data_head[l].tail):
        return 1
    else:
        return 0


cdef int corrupt_head_with_constrain(int tId, DataStruct *ptr, Constrain *constrain, int head, int rel, int entityTotal) nogil:
    cdef int loop
    cdef int tail
    loop = 0
    while True:
        tail = <int>rand64(<long>constrain.left_id_of_tails_of_relation[rel], <long>(constrain.right_id_of_tails_of_relation[rel]+1))
        if not find(ptr, head, rel, tail):
            return tail
        else:
            loop += 1
            if loop > 1000:
                return corrupt_tail_c(tId, head, rel, entityTotal)

cdef int corrupt_tail_with_constrain(int tId, DataStruct *ptr, Constrain *constrain, int tail, int rel, int entityTotal) nogil:
    cdef int loop
    cdef int head
    loop = 0
    while True:
        head = <int>rand64(<long>constrain.left_id_of_heads_of_relation[rel], <long>(constrain.right_id_of_heads_of_relation[rel]+1))
        if not find(ptr, head, rel, tail):
            return head
        else:
            loop += 1
            if loop > 1000:
                return corrupt_head_c(tId, tail, rel, entityTotal)
