cdef void setThreadNumberAndRandSeed(const int thread_number, const int seed):
    setRandSeed(seed)
    randReset(thread_number)

cdef void setRandSeed(int seed):
    global rand_mem
    rand_mem = Pool()
    if seed > 1:
        srand(seed)

cdef void randReset(int thread_number):
    global next_random
    cdef int i
    next_random = <unsigned long long *> rand_mem.alloc(thread_number, sizeof(unsigned long long))
    for i in range(thread_number):
        next_random[i] = rand()

cdef unsigned long long _rand64(int tId) nogil:
    global next_random
    next_random[tId] = next_random[tId] * <unsigned long long>(25214903917) + 11
    return next_random[tId]

cdef long rand_max(const int tId, const long x) nogil:
    cdef long res = _rand64(tId) % x
    while res < 0:
        res += x
    return res

cdef long rand64(const long a, const long b) nogil:
    return (rand() % (b - a)) + a
