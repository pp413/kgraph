# cython: language_level = 3
# distutils: language=c++

cimport cython
from libc.string cimport memset
from libc.string cimport memcpy
from cpython cimport Py_buffer
from libc.stdlib cimport rand, srand
from libcpp.algorithm cimport sort
from cython cimport int, float, sizeof
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef class Memory:
    
    def __cinit__(self, Py_ssize_t number):
        
        self.data = <int*> PyMem_Malloc(number * sizeof(int))
        if not self.data:
            raise MemoryError
    
    def resize(self, Py_ssize_t new_number):
        
        mem = <int *> PyMem_Realloc(self.data, new_number * sizeof(int))
        if not mem:
            raise MemoryError
        self.data = mem
    
    def __dealloc__(self):
        PyMem_Free(self.data)


cdef class Data:
    
    def __cinit__(self, Py_ssize_t ncols):
        self.ncols = ncols
        self.view_flag = 0
    
    @cython.boundscheck(False)
    cdef add_row(self, int* new_row):
        cdef int i
        cdef int start_i = self.v.size()
        cdef int end_i = start_i + self.ncols
        if self.view_flag > 0:
            raise ValueError("can't add row while being viewed")
        
        self.v.resize(end_i)
        
        for i from 0 <= i < <int>self.ncols:
            self.v[start_i+i] = new_row[i]
    
    def __getbuffer__(self, Py_buffer *buffer, int flags):
        
        cdef Py_ssize_t itemsize = sizeof(self.v[0])
        
        self.shape[0] = <Py_ssize_t>(self.v.size() / self.ncols)
        self.shape[1] = self.ncols
        
        self.strides[1] = <Py_ssize_t>(<char *>&(self.v[1]) - <char *>&(self.v[0]))
        self.strides[0] = self.ncols * self.strides[1]
        
        buffer.buf = <char *>&(self.v[0])
        buffer.format = 'i'
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.v.size() * itemsize
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL
        
        self.view_flag += 1
    
    def __releasebuffer__(self, Py_buffer *buffer):
        self.view_flag -= 1


# ctypedef void* (*malloc_t)(size_t n)
# ctypedef void (*free_t)(void *p)


# cdef class PyMalloc:
    
#     def __cinit__(self):
#         self.malloc_func = PyMem_Malloc

# cdef class PyFree:
    
#     def __cinit__(self):
#         self.free_func = PyMem_Free

cdef class PyMalloc:
    cdef void _set(self, malloc_t malloc):
        self.malloc_func = malloc

cdef PyMalloc WrapMalloc(malloc_t malloc):
    cdef PyMalloc o = PyMalloc()
    o._set(malloc)
    return o

cdef class PyFree:
    cdef void _set(self, free_t free):
        self.free_func = free

cdef PyFree WrapFree(free_t free):
    cdef PyFree o = PyFree()
    o._set(free)
    return o

Default_Malloc = WrapMalloc(PyMem_Malloc)
Default_Free = WrapFree(PyMem_Free)

cdef class MemoryPool:

    def __cinit__(self):
        self.size = 0
        self.addresses = {}
        self.refs = []
        self.pymalloc = WrapMalloc(PyMem_Malloc)
        self.pyfree = WrapFree(PyMem_Free)
    
    def __dealloc__(self):
        cdef Py_ssize_t addr 
        if self.addresses is not None:
            for addr in self.addresses:
                if addr != 0:
                    self.pyfree.free_func(<void*>addr)
    
    cdef void* alloc(self, Py_ssize_t number, Py_ssize_t elem_size) except NULL:

        if number == 0:
            warnings.warn("Allocating zero bytes!")
        # printf(b'generating %d bytes\n', number * elem_size)
        cdef void* p = self.pymalloc.malloc_func(number * elem_size)
        # printf(b'generating %d bytes\n', number * elem_size)
        if p == NULL:
            raise MemoryError("Error assigning %d bytes" % (number * elem_size))
        # printf(b'generating %d bytes\n', number * elem_size)
        memset(p, 0, number * elem_size)
        self.addresses[<size_t>p] = number * elem_size
        self.size += number * elem_size
        return p
    
    cdef void* realloc(self, void* p, Py_ssize_t new_size) except NULL:
        if <size_t>p not in self.addresses:
            raise ValueError("Pointer %d not found in Pool %s" % (<size_t>p, self.addresses))
        cdef void* new_ptr = self.alloc(1, new_size)
        memcpy(new_ptr, p, self.addresses[<size_t>p])
        self.free(p)
        self.addresses[<size_t>new_ptr] = new_size
        return new_ptr
    
    cdef void free(self, void* p) except *:
        self.size -= self.addresses.pop(<size_t>p)
        self.pyfree.free_func(p)
    
    def own_pyref(self, object py_ref):
        self.refs.append(py_ref)

cdef void initializeData(DataStruct *ptr):
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

# cdef MemoryPool global_memory_pool

cdef void set_int_ptr(int **ptr, int num, int flags):
    '''
    flags in [0, 1], 1: every element of data is -1.
    '''
    cdef int *tmp_ptr = ptr[0]
    ptr[0] = <int*>global_memory_pool.alloc(num, sizeof(int))
    if tmp_ptr != NULL:
        global_memory_pool.free(tmp_ptr)
    if flags != 0:
        memset(ptr[0], -1, num * sizeof(int))

cdef void set_float_ptr(float **ptr, int num):
    cdef float *tmp_ptr = ptr[0]
    ptr[0] = <float*>global_memory_pool.alloc(num, sizeof(float))
    if tmp_ptr != NULL:
        global_memory_pool.free(tmp_ptr)

cdef void set_pair_ptr(Pair **ptr, int num):
    cdef Pair *tmp_ptr = ptr[0]
    cdef Pair *_ptr = <Pair*>global_memory_pool.alloc(num, sizeof(Pair))
    if tmp_ptr != NULL:
        global_memory_pool.free(tmp_ptr)
    cdef int i
    for i in range(num):
        _ptr[i].lef_id = -1
        _ptr[i].rig_id = -1
    ptr[0] = _ptr

cdef void set_triple_ptr(Triple **ptr, int num):
    cdef Triple *tmp_ptr = ptr[0]
    ptr[0] = <Triple*>global_memory_pool.alloc(num, sizeof(Triple))
    if tmp_ptr != NULL:
        global_memory_pool.free(tmp_ptr)

cdef void load_triple_from_numpy(Triple* ptr, long[:, ::1] data):
    cdef int i
    cdef int n  = data.shape[0]
    if ptr == NULL:
        printf('ptr is NULL!\n')
    for i in range(n):
        # printf('i: %d, data_i: %d', i, data[i, 0])
        ptr[i].head = <int>data[i, 0]
        ptr[i].rel = <int>data[i, 1]
        ptr[i].tail = <int>data[i, 2]

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


# cdef unsigned long long *next_random
# cdef MemoryPool global_random_memory_pool


cdef void setThreadNumberAndRandSeed(const int thread_number, const int seed):
    setRandSeed(seed)
    randReset(thread_number)

cdef void setRandSeed(int seed):
    global global_random_memory_pool
    global_random_memory_pool = MemoryPool()
    if seed > 1:
        srand(seed)

cdef void randReset(int thread_number):
    global next_random
    cdef int i
    next_random = <unsigned long long *> global_random_memory_pool.alloc(thread_number, sizeof(unsigned long long))
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

cdef int _compare(const_void *a, const_void *b):
    cdef float v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0: return -1
    if v >= 0: return 1
