import numpy as np
import multiprocessing as mp


# Add the reverse data to original data.
def add_reverse(data, num_relations, concate=True):
    if concate:
        src_, rel, dst_ = data.transpose(1, 0)
        src = np.concatenate((src_, dst_), 0)
        rel = np.concatenate((rel, rel + num_relations), 0)
        dst = np.concatenate((dst_, src_), 0)
        
        data = np.stack((src, rel, dst)).transpose(1, 0)
    else:
        data[:, 1] += num_relations
    return np.random.permutation(data)

# Get the graph triplets
def get_train_triplets_set(train_data):
    return set([(x[0], x[1], x[2]) for x in train_data])

# Generate the corrupted sample.
def _corrupt_sample(triplet, global_triplets, num_entities):
    
    neg_prob = np.random.random_sample()
    
    while True:
        entity = np.random.choice(num_entities)
        if neg_prob < 0.5:
            neg_triplet = (triplet[0], triplet[1], entity)
        else:
            neg_triplet = (entity, triplet[1], triplet[2])
        
        if neg_triplet not in global_triplets:
            return np.asarray([neg_triplet[0], neg_triplet[1], neg_triplet[2]])


def generate_corrupted_sample(train_data, num_entities, rate):
    global_triplets = get_train_triplets_set(train_data)
    
    def corrupt_batch_sample(in_q, out_q):
        while True:
            raw_batch = in_q.get()
            if raw_batch is None:
                return
            else:
                pos_batch_data = raw_batch
                neg_batch_data = np.zeros((pos_batch_data.shape[0]*rate, 3))
        
                batch_i = 0
                for _ in range(rate):
                    for pos_triplet in pos_batch_data:
                        neg_batch_data[batch_i] += _corrupt_sample(pos_triplet, global_triplets, num_entities)
                        batch_i += 1
                out_q.put((pos_batch_data, neg_batch_data))
    
    return corrupt_batch_sample

class TranDataSample():
    def __init__(self, train_data, num_entities, rate):
        self.global_triplets = get_train_triplets_set(train_data)
        self.num_entities = num_entities
        self.rate = rate
    
    def corrupt_batch_sample(self, in_q, out_q):
        while True:
            raw_batch = in_q.get()
            if raw_batch is None:
                return
            else:
                pos_batch_data = raw_batch
                neg_batch_data = np.zeros((pos_batch_data.shape[0]*self.rate, 3), dtype=int)
        
                batch_i = 0
                for _ in range(self.rate):
                    for pos_triplet in pos_batch_data:
                        neg_batch_data[batch_i] += _corrupt_sample(pos_triplet, self.global_triplets, self.num_entities)
                        batch_i += 1
                out_q.put((pos_batch_data, neg_batch_data))



# Generate the corrupted sample    
def corrupt_sample(kernel, train_data, batch_size, rate, n_generator=5, with_label=False):
    in_q = mp.Queue()
    out_q = mp.Queue()
    
    for _ in range(n_generator):
        mp.Process(target=kernel, kwargs={'in_q': in_q, 'out_q': out_q}).start()
    
    batch_num = (train_data.shape[0] + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch_train_data = train_data[i*batch_size: (i+1)*batch_size]
        in_q.put(batch_train_data)
    
    for _ in range(n_generator):
        in_q.put(None)

    for i in range(batch_num):
        batch_datas = out_q.get()
        
        if not with_label:
            yield batch_datas
        else:
            
            batch_pos_data, batch_neg_data = batch_datas
            batch_size_ = batch_pos_data.shape[0]
            batch_label = np.zeros((batch_size_*(1+rate), 1), dtype=int)
            batch_label[:batch_size_] = 1
            batch_datas = np.concatenate((batch_pos_data, batch_neg_data), 0)
            batch_data_with_label = np.concatenate((batch_datas, batch_label), 1)
            yield batch_data_with_label

#
def sample_with_neg_iter(train_data, num_entities, num_relations, rate=1, add_rev=True):
    """Generate an iterator for sampling with negative samples
    
    Parameters:
        train_data: the train set.
        num_entities: the number of entities
        num_relations: the number of relations
        rate: the generating negative samples / the pos sample, default: 1
        add_rev: add the reverse triplets, default: True, if True, add the reverse
                 triplets to train set.
        
        return the iterator of var epoch training process.
               (batch_size, n_generator): batch_size: the size of batch data, default: 512;
                                          n_generator: the number of the Processes (workers), default: 6.
    
    Examples
    ------------------------------
    >>> from kgraph.datasets import load_fb15k237 
    >>> data, num_ent, num_rel = load_fb15k237()
    >>> x = data['train']
    >>> var_epoch_iter = sample_with_neg_iter(x, num_ent, num_rel)
    >>> for batch_data in var_epoch_iter(512):
    >>> ...  batch_pos_data, batch_neg_data = batch_data
    
    Notes: bath_pos_data and batch_neg_data are two independent numpy.array, and 
           rate = batch_neg_data.shape[0] // batch_pos_data.shape[0]
    """
    train_data = add_reverse(train_data, num_relations) if add_rev else train_data
    length = train_data.shape[0]
    
    train_data_sample = TranDataSample(train_data, num_entities, rate)
    
    def sample_iter(batch_size=512, n_generator=6):
        data_idx = np.random.permutation(length)
        return corrupt_sample(train_data_sample.corrupt_batch_sample,
                              train_data[data_idx], batch_size, rate,
                              n_generator, with_label=False)
    
    return sample_iter

def sample_with_neg_and_label_iter(train_data, num_entities, num_relations, rate=1, add_rev=True):
    """Generate an iterator for sampling with negative samples and labels
    
    Parameters:
        train_data: the train set.
        num_entities: the number of entities
        num_relations: the number of relations
        rate: the generating negative samples / the pos sample, default: 1
        add_rev: add the reverse triplets, default: True, if True, add the reverse
                 triplets to train set.
        
        return the iterator of var epoch training process.
               (batch_size, n_generator): batch_size: the size of batch data, default: 512;
                                          n_generator: the number of the Processes (workers)
                                                       default: 6.
    
    Examples
    ------------------------------
    >>> from kgraph.datasets import load_fb15k237 
    >>> data, num_ent, num_rel = load_fb15k237()
    >>> x = data['train']
    >>> var_epoch_iter = sample_with_neg_and_label_iter(x, num_ent, num_rel)
    >>> for batch_data in var_epoch_iter(512):
    >>> ...  batch_train_data = batch_data[:, : 4]
    >>> ...  batch_labels = batch_data[:, 4]
    
    Notes: batch_data is a numpy.array, and batch_data.shape[1] == 4
    """
    train_data = add_reverse(train_data, num_relations) if add_rev else train_data
    length = train_data.shape[0]
    
    train_data_sample = TranDataSample(train_data, num_entities, rate)
    
    def sample_iter(batch_size=512, n_generator=5):
        data_idx = np.random.permutation(length)
        return corrupt_sample(train_data_sample.corrupt_batch_sample,
                              train_data[data_idx], batch_size, rate,
                              n_generator, with_label=True)
    
    return sample_iter


#
def sample_with_label_iter(train_data, num_entities, rate=1, add_rev=True, filted=True):
    """Generate an iterator for sampling with labels
    
    Parameters:
        train_data: the train set.
        num_entities: the number of entities
        rate: the generating negative samples / the pos sample, default: 1
        add_rev: add the reverse triplets, default: True, if True, add the reverse
                 triplets to train set.
        filted: filter out the duplicate triplets, default: True.
        
        return the iterator of var epoch training process.
               batch_size: the size of batch data, default: 512.
    
    Examples
    ------------------------------
    >>> from kgraph.datasets import load_fb15k237 
    >>> data, num_ent, num_rel = load_fb15k237()
    >>> x = data['train']
    >>> var_epoch_iter = sample_with_label_iter(x, num_ent)
    >>> for batch_data in var_epoch_iter(512):
    >>> ...  batch_train_data, batch_labels = batch_data
    
    Notes: batch_train_data is a numpy.array; batch_labels is a list, and each element also is a list.
    """
    train_data = add_reverse(train_data, num_entities) if add_rev else train_data
    
    orig_pair = train_data[:, :2]
    labels_ = train_data[:, 2] if not filted else []
    data_ = []
    
    if not filted:
        data_ = orig_pair
    else:
        pairs = set()
        graph = dict()
        
        for triplet in train_data:
            pair = (triplet[0], triplet[1])
            if pair in pairs:
                graph[pair].append(triplet[2])
            else:
                pairs.add(pair)
                graph[pair] = [triplet[2]]
        for pair in list(pairs):
            data_.append([pair[0], pair[1]])
            i_label = sorted(graph[pair])
            labels_.append(i_label)
    data_ = np.asarray(data_)
    
    def sample_iter(batch_size=512):
        length = data_.shape[0]
        data_idx = np.random.permutation(length)
        batch_num = (length + batch_size - 1) // batch_size
        
        data = data_[data_idx]
        labels = [labels_[j] for j in data_idx]
        
        for i in range(batch_num):
            yield data[i*batch_size: (i+1)*batch_size], labels[i*batch_size: (i+1)*batch_size]
    
    return sample_iter

#   
def get_test_with_label(test_data, num_relations, concate=True):
    """Generate the test or valid set on test process.
    
    Parameters:
        test_data: the evaluating set, teh valid set or the test set.
        num_entities: the number of relations.
        concate: whether concate the reverse triplets or not in generte the evaluation data, default: True.
        
    Return:
        if concate is True:
            return the evaluation data with reverse triplets. (triplets:numpy.array, labels:list)
        if concate is False:
            return the evaluation data without reverse triplets.[(triplets:numpy.array, labels:list),(reverse_triplets:numpy.array, labels:list)]
            notes: the triplets are for (s, r) -> o; and the reverse_triplets are for (o, r) -> s; and each element in labels also is a label list for each triplet.
    
    """
    def _reverse(test_data, num_relations, concate=True):
        if concate:
            return (add_reverse(test_data, num_relations), )
        else:
            return (test_data, add_reverse(test_data, num_relations, concate))
    
    def _generate_test_graph(data):
        data_ = []
        labels = []
        pairs = set()
        graph = dict()
        for triplet in data:
            pair = (triplet[0], triplet[1])
            if pair in pairs:
                graph[pair].append(triplet[2])
            else:
                pairs.add(pair)
                graph[pair] = [triplet[2]]
        for pair in pairs:
            data_.append([pair[0], pair[1]])
            i_label = sorted(graph[pair])
            labels.append(i_label)
        
        return (np.asarray(data_), labels)
        
    _data = _reverse(test_data, num_relations, concate=concate)
    datas = [_generate_test_graph(x) for x in _data]
    
    return datas[0] if concate else datas
