import numpy as np
# from tqdm import trange

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


def generate_corrupted_sample(train_data, num_entities):
    global_triplets = get_train_triplets_set(train_data)
    
    def corrupt_batch_sample(neg_batch_data, pos_batch_data):
        for i, pos_triplet in enumerate(pos_batch_data):
            neg_batch_data[i] += _corrupt_sample(pos_triplet, global_triplets, num_entities)
    
    return corrupt_batch_sample

# Generate the corrupted sample
def corrupt_sample(kernel, rate=1):
    
    def sample(train_data, num_entities):
        corrupt_data = []
        for _ in range(rate):
            neg_data = np.zeros_like(train_data, dtype=np.float32)
            chunks = (neg_data, train_data)

            map(kernel, *chunks)
            corrupt_data.append(neg_data)
        corrupt_data = np.concatenate(corrupt_data, 0)
        return corrupt_data
    return sample

#
def sample_with_neg_and_label(train_data, num_entities, rate=1, add_rev=True):
    train_data = add_reverse(train_data, num_entities) if add_rev else train_data
    length = train_data.shape[0]
    
    labels = np.zeros((length * (1 + rate), 1))
    labels[: length] = 1
    length *= (1 + rate)
    
    func = corrupt_sample(
            generate_corrupted_sample(train_data, num_entities),
            rate=rate
        )
    
    def sample_iter(epoch=1, var_p=5):
        data_idx = np.random.permutation(length)
        for i in range(epoch):
            if (i + 1) % var_p == 0:
                data_idx = np.random.permutation(length)
            neg_data = func(train_data, num_entities)
            data = np.concatenate((train_data, neg_data), 0)
            yield data[data_idx], labels[data_idx]
    
    return sample_iter

#
def sample_with_label(train_data, num_entities, rate=1, add_rev=True, filted=True):
    train_data = add_reverse(train_data, num_entities) if add_rev else train_data
    
    orig_pair = train_data[:, :2]
    labels = train_data[:, 2] if not filted else []
    data = []
    
    if not filted:
        data = orig_pair
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
            data.append([pair[0], pair[1]])
            i_label = sorted(graph[pair])
            labels.append(i_label)
    data = np.asarray(data)
    
    def sample_iter(epoch=1, var_p=5):
        length = data.shape[0]
        data_idx = np.random.permutation(length)
        for i in range(epoch):
            if (i + 1) % var_p == 0:
                data_idx = np.random.permutation(length)
            yield data[data_idx], [labels[j] for j in data_idx]
    
    return sample_iter

#   
def get_test_with_label(test_data, num_relations, concate=True):
    
    def _reverse(test_data, num_relations, concate=True):
        if concate:
            return (add_reverse(test_data, num_relations), )
        else:
            return (add_reverse(test_data, num_relations, concate), test_data)
    
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
