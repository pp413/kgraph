import arrow
import numpy as np
import prettytable as pt
import torch
from tqdm import tqdm, trange
from .metrics import hits_at_n_score, mrr_score, mr_score


def calculate_rank(test_labels, pred_socres, filter_labels, num_ent=0):   
    def where(dlist, break_num=0):
        rank, frank = [], []
        num = 0
        for i, x in enumerate(dlist):
            if x == 1 or x == -1:
                if x == -1:
                    frank.append(i-num+1)
                    rank.append(i+1)
                num += 1
            if num == break_num:
                break
        return np.array(rank), np.array(frank)
    
    all_labels_ternary = np.zeros(num_ent)
    all_labels_ternary[filter_labels] = 1
    all_labels_ternary[test_labels] = -1
    
    idx = np.argsort(pred_socres)[::-1]
    
    for_break = len(filter_labels)
    
    labels_ord = all_labels_ternary[idx]
    return where(labels_ord, break_num=for_break)

def add_reverse(data, num_relations, concate=True):
    if concate:
        src_, rel, dst_ = data.transpose(1, 0)
        src = np.concatenate((src_, dst_), 0)
        rel = np.concatenate((rel, rel + num_relations), 0)
        dst = np.concatenate((dst_, src_), 0)
        
        data = np.stack((src, rel, dst)).transpose(1, 0)
    else:
        src_, rel, dst_ = data.transpose(1, 0)
        rel += num_relations
        data = np.stack((dst_, rel, src_)).transpose(1, 0)
    return data

def get_triplets_set(train_data):
    return set([(x[0], x[1], x[2]) for x in train_data])

def T(data):
    src, rel, dst = data.transpose(1, 0)
    return np.stack((dst, rel, src)).transpose(1, 0)

def pprint(meg, filename=None):
    
    if filename is not None:
        with open(filename, 'a') as f:
            f.write(meg+'\n')
    print(meg)

def cal_mr_mrr_hits(ranks, des='ranks'):
    mr = mr_score(ranks)
    mrr = mrr_score(ranks)
    hits = [hits_at_n_score(ranks, i) for i in [1, 3, 10]]
    return [des, mr, mrr, hits[0], hits[1], hits[2]]

def build_graph(data):
    data_set = {}
    pairs = set()
    for triplet in data:
        src, rel, dst = triplet
        if (src, rel) in pairs:
            data_set[(src, rel)].add(dst)
        else:
            pairs.add((src, rel))
            data_set[(src, rel)] = {dst}
    pair_tail = {key: np.array(list(v)) for key, v in data_set.items()}
    return pair_tail, np.array([[x[0], x[1]] for x in pairs])

def __cal(target_function, data, num_ent=0, batch_size=512, pair_filter=None,
          device='cpu', pred_tail=True):
    """ 
    paramaters:
    target_function: the function for models to get scores.
    data: the valid / test data
    num_ent: the number of total entities.
    batch_size: the each size of batch test data.
    pair_filter: the set of tail entities for each Pair (head, relation) in all data.
    """
    data = build_graph(data)
    num_batchs = num_ent // batch_size + 1
    ranks, franks = [], []
    
    tbar = tqdm(data[1])
    new_ents = np.arange(num_ent).reshape((-1, 1))
    for pair in tbar:
        scores = np.zeros(num_ent)
        
        new_pairs = np.tile(pair.reshape((1, -1)), (num_ent, 1))
        new_triplets = np.concatenate((new_pairs, new_ents), 1)
        if not pred_tail:
            new_triplets = T(new_triplets)
       
        for i in range(num_batchs):
            j = i * batch_size
            batch_data = new_triplets[j: j + batch_size, :]
            batch_data = torch.from_numpy(batch_data).to(device)
            with torch.no_grad():
                score = target_function(batch_data).squeeze_()
                if not isinstance(score, np.ndarray):
                    if score.is_cuda:
                        score = score.cpu().numpy()
                    else:
                        score = score.numpy()
                scores[j: j + batch_size] += score
        rank, frank = calculate_rank(data[0][(pair[0], pair[1])],
                                     scores, pair_filter[(pair[0], pair[1])],
                                     num_ent=num_ent)
        for i in rank:
            ranks.append(i)
        for i in frank:
            franks.append(i)
    
    del data
    return np.array(ranks), np.array(franks)
                
def original_data_cal(function, data, num_ent=0, num_rel=0, batch_size=512,
                        all_data=None, device='cpu'):
    
    all_data = np.concatenate((all_data['train'], all_data['valid'], all_data['test']))
    pair_filter = build_graph(all_data)[0]    
    tranks, tfranks = __cal(function, data, num_ent, batch_size, pair_filter, device)

    data = T(data)
    all_data = T(all_data)
    pair_filter = build_graph(all_data)[0]
    hranks, hfranks = __cal(function, data, num_ent, batch_size, pair_filter, device, False)

    return (tranks, tfranks), (hranks, hfranks)

def double_data_cal(function, data, num_ent=0, num_rel=0, batch_size=512,
                      all_data=None, device='cpu'):
    all_data = np.concatenate((all_data['train'], all_data['valid'], all_data['test']))
    all_data = add_reverse(all_data, num_rel)
    pair_filter = build_graph(all_data)[0]
    tranks, tfranks =  __cal(function, data, num_ent, batch_size, pair_filter, device)
    
    data = add_reverse(data, num_rel, concate=False)
    hranks, hfranks = __cal(function, data, num_ent, batch_size, pair_filter, device)
    
    return (tranks, tfranks), (hranks, hfranks)


class BaseEval():
    
    def __init__(self, num_ent=0, num_rel=0, data=None, **kwargs):
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.data = data
    
    def cal_rank(self, flags):
        if flags == 'original':
            return original_data_cal
        else:
            return double_data_cal
    
    def eval(self, function, test_data, batch_size=512, flags='original',
             filename='conf.txt', device='cpu'):
        """
        
        paramaters:
        function: the predict function of models.
        test_data: test/valid data.
        batch_size: the size of the each batch data.
        flags: using the original data or not, new only two type, 'original', 'double'
        filename: conf name.
        device: 'cpu' or 'gpu'.
        
        """
        (tranks, tfranks), (hranks, hfranks) = self.cal_rank(flags)(
            function, test_data, self.num_ent, self.num_rel, batch_size, self.data, device
        )
        now = arrow.now().format('\t YYYY-MM-DD HH:mm:ss')
        pprint('\t\t The results of calulating the ranks ' + now, filename)
        tb = pt.PrettyTable()
        tb.float_format = "2.3"
        tb.field_names = [' Evaluation ', 'MR', 'MRR (%)', 'Hits@1 (%)', 'Hits@3 (%)', 'Hits@10 (%)']
        tb.add_row(cal_mr_mrr_hits(hranks, des='PredHead'))
        tb.add_row(cal_mr_mrr_hits(tranks, des='PredTail'))
   
        ranks = np.concatenate((tranks, hranks), 0)
        franks = np.concatenate((tfranks, hfranks), 0)
        tb.add_row(cal_mr_mrr_hits(ranks, des='Average'))
        tb.add_row(['', '', '', '', '', ''])
        tb.add_row(cal_mr_mrr_hits(hfranks, des='PredHeadFilter'))
        tb.add_row(cal_mr_mrr_hits(tfranks, des='PredTailFilter'))
        tb.add_row(cal_mr_mrr_hits(franks, des='AverageFilter'))
        tb.float_format = "2.3"
        
        print(tb)
        
        tb = tb.get_string()
        with open(filename, 'a') as f:
            f.write(tb)
        pprint('\n', filename)


class TrainEval_By_Triplet(BaseEval):
    
    def __init__(self, num_ent=0, num_rel=0, data=None, negative_rate=1, batch_size=0):
        self.num_rel = int(num_rel)
        self.num_ent = int(num_ent)
        self.batch_size = batch_size
        self.negative_rate = negative_rate
        self.data = data

        self.global_triplets = None
    
    def negative_sample(self, pos_samples):
        size = len(pos_samples)
        num_to_generate = size * self.negative_rate
        neg_samples = np.tile(pos_samples, (self.negative_rate, 1))
        labels = np.ones(size * (self.negative_rate + 1),
                         dtype=np.float32) * (-1.)
        labels[: size] = 1
        
        values = np.random.randint(self.num_ent, size=num_to_generate)
        choices = np.random.uniform(size=num_to_generate)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]
        
        for i, p in enumerate(choices):
            while True:
                triplet = (neg_samples[i, 0], neg_samples[i, 1], neg_samples[i, 2])
                if triplet not in self.global_triplets:
                    break
                if p > 0.5:
                    neg_samples[i, 0] = np.random.choice(self.num_ent)
                else:
                    neg_samples[i, 2] = np.random.choice(self.num_ent)
        return np.concatenate((pos_samples, neg_samples)), labels

    def sample_iter(self, batch_size=0, negative_rate=1):
        if batch_size == 0:
            batch_size = self.batch_size
        if negative_rate != 1:
            self.negative_rate = negative_rate
        
        if self.global_triplets is None:
            self.global_triplets = get_triplets_set(self.data['train'])
        
        size = len(self.data['train'])
        num_batch = size // batch_size
        train_data = np.random.permutation(self.data['train'])
        
        for i in trange(num_batch):
            yield self.negative_sample(train_data[i * batch_size: (i + 1) * batch_size, :])
     

class TrainEval_By_Pair(BaseEval):
    pass


class TrainEval_For_Trans(BaseEval):
    
    def __init__(self, num_ent=0, num_rel=0, data=None, negative_rate=1, batch_size=0):
        self.num_rel = int(num_rel)
        self.num_ent = int(num_ent)
        self.batch_size = batch_size
        self.negative_rate = negative_rate
        self.data = data

        self.global_triplets = get_triplets_set(self.data['train'])
    
    def sample(self, mix=True):  
        def filt(samples, all_train_triplet, for_neg_head=True):
            for i, s in enumerate(samples):
                triplet = (s[0], s[1], s[2])
                while triplet in all_train_triplet:
                    if for_neg_head:
                        samples[i, 0] = np.random.choice(self.num_ent)
                        triplet = ((samples[i, 0], s[1], s[2]))
                    else:
                        samples[i, 2] = np.random.choice(self.num_ent)
                        triplet = (s[0], s[1], samples[i, 2])
                    
        
        global_triplets = self.global_triplets
        train_data = self.data['train']
        
        num_to_generate =len(train_data)
        values = np.random.randint(self.num_ent, size=num_to_generate)
        choices = np.random.uniform(size=num_to_generate)
        subj = choices > 0.5
        obj = choices <= 0.5
        
        pos_head_samples = train_data[subj, :]
        neg_head_samples = pos_head_samples.copy()
        neg_head_samples[:, 0] = values[subj]
        filt(neg_head_samples, global_triplets)
        
        pos_tail_samples = train_data[obj, :]
        neg_tail_samples = pos_tail_samples.copy()
        neg_tail_samples[:, 2] = values[obj]
        filt(neg_tail_samples, global_triplets, for_neg_head=False)
        
        if mix:
            pos_samples = np.concatenate((pos_head_samples, pos_tail_samples))
            neg_samples = np.concatenate((neg_head_samples, neg_tail_samples))
            ret_index = np.random.permutation(np.arange(num_to_generate))
            
            return pos_samples[ret_index], neg_samples[ret_index]
        else:
            return (pos_head_samples, neg_head_samples), (pos_tail_samples, neg_tail_samples)

