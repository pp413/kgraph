import arrow
import numpy as np
from tqdm import trange, tqdm
from abc import ABC, abstractmethod
import torch

from .metrics import hits_at_n_score, mrr_score, mr_score

def calculate_rank(test_labels, pred_socres, filter_labels, num_entities:int=0):   
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
    
    all_labels_ternary = np.zeros(num_entities)
    all_labels_ternary[filter_labels] = 1
    all_labels_ternary[test_labels] = -1
    
    idx = np.argsort(pred_socres)[::-1]
    
    for_break = len(filter_labels)
    
    labels_ord = all_labels_ternary[idx]
    return where(labels_ord, break_num=for_break)

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
    return data

# Get the graph triplets
def get_train_triplets_set(train_data):
    return set([(x[0], x[1], x[2]) for x in train_data])

def pprint(meg, filename=None):
    now = arrow.now().format('YYYY-MM-DD HH:mm:ss *==|@=====>  ')
    if filename is not None:
        with open(filename, 'a') as f:
            f.write(now + meg+'\n')
    print(meg)


def print_result(ranks, des='describe', filename='conf.txt'):
    
    mr = mr_score(ranks)
    mrr = mrr_score(ranks)
    hits = [hits_at_n_score(ranks, i) for i in [1, 3, 10]]
    
    meg = '{}: \t {:.2f}\t {:.2f} \t {:.2f} \t {:.2f} \t {:.2f}'.format(
        des, mr, mrr, hits[0], hits[1], hits[2]
    )
    pprint(meg, filename)


class BaseEval(ABC):
    __slots__ = ['num_ent', 'num_rel', 'train_data', 'negative_rate',
                 'test_data', 'valid_data', 'batch_size', 'global_triplets']

    def __init__(self, num_ent=0, num_rel=0, data=None, negative_rate=1, batch_size=0):
        super(BaseEval, self).__init__()

        self.num_rel = int(num_rel)
        self.num_ent = int(num_ent)
        self.batch_size = batch_size
        self.negative_rate = negative_rate
        self.train_data = data['train']
        self.valid_data = data['valid']
        self.test_data = data['test']

        self.global_triplets = None
    
    @staticmethod
    def generate_graph(data):
        data_set = {}
        pairs = set()
        for triplet in data:
            src, rel, dst = triplet
            if (src, rel) in pairs:
                data_set[(src, rel)].add(dst)
            else:
                pairs.add((src, rel))
                data_set[(src, rel)] = {dst}
        return data_set, np.array([[x[0], x[1]] for x in pairs])
    
    @abstractmethod
    def negative_sample(self, *args, **warg):
        pass
    
    @abstractmethod
    def sample_iter(self, *args, **warg):
        pass

    def calculate(self, target_function, data, batch_size=512, pred_tail=True, device='cpu'):
        all_data = np.concatenate((self.train_data, self.valid_data, self.test_data))
        if not pred_tail:
            src, rel, dst = data.transpose(1, 0)
            data = np.stack((dst, rel, src)).transpose(1, 0)
            
            src, rel, dst = all_data.transpose(1, 0)
            all_data = np.stack((dst, rel, src)).transpose(1, 0)
            
        data = self.generate_graph(data)
        pair_filter = self.generate_graph(all_data)[0]
        num_batchs = self.num_ent // batch_size + 1
        ranks, franks = [], []
        for pair in tqdm(data[1]):
            scores = np.zeros(self.num_ent)
            new_ents = np.arange(self.num_ent).reshape((-1, 1))
            if pred_tail:
                new_pairs = np.tile(pair.reshape((1, -1)), (self.num_ent, 1))
                new_triplets = np.concatenate((new_pairs, new_ents), 1)
            else:
                new_pairs = np.tile(np.array([[pair[1], pair[0]]]), (self.num_ent, 1))
                new_triplets = np.concatenate((new_ents, new_pairs), 1)
            for i in range(num_batchs):
                j = i * batch_size
                batch_data = new_triplets[j: j + batch_size, :]
                batch_data = torch.from_numpy(batch_data).to(device)
                with torch.no_grad():
                    score = target_function(batch_data).squeeze()
                    if not isinstance(score, np.ndarray):
                        if score.is_cuda:
                            score = score.cpu().numpy()
                        else:
                            score = score.numpy()
                    scores[j: j + batch_size] += score

            rank, frank = calculate_rank(np.array(list(data[0][(pair[0], pair[1])])),
                                         np.array(scores), np.array(list(pair_filter[(pair[0], pair[1])])),
                                         num_entities=self.num_ent)

            for i in rank:
                ranks.append(i)
            for i in frank:
                franks.append(i)
        del data
        return ranks, franks
    
    def evaluate(self, target_function, test_data=None, batch_size=512, filename='conf.txt', device=None):        
        if torch.cuda.is_available() and device is None:
            device = 'cuda'
        if (not torch.cuda.is_available()) and device is None:
            device = 'cpu'
        
        print('Calulate the ranks for predicting tails')
        tranks, tfranks = self.calculate(target_function, test_data, batch_size=batch_size, device=device)
        print('Calulate the ranks for predicting heads')
        hranks, hfranks = self.calculate(target_function,test_data, batch_size=batch_size,pred_tail=False, device=device)
        print()
        pprint('\t    MR \t MRR \t Hits@1\t Hits@3\t Hits@10', filename)
        print_result(tranks, des='Tail', filename=filename)
        print_result(tfranks, des='TFil', filename=filename)
        pprint('\t', filename)
        
        pprint('\t    MR \t MRR \t Hits@1\t Hits@3\t Hits@10', filename)
        print_result(hranks, des='Head', filename=filename)
        print_result(hfranks, des='HFil', filename=filename)
        pprint('\t', filename)
        
        ranks = tranks + hranks
        franks = tfranks + hfranks
        pprint('\t    MR \t MRR \t Hits@1\t Hits@3\t Hits@10', filename)
        print_result(ranks, des='Aver', filename=filename)
        print_result(franks, des='AFil', filename=filename)
        pprint('\t', filename)
        

class TrainEval_By_Triplet(BaseEval):
    '''triplets.
    
       Example:
       
            from kgraph.utils import Triplet
            poss = Triplet(num_entities, num_relations, train_data,
                           test_data, valid_data, batch_size=1000)
            
            model
            
            for batch_data, labels in poss.sample_iter(batch_size=2000, negative_rate=1):
                    pred = model(batch_data)
                    loss(pred, labels)
            
            poss.evaluate(model.predict, test_data)
       
    '''
    
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
        
        self.global_triplets = get_train_triplets_set(self.train_data)
        
        size = len(self.train_data)
        num_batch = size // batch_size
        train_data = np.random.permutation(self.train_data)
        
        for i in trange(num_batch):
            yield self.negative_sample(train_data[i * batch_size: (i + 1) * batch_size, :])
    

class TrainEval_By_Pair(BaseEval):
    pass


class TrainEval_For_Trans(BaseEval):
    
    def __init__(self, num_ent=0, num_rel=0, data=None, negative_rate=1, batch_size=0):
        # super(TrainEval_For_Trans, self).__init__()
        self.num_rel = num_rel
        self.num_ent = num_ent
        self.batch_size = batch_size
        self.negative_rate = negative_rate
        self.train_data = data['train']
        self.valid_data = data['valid']
        self.test_data = data['test']

        self.global_triplets = get_train_triplets_set(self.train_data)
    
    def negative_sample(self):
        pass
    
    def sample_iter(self):
        pass
    
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
        train_data = self.train_data
        
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
