import os, arrow
import numpy as np
import prettytable as pt
import torch
from tqdm import tqdm, trange
from .metrics import hits_at_n_score, mrr_score, mr_score

beam_size = 10000

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

def add_reverse(data, num_relations=None, concate=True):
    old_data = data.copy()
    if num_relations is None:
        num_relations = np.max(old_data[:, 1])
    if concate:
        src_, rel, dst_ = old_data.transpose(1, 0)
        src = np.concatenate((src_, dst_), 0)
        rel = np.concatenate((rel, rel + num_relations), 0)
        dst = np.concatenate((dst_, src_), 0)
        
        new_data = np.stack((src, rel, dst)).transpose(1, 0)
    else:
        src_, rel, dst_ = old_data.transpose(1, 0)
        rel += num_relations
        new_data = np.stack((dst_, rel, src_)).transpose(1, 0)
    return new_data

def get_triplets_set(train_data):
    return set([(x[0], x[1], x[2]) for x in train_data])

def get_all_triplets_set(data):
    train_set = get_triplets_set(data['train'])
    valid_set = get_triplets_set(data['valid'])
    test_set = get_triplets_set(data['test'])
    return train_set | valid_set | test_set

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

def __pair_calculate(target_function, data, num_ent=0, batch_size=512, pair_filter=None, device='cpu', pred_tail=True):
    global beam_size
    beam_size = beam_size if beam_size < num_ent else num_ent
    num_batchs = len(data) // batch_size + 1
    tbar = tqdm(range(num_batchs), ncols=100)
    ranks, franks = [], []
    for i in tbar:
        batch_data = data[i*batch_size: (i+1)*batch_size, :]
        batch_data = torch.from_numpy(batch_data).long().to(device)
        with torch.no_grad():
            scores = target_function(batch_data).squeeze_()
            top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), beam_size))
            top_k_targets = top_k_targets.cpu().numpy()
            
            for j, example in enumerate(batch_data):
                src, rel, dst = example
                id_list = np.where(top_k_targets[j]==dst.item())[0]
                dst_index = id_list[0] if len(id_list) == 1 else beam_size
                ranks.append(dst_index+1)
                multi_dst = torch.from_numpy(pair_filter[(src.item(), rel.item())]).long().to(device)
                target_score = float(scores[j, dst])
                scores[j, multi_dst] = 0
                scores[j, dst] = target_score
            top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), beam_size))
            top_k_targets = top_k_targets.cpu().numpy()
            for j, example in enumerate(batch_data):
                src, rel, dst = example
                id_list = np.where(top_k_targets[j]==dst.item())[0]
                dst_index = id_list[0] if len(id_list) == 1 else beam_size
                franks.append(dst_index+1)
    beam_size = max(ranks) * 4
    return np.array(ranks).astype('float'), np.array(franks).astype('float')
            

def __triple_calculate(target_function, data, num_ent=0, batch_size=512, pair_filter=None,
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
    
    tbar = tqdm(data[1], ncols=100)
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
            batch_data = torch.from_numpy(batch_data).long().to(device)
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

def __cal(flag='triple'):
    if flag == 'triple':
        return __triple_calculate
    else:
        return __pair_calculate
                
def original_data_cal(function, data, num_ent=0, num_rel=0, batch_size=512,
                        all_data=None, device='cpu'):
    
    all_data = np.concatenate((all_data['train'], all_data['valid'], all_data['test']))
    pair_filter = build_graph(all_data)[0]    
    tranks, tfranks = __cal()(function, data, num_ent, batch_size, pair_filter, device)

    data = T(data)
    all_data = T(all_data)
    pair_filter = build_graph(all_data)[0]
    hranks, hfranks = __cal()(function, data, num_ent, batch_size, pair_filter, device, False)

    return (tranks, tfranks), (hranks, hfranks)

def double_data_cal(function, data, num_ent=0, num_rel=0, batch_size=512,
                      all_data=None, device='cpu'):
    all_data = np.concatenate((all_data['train'], all_data['valid'], all_data['test']))
    all_data = add_reverse(all_data, num_rel)
    pair_filter = build_graph(all_data)[0]
    tranks, tfranks =  __cal('pair')(function, data, num_ent, batch_size, pair_filter, device)
    
    data = add_reverse(data, num_rel, concate=False)
    hranks, hfranks = __cal('pair')(function, data, num_ent, batch_size, pair_filter, device)
    
    return (tranks, tfranks), (hranks, hfranks)

def select_head(data):
    left_entity = {}    # the entity on the left of relation in a triplet.
    right_entity = {}     # the entity on the right of relation in a triplet.
    rel_set = set()
    
    for fp in ['train', 'valid', 'test']:
        for triplet in data[fp]:
            src, rel, dst = triplet[0], triplet[1], triplet[2]
            rel_set.add(rel)
            if rel not in left_entity:
                left_entity[rel] = {}
            if src not in left_entity[rel]:
                left_entity[rel][src] = 0
            left_entity[rel][src] += 1
            
            if rel not in right_entity:
                right_entity[rel] = {}
            if dst not in right_entity[rel]:
                right_entity[rel][dst] = 0
            right_entity[rel][dst] += 1
    
    left_avg = {}
    for i in list(rel_set):
        left_avg[i] = sum(left_entity[i].values()) * 1.0 / len(left_entity[i])
    right_avg = {}
    for i in list(rel_set):
        right_avg[i] = sum(right_entity[i].values()) * 1.0 / len(right_entity[i])
    headSelector = {}
    for i in list(rel_set):
        headSelector[i] = 1.0 * left_avg[i] / (left_avg[i] + right_avg[i])
    return headSelector


class Base():
    def __init__(self, data=None, num_ent=0, num_rel=0, model=None, opt=None, lr=0.001,
                 loss_function=None, batch_size=1, device='cpu', reverse=False, **kwargs):
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.data = data
        self.head_selector = select_head(data)
        if reverse:
            self.train_data = add_reverse(data['train'], self.num_rel)
        else:
            self.train_data = self.data['train']
        
        self.device = device
        for k, v in kwargs.items():
            self.__dict__[k] = v
        print('The structure of model:')
        print(model)
        print('The model is running on the device of {}'.format(device))
        print()
        
        self.__model_name = model.__class__.__name__
        self.model = model.to(device)
        self.use_torch_loss = True
        if loss_function is None:
            self.loss = model.forward
            self.use_torch_loss = False
        else:
            self.loss = loss_function(self.model)

        self.opt = opt(self.model.parameters(), lr=lr)

        self.size = {k: len(v) for k, v in data.items()}
        self.num_batch = len(data['train']) // batch_size + 1
        self.batch_size = batch_size
        
        self.valid_write_at_epoch_id = None
        self.cal_rank_flag = None
    
    def set_opt(self, opt):
        self.opt = opt
    
    def set_loss(self, loss):
        self.loss = loss
    
    def train_step(self, *data):
        batch_data = [torch.from_numpy(i).to(self.device) for i in data]
        
        if len(batch_data) == 1:
            batch_label = batch_data[-1][:, 2]
            batch_data = [batch_data[0], batch_label]
        loss = self.loss(*batch_data)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
    
    def sample_iter(self, *data, **kwargs):
        pass
    
    def train_iter(self, **kwargs):
        train_data = np.random.permutation(self.train_data)
        batch_size = self.batch_size
        if 'global_triplets' in self.__dict__ and self.global_triplets is None:
            self.global_triplets = get_all_triplets_set(train_data)
        pbar = trange(self.num_batch, ncols=120)
        avg_loss = []
        for i in pbar:
            epoch, loss = yield self.sample_iter(train_data[i*batch_size: (i+1)*batch_size,:], **kwargs)
            pbar.set_description(desc=f'Epoch {epoch+1:<3d}')
            avg_loss.append(loss)
            pbar.set_postfix(BatchLoss=f'{loss:<.4f}', AvgLoss=f'{np.mean(avg_loss):<.4f}')
    
    def fit(self, num_epoch=1, batch_size=None, scheduler_step=50, gamma=0.99, valid_predict=None, valid_data='valid', **kwargs):
        '''
        Args:
        num_epoch: Number of training epoch.
        batch_size: Batch size.
        scheduler_step: the step of scheduler method StepLR.
        gamma: the gamma in StepLR.
        valid_predict: the predict function of model in valid processing.
        '''
        if 'num_epoch' in self.__dict__.keys():
            num_epoch = self.num_epoch
        if batch_size is None:
            if 'batch_size' not in self.__dict__.keys():
                return
            else:
                batch_size = self.batch_size
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=scheduler_step, gamma=gamma)
        
        for i in range(num_epoch):
            train_consume = self.train_iter(**kwargs)
            data = next(train_consume)
            for _ in range(self.num_batch):
                try:
                    data = train_consume.send((i, self.train_step(*data)))
                except StopIteration:
                    break
            
            if (i+1) % scheduler_step == 0:
                scheduler.step()
                if not valid_predict is None:
                    self.valid_write_at_epoch_id = i+1
                    self.eval(valid_predict, valid_data, self.batch_size, filename='valid.txt')
    
    def cal_rank(self, flags):
        flags = flags if self.cal_rank_flag is None else self.cal_rank_flag
        if flags == 'original':
            return original_data_cal
        else:
            return double_data_cal
    
    def eval(self, function, test_data='test', batch_size=1024, flags='original',
             filename='test.txt'):
        """
        
        paramaters:
        function: the predict function of models.
        test_data: test/valid data.
        batch_size: the size of the each batch data.
        flags: using the original data or not, new only two type, 'original', 'double'
        filename: conf name.
        device: 'cpu' or 'gpu'.
        
        """
        now = arrow.now().format('YYYY-MM-DD')
        if isinstance(test_data, str):
            if test_data == 'test':
                test_data = self.data['test']
                print('Evaluating on the test dataset.')
            else:
                test_data = self.data['valid'] if not test_data == 'sample_valid' else self.data['valid'][np.random.choice(len(self.data['valid']), size=1000), :]
                print(f'Evaluating on the valid dataset at epoch {self.valid_write_at_epoch_id}.')
        else:
            print('Waiting For the evaluation...')
        
        dir_path = os.path.join(os.getcwd(), 'conf')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        
        data_name = self.data['name']
        fs = filename.split('.')[0].replace('/', '')
        filename = f'{now} {self.__model_name} {fs} {chr(960)} {data_name}.txt'
        filename = os.path.join(dir_path, filename)
        (tranks, tfranks), (hranks, hfranks) = self.cal_rank(flags)(
            function(self.model), test_data, self.num_ent, self.num_rel, batch_size, self.data, self.device
        )
        
        if self.valid_write_at_epoch_id is None:
            pprint('\t The results of calulating the ranks on ' + data_name + now, filename)
        else:
            pprint('\t The results of calulating the ranks on ' + data_name+ ' at Epoch '
                   + str(self.valid_write_at_epoch_id) + arrow.now().format(' HH:mm:ss'), filename)
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


class TrainEval_By_Triplet(Base):
    
    def __init__(self, data=None, num_ent=0, num_rel=0, model=None, opt=None, lr=0.001, loss_function=None,
                 negative_rate=1, batch_size=0, device='cpu', use_negative_sample=True):
        super(TrainEval_By_Triplet, self).__init__(data=data, num_ent=num_ent, lr=lr,
                                    num_rel=num_rel, model=model, opt=opt, loss_function=loss_function,
                                    device=device, negative_rate=negative_rate,
                                    global_triplets=get_all_triplets_set(data),
                                    batch_size=batch_size)

        self.use_negative_sample = use_negative_sample
    
    def negative_sample(self, pos_samples):
        size = len(pos_samples)
        num_to_generate = size * self.negative_rate
        neg_samples = np.tile(pos_samples, (self.negative_rate, 1))
        labels = np.ones(size * (self.negative_rate + 1),
                         dtype=np.float32) * (-1.)
        labels[: size] += 2
        
        values = np.random.randint(self.num_ent, size=num_to_generate)
        choices = np.random.uniform(size=num_to_generate)
        prob = np.array([self.head_selector[i[1]] for i in neg_samples])
        subj = choices <= prob
        obj = choices > prob
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]
        
        if not self.use_negative_sample:
            for i, p in enumerate(choices):
                while True:
                    triplet = (neg_samples[i, 0], neg_samples[i, 1], neg_samples[i, 2])
                    if triplet not in self.global_triplets:
                        break
                    if p <= prob[i]:
                        neg_samples[i, 0] = np.random.choice(self.num_ent)
                    else:
                        neg_samples[i, 2] = np.random.choice(self.num_ent)
        return [np.concatenate((pos_samples, neg_samples)).astype('int64'), labels]

    def sample_iter(self, *data):
        return self.negative_sample(*data)
     

class TrainEval_By_Pair(Base):
    
    def __init__(self, data=None, num_ent=0, num_rel=0, model=None, opt=None, loss_function=None,
                 batch_size=0, lr=0.001, device='cpu', reverse=True, use_negative_sample=True):
        super(TrainEval_By_Pair, self).__init__(
            data=data, num_ent=num_ent, num_rel=num_rel, model=model, opt=opt, loss_function=loss_function,
            batch_size=batch_size, device=device, reverse=reverse, lr=lr)
        if reverse:
            len_data = len(data['train']) * 2
            self.num_batch = len_data // batch_size + 1
            self.cal_rank_flag = 'pair'
        self.train_graph = build_graph(self.train_data)[0]
        
        self.use_negative_sample = use_negative_sample
        if use_negative_sample:
            self.negative_sample_rate = 20
    
    def sample_iter(self, *data):
        data = data[0]
        if self.use_negative_sample:
            batch_size = len(data)
            label = np.zeros([batch_size, self.num_ent])
            for i, triple in enumerate(data):
                src, rel, _ = triple
                multi_dst = self.train_graph[(src, rel)]
                if len(multi_dst) > 100:
                    multi_dst = np.random.permutation(multi_dst)[:100]
                label[i][multi_dst] += 1
            return [data, label]
        return [data]
            


class TrainEval_For_Trans(Base):
    
    def __init__(self, data=None, num_ent=0, num_rel=0, model=None, opt=None, loss_function=None,
                 batch_size=0, device='cpu', lr=0.001, use_negative_sample=True):
        self.use_negative_sample = use_negative_sample
        super(TrainEval_For_Trans, self).__init__(data=data, num_ent=num_ent,
                                    num_rel=num_rel, model=model, opt=opt, lloss_function=loss_function,
                                    device=device, batch_size=batch_size, lr=lr,
                                    global_triplets=get_all_triplets_set(data))
    
    def sample(self, pos_data):  
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
        train_data = pos_data
        
        num_to_generate = len(pos_data)
        values = np.random.randint(self.num_ent, size=num_to_generate)
        choices = np.random.uniform(size=num_to_generate)
        p = np.array([self.head_selector[i[1]] for i in train_data])
        subj = choices <= p
        obj = choices > p
        
        pos_head_samples = train_data[subj, :]
        neg_head_samples = pos_head_samples.copy()
        neg_head_samples[:, 0] = values[subj]
        if not self.use_negative_sample:
            filt(neg_head_samples, global_triplets)
        
        pos_tail_samples = train_data[obj, :]
        neg_tail_samples = pos_tail_samples.copy()
        neg_tail_samples[:, 2] = values[obj]
        if not self.use_negative_sample:
            filt(neg_tail_samples, global_triplets, for_neg_head=False)
        
        pos_samples = np.concatenate((pos_head_samples, pos_tail_samples))
        neg_samples = np.concatenate((neg_head_samples, neg_tail_samples))
        ret_index = np.random.permutation(np.arange(num_to_generate))
            
        return pos_samples[ret_index].astype('int64'), neg_samples[ret_index].astype('int64')
    
    def sample_iter(self, *data):
        return self.sample(*data)

