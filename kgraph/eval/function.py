#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/28 22:19:18
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
import os, arrow
import torch
import numpy as np
import prettytable as pt

from tqdm import tqdm

from .metrics import mr_score, mrr_score, hits_at_n_score
from ..data.data_utils import build_graph, src_T_dst
from ..data.data_utils import get_triple_set
from ..datasets.data import DataBase
from ..function.sample import check_sample_by_corrupting_src
from ..function.sample import check_sample_by_corrupting_dst


def pprint(meg, filename=None):
    
    if filename is not None:
        with open(filename, 'a') as f:
            f.write(meg+'\n')
    print(meg)

def add_reverse(data, concate=True):
    old_data = data.copy()
    src_, rel, dst_ = old_data.transpose(1, 0)
    if concate:
        src = np.concatenate((src_, dst_), 0)
        rel = np.concatenate((rel*2, rel*2 + 1))
        dst = np.concatenate((dst_, src_), 0)
        return np.stack((src, rel, dst)).transpose(1, 0)
    rel = rel * 2 + 1
    return np.stack((dst_, rel, src_)).transpose(1, 0)


# #################################################################################################
# Link prediction    

def cal_mr_mrr_hits(ranks, des='ranks'):
    mr = mr_score(ranks)
    mrr = mrr_score(ranks)
    hits = [hits_at_n_score(ranks, i) for i in [1, 3, 5, 10, 100]]
    return [des, mr, mrr, hits[0], hits[1], hits[2], hits[3], hits[4]]

def calculate_rank(test_labels, pred_socres, filter_labels, num_ent=0, constraint=None):   
    def where(ternary_list, index, break_num=0):
        rank, frank = [], []
        num = 0
        j = 0
        for i, x in enumerate(ternary_list):
            if constraint is not None:
                if index[i] not in constraint:
                    j += 1
            if x == 1 or x == -1:
                if x == -1:
                    frank.append(i-num-j+1)
                    rank.append(i-j+1)
                num += 1
            if num == break_num:
                break
        return np.array(rank), np.array(frank)
    
    all_labels_ternary = np.zeros(num_ent)
    filter_labels = list(filter_labels)
    test_labels = list(test_labels)
    all_labels_ternary[filter_labels] = 1
    all_labels_ternary[test_labels] = -1
    
    idx = np.argsort(pred_socres)[::-1]
    
    for_break = len(filter_labels)
    
    labels_ord = all_labels_ternary[idx]
    return where(labels_ord, idx, break_num=for_break)

def calculate_ranks(target_function, data, num_ent=0, batch_size=512, pair_filter=None,
                    device='cpu', pred_tail=True, constraint=None):
    """ 
    paramaters:
    target_function: the function for models to get scores.
    data: the valid / test data
    num_ent: the number of total entities.
    batch_size: the each size of batch test data.
    pair_filter: the set of tail entities for each Pair (head, relation) in all data.
    """
    graph = build_graph(data)
    num_batchs = num_ent // batch_size + 1
    ranks, franks = [], []
    
    try:
        with tqdm(graph['pairs'], ncols=100, leave=False) as tbar:
            new_ents = np.arange(num_ent).reshape((-1, 1))
            scores = np.zeros(num_ent)
            for pair in tbar:
                scores -= scores
                
                new_pairs = np.tile(pair.reshape((1, -1)), (num_ent, 1))
                new_triplets = np.concatenate((new_pairs, new_ents), 1)
                if not pred_tail:
                    new_triplets = src_T_dst(new_triplets)
            
                for i in range(num_batchs):
                    j = i * batch_size
                    batch_data = new_triplets[j: j + batch_size, :]
                    batch_data = torch.from_numpy(batch_data).long()
                    batch_data = batch_data.to(device)
                    with torch.no_grad():
                        score = target_function(batch_data).squeeze_()
                        if not isinstance(score, np.ndarray):
                            if score.is_cuda:
                                score = score.cpu().numpy()
                            else:
                                score = score.numpy()
                        scores[j: j + batch_size] += score
                rank, frank = calculate_rank(graph['pair->rel_set'][(pair[0], pair[1])],
                                            scores, pair_filter[(pair[0], pair[1])],
                                            num_ent=num_ent, constraint=constraint[pair[1]])
                for i in rank:
                    ranks.append(i)
                for i in frank:
                    franks.append(i)
    except KeyboardInterrupt:
        tbar.close()
        raise
    tbar.close()
    
    del graph
    return np.array(ranks), np.array(franks)

def original_data_cal(function, data, num_ent=0, num_rel=0, batch_size=512,
                        all_data=None, constraint=None, device='cpu'):
    
    all_data = np.concatenate((all_data['train'], all_data['valid'], all_data['test']))
    pair_filter = build_graph(all_data)['pair->rel_set']  
    tranks, tfranks = calculate_ranks(function, data, num_ent, batch_size,
                                      pair_filter, device, True, constraint['rel_dst'])

    data = src_T_dst(data)
    all_data = src_T_dst(all_data)
    pair_filter = build_graph(all_data)['pair->rel_set']
    hranks, hfranks = calculate_ranks(function, data, num_ent, batch_size,
                                      pair_filter, device, False, constraint['rel_src'])

    return (tranks, tfranks), (hranks, hfranks)

def double_data_cal(function, data, num_ent=0, num_rel=0, batch_size=512,
                    all_data=None, constraint=None, device='cpu'):
    all_data = np.concatenate((all_data['train'], all_data['valid'], all_data['test']))
    all_data = add_reverse(all_data)
    pair_filter = build_graph(all_data)['pair->rel_set']
    _data = data.copy()
    _data[:, 1] *= 2
    tranks, tfranks = calculate_ranks(function, _data, num_ent, batch_size,
                                      pair_filter, device)
    
    _data = add_reverse(data, concate=False)
    hranks, hfranks = calculate_ranks(function, _data, num_ent, batch_size,
                                      pair_filter, device)
    
    return (tranks, tfranks), (hranks, hfranks)

def cal_rank(flags):
    if flags == 'original':
        return original_data_cal
    return double_data_cal


def link_prediction(function, model_name, data, num_ent, num_rel, test_flag='test',
                    epoch_i=None, batch_eval_size=2048, constraint=True, flags='original', device='cpu'):
    
    assert isinstance(data, DataBase)
    # data_name = data['name']
    data_name = data.name
    constraint = data.load_constraint() if constraint else None
    data = data.data
    test_data = data[test_flag]
    
    
    dir_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    file_now = arrow.now().format('YYYY-MM-DD')
    
    file_name = f'{file_now}_{model_name}_{chr(960)}_{data_name}.txt'
    file_path = os.path.join(dir_path, file_name)
    (rhs_ranks, rhs_franks), (lhs_ranks, lhs_franks) = cal_rank(flags)(function, test_data,
                                num_ent, num_rel, batch_eval_size, data, constraint, device)
    
    
    if epoch_i is None:
        now = arrow.now().format('YYYY-MM-DD HH:mm')
        pprint(f'\t\t\t   The results of ranks on {data_name} {now}', file_path)
    else:
        now = arrow.now().format('YYYY-MM-DD HH:mm:ss')
        pprint(f'\t\t\t   The results of ranks on {data_name} at Epoch {epoch_i} {now}', file_path)
    
    tb = pt.PrettyTable()
    tb.field_names = [' Evaluation ', 'MR', 'MRR (%)', 'Hits@1 (%)', 'Hits@3 (%)', 'Hits@5 (%)', 'Hits@10 (%)', 'Hits@100 (%)']
    tb.add_row(cal_mr_mrr_hits(lhs_ranks, des='Head'))
    tb.add_row(cal_mr_mrr_hits(rhs_ranks, des='Tail'))

    ranks = np.concatenate((rhs_ranks, lhs_ranks), 0)
    franks = np.concatenate((rhs_franks, lhs_franks), 0)
    tb.add_row(cal_mr_mrr_hits(ranks, des='Average'))
    tb.add_row(['', '', '', '', '', '', '', ''])
    tb.add_row(cal_mr_mrr_hits(lhs_franks, des='HeadFilter'))
    tb.add_row(cal_mr_mrr_hits(rhs_franks, des='TailFilter'))
    tb.add_row(cal_mr_mrr_hits(franks, des='AverageFilter'))
    tb.float_format = "2.4"
    
    print(tb)
    
    tb = tb.get_string()
    print(f'The results are writing in this file {file_path}')
    
    with open(file_path, 'a') as f:
        f.write(tb)
    pprint('\n', file_path)

def link_n2n_prediction(function, model_name, data, num_ent, num_rel, batch_eval_size=2048, flags='original', device='cpu'):
    
    assert isinstance(data, DataBase)
    data_name = data.name
    constraint = data.load_constraint()
  
    dir_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    file_now = arrow.now().format('YYYY-MM-DD')
    
    file_name = f'{file_now}_{model_name}_{chr(960)}_{data_name}.txt'
    file_path = os.path.join(dir_path, file_name)
    
    all_data = data.data
    
    data = {'1 to 1': data.load_1_1(),
            '1 to N': data.load_1_n(),
            'N to 1': data.load_n_1(),
            'N to N': data.load_n_n()}
    now = arrow.now().format('YYYY-MM-DD HH:mm')
    pprint(f'\t\t\t   The results of ranks on {data_name} {now}', file_path)
    tb = pt.PrettyTable()
    tb.field_names = [f' {data_name} ', ' Evaluation ', 'MR', 'MRR (%)', 'Hits@1 (%)',
                      'Hits@3 (%)', 'Hits@5 (%)', 'Hits@10 (%)', 'Hits@100 (%)']
    
    for key, test_data in data.items():
 
        (rhs_ranks, rhs_franks), (lhs_ranks, lhs_franks) = cal_rank(flags)(function, test_data, num_ent,
                                                num_rel, batch_eval_size, all_data, constraint, device)
        ranks = np.concatenate((rhs_ranks, lhs_ranks), 0)
        franks = np.concatenate((rhs_franks, lhs_franks), 0)
        
        tb.add_row([f'{key}']+cal_mr_mrr_hits(lhs_ranks, des='Head'))
        tb.add_row([f'{key}']+cal_mr_mrr_hits(rhs_ranks, des='Tail'))
        tb.add_row([f'{key}']+cal_mr_mrr_hits(ranks, des='Average'))
        tb.add_row([f'{key}']+cal_mr_mrr_hits(lhs_franks, des='HeadFilter'))
        tb.add_row([f'{key}']+cal_mr_mrr_hits(rhs_franks, des='TailFilter'))
        tb.add_row([f'{key}']+cal_mr_mrr_hits(franks, des='AverageFilter'))
        if not key == 'N to N':
            tb.add_row(['', '', '', '', '', '', '', '', ''])
        
    tb.float_format = "2.4"
    
    print(tb)
    
    tb = tb.get_string()
    with open(file_path, 'a') as f:
        f.write(tb)
    pprint('\n', file_path)
    print(f'The results are writing in this file {file_path}')

# ######################################################################################################
# classification

def get_neg_test(data, flags='test'):
    data = data.data
    all_data = np.concatenate((data['train'], data['valid'], data['test']))
    all_ent = np.unique(np.concatenate((all_data[:, 0], all_data[:, 2])), axis=0)
    num_ent = len(all_ent)
    del all_ent
    all_triples = get_triple_set(all_data)
    
    neg_samples = data[flags].copy()
    generate_choice_rate = np.random.uniform(0, 1, len(neg_samples))
    lhs_idx = generate_choice_rate <= 0.5
    rhs_idx = generate_choice_rate > 0.5
    
    check_sample_by_corrupting_src(neg_samples[lhs_idx, :], num_ent, all_triples)
    check_sample_by_corrupting_dst(neg_samples[rhs_idx, :], num_ent, all_triples)
    
    return data[flags], neg_samples

def get_best_threshold(score, ans):
    res = np.concatenate([ans.reshape(-1, 1), score.reshape(-1, 1)], axis=-1)
    order = np.argsort(score)
    res = res[order]
    
    total_all = (float)(len(score))
    total_current = 0.0
    total_true = np.sum(ans)
    total_false = total_all - total_true
    res_mx = 0.0
    threshold = None
    for index, [ans, score] in enumerate(res):
        if ans == 1:
            total_current += 1.0
        res_current = (2 * total_current + total_false - index + 1) / total_all
        if res_current > res_mx:
            res_mx = res_current
            threshold = score
    return threshold

def get_acc(score, ans, threshold):
    res = np.concatenate([ans.reshape(-1, 1), score.reshape(-1, 1)], axis=-1)
    order = np.argsort(score)
    res = res[order]
    
    total_all = (float)(len(score))
    total_current = 0.0
    total_true = np.sum(ans)
    total_false = total_all - total_true

    for index, [ans, score] in enumerate(res):
        if score > threshold:
            return (2 * total_current + total_false - index) / total_all
        if ans == 1:
            total_current += 1.0

def calculate_acc(target_function, data, batch_size=1000, test_flags='test', device='cpu'):
    def get_test_batch():
        pos_samples, neg_samples = get_neg_test(data, test_flags)
        
        num_batchs = len(pos_samples) // batch_size + 1
        for i in range(num_batchs):
            yield pos_samples[i*batch_size: (i+1)*batch_size, :], neg_samples[i*batch_size: (i+1)*batch_size, :]
    
    score = []
    ans = []
    try:
        with tqdm(get_test_batch()) as tbar:
            for batch_pos, batch_neg in tbar:
                batch_pos = torch.from_numpy(batch_pos).long().to(device)
                batch_neg = torch.from_numpy(batch_neg).long().to(device)
                
                with torch.no_grad():
                    pos_score = target_function(batch_pos).squeeze_()
                    neg_score = target_function(batch_neg).squeeze_()
                    if not isinstance(pos_score, np.ndarray):
                        if pos_score.is_cuda:
                            pos_score = pos_score.cpu().numpy()
                        else:
                            pos_score = pos_score.numpy()
                    if not isinstance(neg_score, np.ndarray):
                        if neg_score.is_cuda:
                            neg_score = neg_score.cpu().numpy()
                        else:
                            neg_score = neg_score.numpy()
                
                ans += [1 for i in range(len(pos_score))]
                score.append(pos_score*(-1))
                
                ans += [0 for i in range(len(neg_score))]
                score.append(neg_score*(-1))
    except KeyboardInterrupt:
        tbar.close()
        raise
    tbar.close()
    
    score = np.concatenate(score, 0)
    ans = np.asarray(ans)
    threshold = get_best_threshold(score, ans)
    acc = get_acc(score, ans, threshold)
    return [acc, threshold]
    
def classification(function, model_name, data, batch_size=1000, test_flag='test', device='cpu'):
    data_name = data.name   

    dir_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_now = arrow.now().format('YYYY-MM-DD')
    file_name = f'{file_now}_{model_name}_{chr(960)}_{data_name}.txt'
    file_path = os.path.join(dir_path, file_name)
    
    now = arrow.now().format('YYYY-MM-DD HH:mm')
    pprint(f'\t The results of ranks on {data_name} {now}', file_path)
    
    tb = pt.PrettyTable()
    tb.field_names = [' Evaluation ', 'Acc', 'Threshold']
    tb.add_row([data_name,] + calculate_acc(function, data, batch_size, test_flag, device))
    tb.float_format = "2.4"
    
    print(tb)
    tb = tb.get_string()
    print(f'The results are writing in this file {file_path}')
    
    with open(file_path, 'a') as f:
        f.write(tb)
    pprint('\n', file_path)





















