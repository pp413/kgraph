#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2021/02/27 13:42:59
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
import torch
import numpy as np
import prettytable as pt
from torch._C import device

from tqdm import tqdm
from ..data import build_graph, src_T_dst
from .metrics import mr_score
from .metrics import mrr_score
from .metrics import hits_at_n_score

# calculate the mr, mrr and hits
def cal_mr_mrr_hits(ranks, des='results'):
    mr = mr_score(ranks)
    mrr = mrr_score(ranks)
    hits = [hits_at_n_score(ranks, i) for i in [1, 3, 5, 10, 100]]
    return [des, mr, mrr,] + hits

def _get_rank(current_index, pred_socres, filter_index, num_ent, constraint=None):
    
    def where(temp, index, break_sign=0):
        _rank, _frank = [], []
        num, j = 0, 0
        for i, x in enumerate(temp):
            if constraint is not None and index[i] not in constraint:
                j += 1
            if x == 1 or x == -1:
                if x == -1:
                    _frank.append(i-num-j+1)
                    _rank.append(i-j+1)
                num += 1
            if num == break_sign:
                break
        return np.array(_rank), np.array(_frank)
    
    all_labels_ternary = np.zeros(num_ent)
    filter_labels = list(filter_index)
    test_labels = list(current_index)
    all_labels_ternary[filter_labels] += 1
    all_labels_ternary[test_labels] = -1
    
    idx = np.argsort(pred_socres)[::-1]
    _break_sign = len(filter_labels)
    labels_ord = all_labels_ternary[idx]
    
    return where(labels_ord, idx, break_sign=_break_sign)

# calculate the ranks
def _calculate_ranks(pred_function, data, unique_entities, batch_size, device,
                     pair_filter, pred_tail=True, constraint=None):
    
    num_ent = len(unique_entities)
    pred_ents = np.array(unique_entities).reshape((-1, 1))
    
    graph = build_graph(data)
    
    num_batchs = (num_ent - 1) // batch_size + 1
    _ranks, _franks = [], []
    
    try:
        with tqdm(graph['pairs'], ncols=100, leave=False) as tbar:
            scores = np.zeros(num_ent)
            pred_ents = torch.from_numpy(pred_ents).long()
            
            for pair in tbar:
                scores -= scores
                _pair = torch.from_numpy(pair).long()
                per_pairs = _pair.view(1, -1).repeat(num_ent, 1)
                
                new_triples = torch.cat((per_pairs, pred_ents), 1)
                if not pred_tail:
                    src, rel, dst = new_triples.transpose(1, 0)
                    new_triples = torch.stack((dst, rel, src), 0).transpose(1, 0)
                
                for i in range(num_batchs):
                    j = i * batch_size
                    batch_data = new_triples[j: j+batch_size, :].to(device)
                    with torch.no_grad():
                        score = pred_function(batch_data).squeeze_()
                        if not isinstance(score, np.ndarray):
                            if score.is_cuda:
                                score = score.cpu().numpy()
                            else:
                                score = score.numpy()
                        scores[j: j + batch_size] += score
                
                _rank, _frank = _get_rank(graph['pair->rel_set'][(pair[0], pair[1])],
                                scores, pair_filter[(pair[0], pair[1])], num_ent=num_ent,
                                constraint=None if constraint is None else constraint[pair[1]])
                for i in _rank:
                    _ranks.append(i)
                for i in _frank:
                    _franks.append(i)
                # print(_ranks)
                # print(_franks)
    except KeyboardInterrupt:
        tbar.close()
        raise
    del graph
    return np.array(_ranks), np.array(_franks)
                    
# ########################################
def calculate_ranks(function, dataset, for_test=True, batch_size=None, device='cpu', constraint=None):
    
    data = dataset.test if for_test else dataset.valid
    unique_entities = dataset.unique_entities
    batch_size = len(unique_entities) if batch_size is None else batch_size
    all_data = dataset.all_triples
    _constraint = None if constraint is None else dataset.load_constraint()
    pair_filter = build_graph(all_data)['pair->rel_set']
    rhs_ranks, rhs_franks = _calculate_ranks(function, data,
                    unique_entities, batch_size, device, pair_filter=pair_filter, pred_tail=True,
                    constraint=None if constraint is None else _constraint['rel_dst'])
    
    data = src_T_dst(data)
    all_data = src_T_dst(all_data)
    pair_filter = build_graph(all_data)['pair->rel_set']
    lhs_ranks, lhs_franks = _calculate_ranks(function, data,
                    unique_entities, batch_size, device, pair_filter=pair_filter, pred_tail=False,
                    constraint=None if constraint is None else _constraint['rel_src'])
    
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
    
    return tb.get_string()
                   
def calculate_n2n_ranks(function, dataset, batch_size=None, device='cpu', constraint=None):
    data = {'1 to 1': dataset.load_1_1(),
            '1 to N': dataset.load_1_n(),
            'N to 1': dataset.load_n_1(),
            'N to N': dataset.load_n_n()}
    unique_entities = dataset.unique_entities
    batch_size = len(unique_entities) if batch_size is None else batch_size
    all_data = dataset.all_triples
    pair_filter = build_graph(all_data)['pair->rel_set']
    _all_data = src_T_dst(all_data)
    _pair_filter = build_graph(_all_data)['pair->rel_set']
    _constraint = None if constraint is None else dataset.load_constraint()
    
    tb = pt.PrettyTable()
    tb.field_names = [f' {dataset.name} ', ' Evaluation ', 'MR', 'MRR (%)', 'Hits@1 (%)',
                      'Hits@3 (%)', 'Hits@5 (%)', 'Hits@10 (%)', 'Hits@100 (%)']
    
    for key, test_data in data.items():
        rhs_ranks, rhs_franks = _calculate_ranks(function, test_data, unique_entities,
                                            batch_size, device, pair_filter=pair_filter, pred_tail=True,
                                            constraint=None if constraint is None else _constraint['rel_dst'])
        _test_data = src_T_dst(test_data)
        lhs_ranks, lhs_franks = _calculate_ranks(function, _test_data, unique_entities,
                                            batch_size, device, pair_filter=_pair_filter, pred_tail=False,
                                            constraint=None if constraint is None else _constraint['rel_src'])
        ranks = np.concatenate((rhs_ranks, lhs_ranks), 0)
        franks = np.concatenate((rhs_franks, lhs_franks), 0)
        
        tb.add_row([f'{key}']+cal_mr_mrr_hits(lhs_ranks, des='Head'))
        tb.add_row([f'{key}']+cal_mr_mrr_hits(rhs_ranks, des='Tail'))
        tb.add_row([f'{key}']+cal_mr_mrr_hits(ranks, des='Average'))
        tb.add_row([f'{key}']+cal_mr_mrr_hits(lhs_franks, des='HeadFilter'))
        tb.add_row([f'{key}']+cal_mr_mrr_hits(rhs_franks, des='TailFilter'))
        tb.add_row([f'{key}']+cal_mr_mrr_hits(franks, des='AverageFilter'))
        tb.add_row(['', '', '', '', '', '', '', '', ''])
    tb.float_format = "2.4"
    
    return tb.get_string()
                 
            



