#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/26 17:52:42
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#

import numpy as np

def dim(numpy_array):
    return len(numpy_array.shape)

def check_sample_by_corrupting_dst(sample, num_ent, all_triples):
    # sample = sample if dim(sample) == 2 else sample.reshape(1, -1)
    for i in range(len(sample)):
        while (sample[i, 0], sample[i, 1], sample[i, 2]) in all_triples:
            sample[i, 2] = np.random.choice(num_ent)
            

def check_sample_by_corrupting_src(sample, num_ent, all_triples):
    # sample = sample if dim(sample) == 2 else sample.reshape(1, -1)
    for i in range(len(sample)):
        while (sample[i, 0], sample[i, 1], sample[i, 2]) in all_triples:
            sample[i, 0] = np.random.choice(num_ent)

def check_sample_by_corrupting_rel(sample, num_rel, all_triples):
    # sample = sample if dim(sample) == 2 else sample.reshape(1, -1)
    for i in range(len(sample)):
        while (sample[i, 0], sample[i, 1], sample[i, 2]) in all_triples:
            sample[i, 1] = np.random.choice(num_ent)

def generate_negative_sample(pos_sample, generate_negative_rate, num_ent, select_src_rate, all_triples):
    """
    lhs: (h, r, t) used to compute with (h', r, t)
    rhs: (h, r, t) used to compute with (h, r, t')
    """
    assert not generate_negative_rate == 0, f'the negative rate is {generate_negative_rate}'
    batch_size = len(pos_sample)
    num_negative_sample = int(batch_size * generate_negative_rate)
    neg_sample = np.tile(pos_sample.copy(), (generate_negative_rate, 1))
    
    corrupted_values = np.random.choice(num_ent, num_negative_sample)
    generate_choice_rate = np.random.uniform(0, 1, num_negative_sample)
    choice_prob = np.array([select_src_rate[i[1]] for i in neg_sample])
    
    lhs_idx = generate_choice_rate <= choice_prob
    rhs_idx = generate_choice_rate > choice_prob
    
    lhs_pos = pos_sample[lhs_idx[:batch_size], :]
    lhs_neg = neg_sample.copy()[lhs_idx, :]
    lhs_neg[:, 0] = corrupted_values[lhs_idx]
    check_sample_by_corrupting_src(lhs_neg, num_ent, all_triples)
    
    rhs_pos = pos_sample[rhs_idx[:batch_size], :]
    rhs_neg = neg_sample.copy()[rhs_idx, :]
    rhs_neg[:, 2] = corrupted_values[rhs_idx]
    check_sample_by_corrupting_dst(rhs_neg, num_ent, all_triples)
    
    return lhs_pos, rhs_pos, lhs_neg, rhs_neg
    
    
    
    
    



