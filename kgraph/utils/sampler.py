#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2021/02/27 06:59:49
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
            sample[i, 1] = np.random.choice(num_rel)


class Sampler(object):
    
    def __init__(self, invalid_valid_ratio=1) -> None:
        self.all_triples = None
        self.num_entities = None
        self.select_src_rate = None
        
        assert invalid_valid_ratio > 0, 'invalid_valid_ratio should be greater than 0'
        self.reset_invalid_valid_ratio(invalid_valid_ratio)
    
    def reset_invalid_valid_ratio(self, ratio=1):
        self.invalid_valid_ratio = ratio
    
    def sampler(self, pos_sample, batch_size):
        
        num_negative_sample = int(batch_size * self.invalid_valid_ratio)
        neg_sample = np.tile(pos_sample.copy(), (self.invalid_valid_ratio, 1))
        
        corrupted_values = np.random.choice(self.num_entities, num_negative_sample)
        generate_choice_rate = np.random.uniform(0, 1, num_negative_sample)
        if self.select_src_rate is not None:
            choice_prob = np.array([self.select_src_rate[i[1]] for i in neg_sample])
        else:
            choice_prob = 0.5
        
        lhs_idx = generate_choice_rate <= choice_prob
        rhs_idx = generate_choice_rate > choice_prob
        
        lhs_pos = pos_sample[lhs_idx[:batch_size], :]
        lhs_neg = neg_sample.copy()[lhs_idx, :]
        lhs_neg[:, 0] = corrupted_values[lhs_idx]
        check_sample_by_corrupting_src(lhs_neg, self.num_entities, self.all_triples)
        
        rhs_pos = pos_sample[rhs_idx[:batch_size], :]
        rhs_neg = neg_sample.copy()[rhs_idx, :]
        rhs_neg[:, 2] = corrupted_values[rhs_idx]
        check_sample_by_corrupting_dst(rhs_neg, self.num_entities, self.all_triples)
        
        return lhs_pos, rhs_pos, lhs_neg, rhs_neg
    
    def __call__(self, batch_data, batch_size):
        return self.sampler(batch_data, batch_size)

