#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/12/23 10:04:22
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from kgraph.datasets import FB15k237
from kgraph.model import Module

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TransE(Module):
    def __init__(self, num_ent: int, num_rel: int, embedding_dim: int, margin_value: float=0.1):
        super(TransE, self).__init__()
        
        self.margin = margin_value
        self.k = int(num_ent)
        
        self.ent_embeddings = nn.Embedding(num_ent, embedding_dim)
        self.rel_embeddings = nn.Embedding(num_rel, embedding_dim)
        
        nn.init.xavier_normal_(self.ent_embeddings.weight.data)
        nn.init.xavier_normal_(self.rel_embeddings.weight.data)
        
    def embed_lookup(self, data):
        head = self.ent_embeddings(data[:, 0])
        rel = self.rel_embeddings(data[:, 1])
        tail = self.ent_embeddings(data[:, 2])
        return head, rel, tail
    
    def regul(self):
        ent_weight = torch.norm(self.ent_embeddings.weight.data, dim=-1)
        rel_weight = torch.norm(self.rel_embeddings.weight.data, dim=-1)
        
        return (ent_weight.sum() + rel_weight.sum()) / 2.
    
    
    def forward(self, lhs_pos, rhs_pos, lhs_neg, rhs_neg):
        pos_samples = torch.cat([lhs_pos, rhs_pos], dim=0)
        neg_samples = torch.cat([lhs_neg, rhs_neg], dim=0)
        
        pos_head, pos_rel, pos_tail = self.embed_lookup(pos_samples)
        neg_head, neg_rel, neg_tail = self.embed_lookup(neg_samples)
        
        pos_distance = pos_head + pos_rel - pos_tail
        neg_distance = neg_head + neg_rel - neg_tail
        
        pos_distance = torch.norm(pos_distance, dim=-1)
        neg_distance = torch.norm(neg_distance, dim=-1)
        
        return F.relu(self.margin + pos_distance - neg_distance).sum()
    
    def loss(self, lhs_pos, rhs_pos, lhs_neg, rhs_neg):
        return self.forward(lhs_pos, rhs_pos, lhs_neg, rhs_neg)
    
    def predict(self, samples):
        head, rel, tail = self.embed_lookup(samples)
        
        return -torch.norm(head + rel - tail, dim=-1)


dataset = FB15k237()
model = TransE(dataset.entity_total, dataset.relation_total, embedding_dim=100,
               margin_value=0.1).to(device)

# model.fit(dataset, 1000, batch_size=10000, generate_negative_rate=1, lr=1e-4,
#           weight_decay=1e-6)

model.link_prediction(dataset)
# model.link_n2n_prediction(dataset)
# model.classification(dataset)
