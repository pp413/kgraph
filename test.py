#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2021/02/28 20:10:59
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from kgraph._train import initial_graph_model
from kgraph.utils import DataIter
from kgraph.data import FB15k237
from kgraph.log import log_pred

dataset = FB15k237()

data_iter = DataIter(dataset, batch_size=10000, shuffle=True, num_workers=0)

@initial_graph_model(data_iter)
class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, embedding_dim, margin_value=0.1):
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


model = TransE(dataset.entity_total, dataset.relation_total, 100)

model.fit(num_epoch=600, device='cuda')
# model.pred_train_from('./TransE/FB15k-237_2021-03-01.tgz')
model.device = 'cuda'

table = model.link_prediction()
log_pred(table)

