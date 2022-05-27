import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
from kgraph import FB15k237
from kgraph import DataIter
from kgraph import Predict
from kgraph.loss import MarginLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class TransE(nn.Module):
    
    def __init__(self, num_ent: int, num_rel: int, embedding_dim: int, p: int=1, norm_flag=True, margin: float=5.0):
        super(TransE, self).__init__()
        
        self.p = 1
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.dim = embedding_dim
        self.norm_flag = norm_flag
        
        self.ent_embeddings = nn.Embedding(num_ent, embedding_dim)
        self.rel_embeddings = nn.Embedding(num_rel, embedding_dim)
        
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        self.criterion = MarginLoss(margin=margin)
    
    def embed_lookup(self, data):
        head = self.ent_embeddings(data[:, 0])
        rel = self.rel_embeddings(data[:, 1])
        tail = self.ent_embeddings(data[:, 2])
        return head, rel, tail
    
    def _calc(self, head, rel, tail):
        if self.norm_flag:
            head = F.normalize(head, p=2, dim=-1)
            rel = F.normalize(rel, p=2, dim=-1)
            tail = F.normalize(tail, p=2, dim=-1)
        score = head + rel - tail
        
        score = torch.norm(score, p=self.p, dim=-1).flatten()
        return score
    
    def regul(self):
        ent_weight = torch.norm(self.ent_embeddings.weight, p=self.p, dim=-1)
        rel_weight = torch.norm(self.rel_embeddings.weight, p=self.p, dim=-1)
        return (ent_weight + rel_weight) / 2
    
    def forward(self, data):
        head, rel, tail = self.embed_lookup(data)
        score = self._calc(head, rel, tail)
        return score
    
    def loss(self, data, label):
        data = torch.from_numpy(data).to(self.ent_embeddings.weight.data.device)
        label = torch.from_numpy(label).to(self.ent_embeddings.weight.data.device)
        score = self.forward(data)
        loss = self.criterion(score, label)
        return loss
    
    @torch.no_grad()
    def predict(self, data):
        data = torch.from_numpy(data).to(self.ent_embeddings.weight.data.device)
        # score = self.forward(data)
        # return score.cpu().numpy()
        mask = torch.arange(data.size(0))
        mask_tail = mask[data[:, 1] < self.num_rel]
        mask_head = mask[data[:, 1] >= self.num_rel]
        
        h = self.ent_embeddings(data[:, 0])
        r = self.rel_embeddings(data[:, 1] % self.num_rel)
        t = self.ent_embeddings.weight.data
        
        if self.norm_flag:
            h = F.normalize(h, p=2, dim=-1)
            r = F.normalize(r, p=2, dim=-1)
            t = F.normalize(t, p=2, dim=-1)
        
        h = h.view(-1, 1, self.dim)
        r = r.view(-1, 1, self.dim)
        t = t.view(1, -1, self.dim)
        
        score = h + r - t
        score[mask_head, :, :] = t + r[mask_head, :, :] - h[mask_head, :, :]
        
        score = torch.norm(score, self.p, -1)
        return score.cpu().numpy()
        

data = FB15k237()
dataiter = DataIter(data, 1000, 16,
                    num_neg=25, bern_flag=1,
                    element_type='triple')

model = TransE(data.num_ent, data.num_rel, 200, 1, norm_flag=True, margin=5.).to(device)
opt = optim.SGD(model.parameters(), lr=1.)

predict = Predict(data, element_type='pair')


avg_loss = 0.0

for batch_data, batch_label in dataiter.generate_triple_with_negative():

    opt.zero_grad()
    l = model.loss(batch_data, batch_label)
    l.backward()
    opt.step()
    avg_loss += l.item()

for batch_data, batch_label in dataiter.generate_triple_with_negative_on_random():
    opt.zero_grad()
    l = model.loss(batch_data, batch_label)
    l.backward()
    opt.step()
    avg_loss += l.item()

results = predict.predict_test(model.predict, data.num_ent, data.num_rel, 200)

print(results[0])
