# kgraph
A Python library for Graph Embedding on knowledge graphs. Most of the code in this library from an Open-source Framework  OpenKE.

kgraph 是一个知识图谱在知识表示研究上的 ，主要解决常见数据的预处理、加载以及常用的评估。



If you use the code, please cite the following papers:

```tex
@inproceedings{han2018openke,
   title={OpenKE: An Open Toolkit for Knowledge Embedding},
   author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
   booktitle={Proceedings of EMNLP},
   year={2018}
 }
```



### kgraph 主要功能

##### data
kgraph.data 主要解决数据的**预处理**和**加载**。在该模块可以加载常见的用于知识表示研究中的数据集。



```python
from kgraph import *
```



### WN18

|          |  Train  | Valid | Test  | Entities | Relations |
| :------: | :-----: | :---: | :---: | :------: | :-------: |
| Original | 141,442 | 5,000 | 5,000 |  40,943  |    18     |
| Cleaned  |         |       |       |          |           |



```python
data = WN18()
```



### WN18RR

|          | Train  | Valid | Test  | Entities | Relations |
| :------: | :----: | :---: | :---: | :------: | :-------: |
| Original | 86,835 | 3,034 | 3,134 |  40,943  |    11     |
| Cleaned  |        |       |       |          |           |



```python
data = WN18RR()
```



### FB15k

|          |  Train  | Valid  |  Test  | Entities | Relations |
| :------: | :-----: | :----: | :----: | :------: | :-------: |
| Original | 483,142 | 50,000 | 59,071 |  14,951  |   1,345   |
| Cleaned  |         |        |        |          |           |



```python
data = FB15k()
```



### FB15k-237

|          |  Train  | Valid  |  Test  | Entities | Relations |
| :------: | :-----: | :----: | :----: | :------: | :-------: |
| Original | 272,115 | 17,535 | 20,466 |  14,541  |    237    |
| Cleaned  | 272,115 | 17,516 | 20,438 |  14,505  |    237    |



```python
data = FB15k-237()
```



### YAGO3-10

|          |   Train   | Valid | Test  | Entities | Relations |
| :------: | :-------: | :---: | :---: | :------: | :-------: |
| Original | 1,079,040 | 5,000 | 5,000 | 123,182  |    37     |
| Cleaned  |           |       |       |          |           |



```python

```



### Wordnet11

|          | Train  | Valid Pos | Valid Neg | Test Pos | Test Neg | Entities | Relations |
| :------: | :----: | :-------: | :-------: | :------: | :------: | :------: | :-------: |
| Original | 110361 |   2606    |   2609    |  10493   |  10542   |  38588   |    11     |
| Cleaned  |        |           |           |          |          |          |           |



```python

```



### Freebase13

|          | Train  | Valid Pos | Valid Neg | Test Pos | Test Neg | Entities | Relations |
| :------: | :----: | :-------: | :-------: | :------: | :------: | :------: | :-------: |
| Original | 316232 |   5908    |   5908    |  23733   |  23731   |  75043   |    13     |
| Cleaned  |        |           |           |          |          |          |           |



```python

```



### Benchmark DataSets 基数据

```
from kgraph import FB15k237, FB15k, WN18, WN18RR
```



## utils 模块

kgraph.utils 模块主要解决数据的下载。



### DataIter 数据迭代器

DataIter(num_ent,

​			num_rel,

​			batch_size=128,

​			num_threads=2,

​			smooth_lambda=0.1,

​			num_neg=1,

​			mode=‘all’,

​			bern_flag=False,

​			seed=1

)



num_ent: 实体数量

num_rel: 关系数量

batch_size: 

num_threads:

smooth_lambda:

mode: in {‘all’, ‘head’, ‘tail’, ‘head_tail’}

ber_flag: whether to use bernoulli sampling



### Predict测试类。







## Kgraph 应用


### 


#### 例子

```python
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
from kgraph import FB15k237
from kgraph import DataIter
from kgraph import Predict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def link_predict(function):
    
    def f(data):
        with torch.no_grad():
            score = function(data)
            return -score
    return f

class TransE(nn.Module):
    
    def __init__(self, num_ent: int, num_rel: int, embedding_dim: int, p: int=1):
        super(TransE, self).__init__()
        
        self.p = 1
        
        self.ent_embeddings = nn.Embedding(num_ent, embedding_dim)
        self.rel_embeddings = nn.Embedding(num_rel, embedding_dim)
        
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
    
    def embed_lookup(self, data):
        head = self.ent_embeddings(data[:, 0])
        rel = self.rel_embeddings(data[:, 1])
        tail = self.ent_embeddings(data[:, 2])
        return head, rel, tail
    
    def _calc(self, head, rel, tail):
        head = F.normalize(head, 2, -1)
        rel = F.normalize(rel, 2, -1)
        tail = F.normalize(tail, 2, -1)
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
    
    def predict(self, data):
        global device
        with torch.no_grad():
            data = torch.from_numpy(data).to(device)
            score = self.forward(data)
            return score.cpu().numpy()

EPOCH = 1000
BATCH_SIZE = 1000
MARGIN = 5.
DIM = 200
NUM_NEG = 25
P = 1
LR = 1.0

data = FB15k237()

def train_step(model, lr, margin):
    opt = optim.SGD(model.parameters(), lr=lr)
    margin = torch.nn.Parameter(torch.Tensor([margin])).cuda()
    
    def cal_loss(score):
        batch_size = score.size(0) // (NUM_NEG + 1)
        pos_score = score[:batch_size]
        pos_score = pos_score.view(-1, batch_size).permute(1, 0)
        
        neg_score = score[batch_size:]
        neg_score = neg_score.view(-1, batch_size).permute(1, 0)
        return (torch.max(pos_score - neg_score, -margin)).mean() + margin
    
    def _(tmp_batch_data):
        opt.zero_grad()
        score = model(tmp_batch_data)
        loss = cal_loss(score)
        loss.backward()
        opt.step()
        return loss.item()
    return _

dataiter = DataIter(data.num_ent, data.num_rel, BATCH_SIZE, 8, num_neg=NUM_NEG, bern_flag=1)
dataiter.num_batch = 100
predict = Predict()

model = TransE(data.num_ent, data.num_rel, DIM, P).to(device)

training = train_step(model, lr=LR, margin=MARGIN)

for i in trange(1, EPOCH+1):
    avg_loss = 0.0
    for batch_data, batch_label in dataiter.generate_triple_with_negative_on_random():

        batch_data = torch.from_numpy(batch_data).to(device)
        l = training(batch_data)
        avg_loss += l
    
    if i % 50 == 0:
        print('Epoch', i, 'Loss', avg_loss)
        rank = predict.predict_valid(model.predict, data.num_ent, data.num_rel, 10 * BATCH_SIZE)
        rank = np.append(rank[1], rank[-1])
        print(f"MR: {np.mean(rank)}, MRR: {np.mean(1.0 / rank)}, Hits@10: {np.sum(rank <= 10)/len(rank)}\n")

rank = predict.predict_test(model.predict, data.num_ent, data.num_rel, 10 * BATCH_SIZE)
rank = np.append(rank[1], rank[-1])
print(f"MR: {np.mean(rank)}, MRR: {np.mean(1.0 / rank)}, Hits@10: {np.sum(rank <= 10)/len(rank)}\n") 

```

