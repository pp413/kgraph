# kgraph
A Python library for Graph Embedding on knowledge graphs. Most of the code in this library from an Open-source Framework  OpenKE.

[OpenKE-PyTorch]: https://github.com/thunlp/OpenKE	"openke"

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

##### datasets
kgraph.datasets 主要解决数据的**预处理**和**加载**。在该模块可以加载常见的用于知识表示研究中的数据集。



```python
from kgraph.datasets import *
```



### WN18

|          |  Train  | Valid | Test  | Entities | Relations |
| :------: | :-----: | :---: | :---: | :------: | :-------: |
| Original | 141,442 | 5,000 | 5,000 |  40,943  |    18     |
| Cleaned  |         |       |       |          |           |



```python
data, num_ent, num_rel = load_wn18(clean_unseen=True)
train_data = data['train']
valid_data = data['valid']
test_data = data['test']
```



### WN18RR

|          | Train  | Valid | Test  | Entities | Relations |
| :------: | :----: | :---: | :---: | :------: | :-------: |
| Original | 86,835 | 3,034 | 3,134 |  40,943  |    11     |
| Cleaned  |        |       |       |          |           |



```python
data, num_ent, num_rel = load_wn18rr(clean_unseen=True)
train_data = data['train']
valid_data = data['valid']
test_data = data['test']
```



### FB15k

|          |  Train  | Valid  |  Test  | Entities | Relations |
| :------: | :-----: | :----: | :----: | :------: | :-------: |
| Original | 483,142 | 50,000 | 59,071 |  14,951  |   1,345   |
| Cleaned  |         |        |        |          |           |



```python
data, num_ent, num_rel = load_fb15(clean_unseen=True)
train_data = data['train']
valid_data = data['valid']
test_data = data['test']
```



### FB15k-237

|          |  Train  | Valid  |  Test  | Entities | Relations |
| :------: | :-----: | :----: | :----: | :------: | :-------: |
| Original | 272,115 | 17,535 | 20,466 |  14,541  |    237    |
| Cleaned  | 272,115 | 17,516 | 20,438 |  14,505  |    237    |



```python
data, num_ent, num_rel = load_fb15k237(clean_unseen=True)
train_data = data['train']
valid_data = data['valid']
test_data = data['test']
```



### YAGO3-10

|          |   Train   | Valid | Test  | Entities | Relations |
| :------: | :-------: | :---: | :---: | :------: | :-------: |
| Original | 1,079,040 | 5,000 | 5,000 | 123,182  |    37     |
| Cleaned  |           |       |       |          |           |



```python
data, num_ent, num_rel = load_yago3_10(clean_unseen=True)
train_data = data['train']
valid_data = data['valid']
test_data = data['test']
```



### Wordnet11

|          | Train  | Valid Pos | Valid Neg | Test Pos | Test Neg | Entities | Relations |
| :------: | :----: | :-------: | :-------: | :------: | :------: | :------: | :-------: |
| Original | 110361 |   2606    |   2609    |  10493   |  10542   |  38588   |    11     |
| Cleaned  |        |           |           |          |          |          |           |



```python
data, num_ent, num_rel = load_wn11(clean_unseen=True)
train_data = data['train']
valid_data = data['valid']
test_data = data['test']
```



### Freebase13

|          | Train  | Valid Pos | Valid Neg | Test Pos | Test Neg | Entities | Relations |
| :------: | :----: | :-------: | :-------: | :------: | :------: | :------: | :-------: |
| Original | 316232 |   5908    |   5908    |  23733   |  23731   |  75043   |    13     |
| Cleaned  |        |           |           |          |          |          |           |



```python
data, num_ent, num_rel = load_fb13(clean_unseen=True)
train_data = data['train']
valid_data = data['valid']
test_data = data['test']
```



## utils 模块

kgraph.utils 模块主要解决模型的测试功能。值得注意的是模块也提供部分表示模型类型在训练过程中的batch数据生成过程。



### BaseEval 类

BaseEval类是一个基类。


### TrainEval_For_Trans
TrainEval_For_Trans类是基于BaseEval的子类

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_fb15k237
from kgraph.utils import TrainEval_For_Trans

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data, num_ent, num_rel = load_fb15k237()

class TransE(nn.Module):
    def __init__(self, num_ent: int, num_rel: int, dim: int, margin_value: float=5.):
        super(TransE, self).__init__()
        
        self.margin = margin_value
        self.k = int(num_ent)
        
        self.ent_embeddings = nn.Embedding(num_ent, dim)
        self.rel_embeddings = nn.Embedding(num_rel, dim)
        
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        
    def embed_lookup(self, data):
        # print(torch.max(data, dim=0))
        head = self.ent_embeddings(data[:, 0])
        rel = self.rel_embeddings(data[:, 1])
        tail = self.ent_embeddings(data[:, 2])
        return head, rel, tail
    
    def regul(self, data):
        head, rel, tail = self.embed_lookup(data)
        reg = (torch.mean(head ** 2) +
               torch.mean(rel ** 2) +
               torch.mean(tail ** 2)) / 3.
        
        return reg
    
    def forward(self, pos_samples, neg_samples):
        pos_distance = self.score(pos_samples)
        neg_distance = self.score(neg_samples)
        
        return F.relu(self.margin + pos_distance - neg_distance).mean()
    
    def score(self, samples):
        head, rel, tail = self.embed_lookup(samples)
        head = F.normalize(head, 2, -1)
        rel = F.normalize(rel, 2, -1)
        tail = F.normalize(tail, 2, -1)
        
        return torch.norm(head + rel - tail, p=1, dim=-1).flatten()

def loss_function(model):
    def f(pos_data, neg_data):
        model.train()
          if isinstance(data, np.ndarray):
              data = torch.from_numpy(data).to(device)
          loss = model(pos_data, neg_data) + 0.005 * (self.regul(pos_data) + self.regul(neg_data))
          return loss
    return f

def eva_function(model):
    def f(data):
        model.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).to(device)
            score = model.score(data)
            return -score
    return f

model = TransE(num_ent, num_rel, dim=200).to(device)
process = TrainEval_For_Trans(data=data, num_ent=num_ent, num_rel=num_rel, lr=0.001, model=model,
                      opt=optim.Adam, batch_size=10000, loss_function=loss_function, device=device)

process.fit(num_epoch=1000, scheduler_step=100, valid_predict=eva_function)
torch.save(model.state_dict(), 'transe.ckpt')
model.load_state_dict(torch.load('transe.ckpt'))
process.eval(eva_function, batch_size=10000)

```


### TrainEval_By_Triplet

TrainEval_By_Triplet类是基于BaseEval的子类，完成了negative_sample和sample_iter两个函数，在训练过程中每batch中的数据:
$$
batchData \in \mathbb{R}^{batchSize\times 3} \\
batchLabel \in \mathbb{R}^{batchSize}.
$$


需要注意的是在标签中正样本的标签是1，负样本的标签是-1.