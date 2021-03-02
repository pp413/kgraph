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
from kgraph.data import *
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



### Benchmark DataSets 基数据

```
from kgraph.data import FB15k237, FB15k, WN18, WN18RR
```



## utils 模块

kgraph.utils 模块主要解决数据的抽样，迭代器的生成。



### DataIter 数据迭代器

DataIter(dataset, batch_size, batch_sampler=None, shuffle=True, neg_ratio=None, num_workers=0, use_selecting_src_rate=True, flags='train', device='cpu')



例子

```python
from kgraph import Sampler
from kgraph import DataIter
from kgraph.data import FB15k237

dataset = FB15k237()
sampler = Sampler(invalid_valid_ratio=10)
data_iter = DataIter(dataset, batch_size=2000, batch_sampler=sampler, shuffle=True)

```



### Sampler 抽样基类。

```python
Sampler(invalid_valid_ratio=1)
```

invalid_valid_ratio: the ratio of generating negative samples.



例子

```python
from kgraph import Sampler

sampler = Sampler(invalid_valid_ratio=1)

lhs_pos, rhs_pos, lhs_neg, rhs_neg = sampler(batch_data, batch_size)
```







## Kgraph 应用


### @initial_graph_model(data_iter)
initial_graph_model 是一个模型的初始化魔法函数，为模型添加了基本的训练和测试函数。具体的函数如下：

```python
link_prediction(batch_size=None, for_test=True, constraint=None)
```

batch_size: the batch size of data on testing,  default: None (the number of entities).

for_test: use the 'test' set. if not, use the 'valid' set.

constraint: use constraint. 



```python
link_n2n_prediction(batch_size=None, constraint=None)
```

batch_size: the batch size of data on testing,  default: None (the number of entities).

constraint: use constraint. 



```python
fit(num_epoch, opt='adam', lr=1e-4, weight_decay=0, scheduler_step=None, scheduler_gamme=0.75, valid_step=None, valid_function=None, device='cpu')
```

num_epoch: the number of epochs on training.

opt: 

lr: the learning rate.



```python
pred_train_from(path)
```

path: the save path of the model which has been pred_trained.



```python
load_checkpoint(path)
```

path: the save path of the model.



```python
save_checkpoint(path)
```

path: the path to save the model.



#### 例子

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from kgraph import initial_graph_model
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

```

