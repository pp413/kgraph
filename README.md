# kgraph
A Python library for Graph Embedding on knowledge graphs


kgraph 是一个知识图谱在知识表示研究上的 ，主要解决常见数据的预处理、加载以及常用的评估。

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



### TrainEval_By_Triplet

TrainEval_By_Triplet类是基于BaseEval的子类，完成了negative_sample和sample_iter两个函数，在训练过程中每batch中的数据:
$$
batchData \in \mathbb{R}^{batchSize\times 3} \\
batchLabel \in \mathbb{R}^{batchSize}.
$$

```python
from kgraph.utils import TrainEval_By_Triplet

trainEval = TrainEval_By_Triplet(num_ent, num_rel, data)

for batch_data, batch_labels in trainEval.sample_iter(batch_size=1000, negative_rate=1):
    batch_data = torch.from_numpy(batch_data).to(device)
    batch_labels = torch.from_numpy(batch_labels).to(device)


train.eval(predict_function_name, test_data, batch_size=512, flags='original',
           filename='conf.txt', device='cpu')
```

需要注意的是在标签中正样本的标签是1，负样本的标签是-1.