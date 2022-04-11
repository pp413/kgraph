## kgraph

kgraph 是一个knowledge graph embedding 库。



### Knowledge Graph

知识图谱是以图结构的数据库，通常使用$G=(V, R, E)$的结构表示，其中$V$表示知识图谱中的实体集合，$R$表示知识图谱中的关系集合，$E$表示在知识图谱中以三元组形式表示的边集合。因此，对于知识图谱$G$，每一个实体（头实体$h$或尾实体$t$）$h,t\in V$，每一种关系$r\in R$，每一条link $(h, r, t)\in E$。



A knowledge graph contains a set of entities and relations between entities. The set of facts in knowledge graph are represented in the form of triples, where are referred to as the head (or subject) and tail (or object) entities, and is referred to as the relationship (or predicate).



The problem of Knowledge graph embedding is in finding a function that learning the embeddings of triples using low dimensional vectors such that it preserve structural information, . To accomplish this, the general principle is enforce the learning of entities and relationships to be compatible with the information in. The representation choices include deterministic point, multivariate Gaussian distribution, or complex number. Under the Open Word Assumption, a set of unseen negative triples, , are sampled from positive triples by either corrupting the head or tail entity. Then, a scoring function, is defined to reward the positive triples and penalize the negative triples. Finally, an optimization algorithm is used to minimize or maximize the scoring function.



The algorithm is often evaluated in terms of their capability of predicting the missing entities in negative triples or , or predicting whether an unseen fact is true or not. The evaluation metrics include the rank of the answer in the predicted list (mean rank), and the ratio of answers ranked top-k in the list (hit-k ratio).



### Dataset

基类

Data(path, no_sort=True)



FB15k(path, no_sort=True)



FB15k237(path, no_sort=True)



WN18(path, no_sort=True)



WN18RR(path, no_sort=True)



### DataIter

训练batch数据生成器

DataIter(num_ent, num_rel, batch_size, num_threads, smooth_lambda, num_neg, mode, bern_flag, seed, element_type)



### Predict

测试器

Predict(element_type)



### log

日志文件记录





