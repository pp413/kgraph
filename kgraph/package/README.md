# KGraph

知识图谱是一种具有多种关系的异构有向网络。在已公开的知识图谱中，收集到的知识通常被定义为事件，以三元组的形式进行存储。每个三元组通常被定义为$(h,r,t)$，其中$h$表示头实体，$t$表示尾实体，$r$表示由头实体到尾实体的链路关系。虽然随着人们日常活动产生了大量的知识，但是收集并存储这些海量的知识事件需要花费巨大的人力和时间成本。因此即便是规模超过亿万级的知识图谱仍然存在着大量的缺失事件。因此利用已有的知识，学习到合理高效的知识补全模型，成为当前知识图谱研究中的一个热点。

KGraph旨在帮助研究者快速构建基于数值表示的知识补全模型的工具库。KGraph具有丰富的API，其中包含了当前学术界使用到的主流开源知识图谱数据集，例如FB15k、FB15k-237、WN18、WN18RR、FB13、WN11等，以及高效地评估方法。利用KGraph，知识图谱研究者可以更加专注于对模型本身的研究，而不需要关注繁琐的数据加载、预处理以及评估方法的编写。

可以通过pip进行安装。具体安装命令：pip install kgraph