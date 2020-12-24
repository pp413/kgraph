#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/26 17:52:27
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#

# data
from .data import KG_BENCHMARK_DATASETS
from .data import load_fb15k, load_fb15k237
from .data import load_wn18, load_wn18rr
from .data import load_all_datasets

# benchmark
from .data import DataBase
from .data import FB15k, FB15k237
from .data import WN18, WN18RR

# dataloader
from .data_loader import DataLoader
