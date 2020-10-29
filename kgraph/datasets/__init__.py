"""Helper functions to load knowledge graphs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .read import load_fb15k, load_fb15k237, load_from_csv, low_name
from .read import load_wn11, load_wn18, load_wn18rr, load_yago3_10
from .read import load_all_datasets, load_fb13

from .TestDataLoader import TestDataLoader
from .TrainDataLoader import TrainDataLoader
from .PyTorchTrainDataLoader import PyTorchTrainDataset
from .PyTorchTrainDataLoader import PyTorchTrainDataLoader

__all__ = ['load_fb15k', 'load_fb15k237', 'load_from_csv', 'low_name',
           'load_fb13', 'load_all_datasets', 'load_wn11', 'load_wn18',
           'load_wn18rr', 'load_yago3_10', 'load_wn18',
           'TrainDataLoader', 'TestDataLoader', 'PyTorchTrainDataset',
           'PyTorchTrainDataLoader']
