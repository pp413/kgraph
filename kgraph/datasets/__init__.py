"""Helper functions to load knowledge graphs."""
from .read import load_fb15k, load_fb15k237, load_from_csv, low_name
from .read import load_wn11, load_wn18, load_wn18rr, load_yago3_10
from .read import load_all_datasets
from .protocol import sample_with_neg_and_label, sample_with_label
from .protocol import get_test_with_label

__all__ = ['load_fb15k', 'load_fb15k237', 'load_from_csv', 'low_name',
           'sample_with_neg_and_label', 'sample_with_label',
           'get_test_with_label', 'load_all_datasets',
           'load_wn11', 'load_wn18', 'load_wn18rr', 'load_yago3_10']
