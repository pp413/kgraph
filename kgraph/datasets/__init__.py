"""Helper functions to load knowledge graphs."""
from .read import load_fb15k, load_fb15k237, load_from_csv, low_name
from .read import load_wn11, load_wn18, load_wn18rr, load_yago3_10
from .read import load_all_datasets, load_fb13

__all__ = ['load_fb15k', 'load_fb15k237', 'load_from_csv', 'low_name',
           'load_fb13', 'load_all_datasets', 'load_wn11', 'load_wn18',
           'load_wn18rr', 'load_yago3_10', 'load_wn18']
