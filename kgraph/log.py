import os, sys
import json
import numpy as np
import logging, logging.config
import prettytable as pt
from pprint import pprint

from .utils.__utils import get_results_from_rank

__all__ = ['set_logger', 'get_result_table']

# Create a logger object
def set_logger(name, log_dir='./log'):
    """"""
    # config_dict = json.load(open(config_dir))
    config_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
        }
    },

    "handlers": {
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "python_logging.log",
            "encoding": "utf8"
        }
    },

    "root": {
        "level": "DEBUG",
        "handlers": ["file_handler"]
    }}
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    config_dict['handlers']['file_handler']['filename'] = os.path.join(log_dir, name.replace('/', '_').replace(':', '|')+'.log')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)
    
    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)
    
    return logger

#
def get_result_table(rhs_ranks, rhs_franks, lhs_ranks, lhs_franks, data_name='benchmark'):
    """
    rhs: predict the tails of triples
    lhs: predict the heads of triples
    """
    
    ranks = {
        'rhs_original': rhs_ranks,
        'lhs_original': lhs_ranks,
        'avg_original': np.append(rhs_ranks, lhs_ranks),
        
        'rhs_filtered': rhs_franks,
        'lhs_filtered': lhs_franks,
        'avg_filtered': np.append(rhs_franks, lhs_franks),
    }
    
    final_results = {}
    
    tb = pt.PrettyTable(header=True)
    tb.title = 'The results of the {} datasets.'.format(data_name)
    tb.field_names = ['Category', 'MR', 'MRR', 'Hits@1', 'Hits@3', 'Hits@5', 'Hits@10']
    for name, rank in ranks.items():
        results = get_results_from_rank(rank)
        tb.add_row([name, results['mr'], results['mrr'], results['hits@1'], results['hits@3'], results['hits@5'], results['hits@10']])
        final_results[name] = results
        if name == 'avg_original':
            tb.add_row(['', '', '', '', '', '', ''])
    return tb.get_string(), final_results




