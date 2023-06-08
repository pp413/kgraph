import os, sys
import json
import numpy as np
import logging, logging.config
import prettytable as pt
from pprint import pprint
# from dvclive import Live
from typing import Optional

from .utils import get_results_from_rank

__all__ = ['set_logger', 'get_result_table', 'log_N2N']

# Create a logger object
def set_logger(name, log_dir='./log'):
    """"""
    # config_dict = json.load(open(config_dir))
    config_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - [%(name)s] - %(message)s"
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
def get_result_table(rhs_ranks, rhs_franks, lhs_ranks, lhs_franks, data_name='benchmark', flags='Valid'):
    """
    rhs: predict the tails of triples
    lhs: predict the heads of triples
    """
    
    ranks = {
        'rhs_original': get_results_from_rank(rhs_ranks),
        'lhs_original': get_results_from_rank(lhs_ranks),
        'avg_original': dict(),
        
        'rhs_filtered': get_results_from_rank(rhs_franks),
        'lhs_filtered': get_results_from_rank(lhs_franks),
        'avg_filtered': dict(),
    }
    
    for key, value in ranks['rhs_original'].items():
        ranks['avg_original'][key] = round((value + ranks['lhs_original'][key]) * 1. / 2, 5)
    
    for key, value in ranks['rhs_filtered'].items():
        ranks['avg_filtered'][key] = round((value + ranks['lhs_filtered'][key]) * 1. / 2, 5)
    
    final_results = {}
    
    tb = pt.PrettyTable(header=True)
    tb.title = 'The results of the {} datasets on the {} Process.'.format(data_name, flags)
    tb.field_names = ['Categories', 'MR', 'MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    for name, results in ranks.items():
        tb.add_row([name, results['mr'], results['mrr'], results['hits@1'], results['hits@3'], results['hits@10']])
        final_results[name] = results
        if name == 'avg_original':
            tb.add_row(['', '', '', '', '', ''])
    return tb.get_string(), final_results


def log_N2N(results, data_name='benchmark'):
    
    def cal_results_of_per(rhs_ranks, rhs_franks, lhs_ranks, lhs_franks):
        """
        rhs: predict the tails of triples
        lhs: predict the heads of triples
        """
        ranks = {
            'predict_tail': get_results_from_rank(rhs_franks),
            'predict_head': get_results_from_rank(lhs_franks),
        }
        
        for name, rank in ranks.items():
            ranks[name] = [rank['mrr'], rank['hits@1'], rank['hits@3'], rank['hits@10']]
        return ranks
    
    def cal_results(tmp_results):
        final_results = {key: {} for key in ['1to1', '1toN', 'Nto1', 'NtoN']}
        
        for key, value in tmp_results.items():
            ranks = cal_results_of_per(*value)
            for name, rank in ranks.items():
                final_results[key][name] = rank
        
        tab_rows = []
        for name in ['predict_head', 'predict_tail']:
            for key in ['1to1', '1toN', 'Nto1', 'NtoN']:
                if name == 'predict_head':
                    tab_row_name = name if key == 'Nto1' else ''
                else:
                    tab_row_name = name if key == '1toN' else ''
                tab_rows.append([tab_row_name, key, ] + final_results[key][name])
            tab_rows.append(['', '', '', '', '', ''])

        return tab_rows[:-1]
    
    tb = pt.PrettyTable(header=True)
    tb.title = 'Results on link prediction by relation category on {} dataset'.format(data_name)
    tb.field_names = ['', 'Categories', 'MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    
    for row in cal_results(results):
        tb.add_row(row)
    
    return tb.get_string()
    


