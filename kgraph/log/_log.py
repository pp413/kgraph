#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2021/02/27 20:01:39
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#
import os, arrow

#
def log_pred(table, data_name='benchmark', model_name='model', epoch_i=None):
    
    dir_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    _now = arrow.now().format('YYYY-MM-DD')
    name = f'{_now}_{model_name}_{chr(960)}_{data_name}.txt'
    file_path = os.path.join(dir_path, name)
    
    if epoch_i is None:
        now = arrow.now().format('YYYY-MM-DD HH:mm')
        title = f'\t\t\t   The results of ranks on {data_name} {now}'
    else:
        now = arrow.now().format('YYYY-MM-DD HH:mm:ss')
        title = f'\t\t\t   The results of ranks on {data_name} at Epoch {epoch_i} {now}'
    
    with open(file_path, 'a') as f:
        print(title)
        print(table)
        print(f'The results are writing in this file {file_path}')
        f.write(title+'\n')
        f.write(table)
        f.write('\n')
        print()

    
    
    
