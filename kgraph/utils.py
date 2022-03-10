import os
import tarfile
import zipfile
import prettytable as pt

from torch.hub import download_url_to_file
from ._utils.tools import generateTripleIdFile

os.environ['KG_DIR'] = os.path.join(os.path.expanduser('~/Documents'), 'KGDataSets')

def download_data(url, save_as_file=None):
    if os.path.exists(save_as_file):
        return
    else:
        try:
            download_url_to_file(url, save_as_file)
        except:
            print("download failed")


def unzip(from_file, unzip_to_folder):
    if from_file.endswith('.gz') or from_file.endswith('.tar') or from_file.endswith('.tgz'):
        archive = tarfile.open(from_file, 'r')
    elif from_file.endswith('.zip'):
        archive = zipfile.ZipFile(from_file, 'r')
    else:
        raise Exception('Unrecognized file type: ' + from_file)
    print('Extracting file to {}'.format(unzip_to_folder))
    archive.extractall(path=unzip_to_folder)
    archive.close()

def get_statistics(data_path):
    files = [
        'train2id.txt',
        'valid2id.txt',
        'test2id.txt',
        'entity2id.txt',
        'relation2id.txt'
    ]
    
    statistics = []
    for filename in files:
        with open(os.path.join(data_path, filename), 'r') as f:
            statistics.append(int(f.readline()))
    return statistics

def log_statistics(statistics, data_name, path=None):
    
    tb = pt.PrettyTable(header=True)
    tb.title = f'The statistics of benchmark datasets.'
    tb.field_names = ['DataName', 'TrainSet', 'ValidSet', 'TestSet', 'Entities', 'Relations']
    tb.add_row([data_name,] + statistics[:5])
    
    print(tb)
    if not os.path.exists(path):
        print("Write statistics of data error!")
        return
    if os.path.exists(os.path.join(path, 'statistics.txt')):
        tb = tb.get_string()
        with open(os.path.join(path, 'statistics.txt'), 'w') as f:
            f.write(tb)

def load_data(url, path, sep='\t', no_sort=True):
    data_zip_name = url.split('/')[-1]

    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))

    if not os.path.exists(os.path.join(path, data_zip_name.split('.')[0])):
        if not os.path.exists(os.path.join(path, data_zip_name)):
            download_data(url, os.path.join(path, data_zip_name))
        unzip(os.path.join(path, data_zip_name), os.path.join(path))
    if not os.path.exists(os.path.join(path, data_zip_name.split('.')[0], 'train2id.txt')):
        generateTripleIdFile(os.path.join(path, data_zip_name.split('.')[0]),
                             os.path.join(path, data_zip_name.split('.')[0]),
                             sep=sep, no_sort=no_sort)
    data_name = data_zip_name.split('.')[0]
    
    path = os.path.join(path, data_name)
    statistics = get_statistics(path)
    log_statistics(statistics, data_name, path)
    
    return path
