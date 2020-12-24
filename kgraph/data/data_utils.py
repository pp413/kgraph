#!/user/bin/python
# -*- coding: utf-8 -*-
#
# @ Author: Yao Shuang-Long
# @ Date: 2020/11/26 17:52:19
# @ Summary: the summary.
# @ Contact: xxxxxxxx@email.com
# @ Paper Link: 
#

import os
import csv
import hashlib
import zipfile
import tarfile
import pandas as pd
import numpy as np
import prettytable as pt
from collections import Counter

try:
    import requests
except ImportError:
    class requests_failed_to_import(object):
        pass
    requests = requests_failed_to_import

os.environ['KG_DIR'] = os.path.join(os.path.expanduser('~'), '.KGDataSets')

def get_download_dir():
    """
    return the Path to the download directory.
    """
    
    default_dir = os.path.join(os.path.expanduser('~'), '.KGDataSets')
    dirname = os.environ.get('KG_DIR', default_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def set_download_dir(path=None):
    if path is None and not os.path.exists(os.environ['KG_DIR']):
        os.makedirs(os.environ['KG_DIR'])
    
    if path is not None and not os.path.exists(path):
        print(f'Set the root path is {path}')
        os.makedirs(path)
        os.environ['KG_DIR'] = path
    return os.environ['KG_DIR']
    

def get_from_dgl_url(data_name):
    '''
    Get dataset from DGL online url for download.
    '''
    dgl_repo_url = 'https://data.dgl.ai/dataset'
    repo_url = os.environ.get('KG_BENCHMARK_DGL_URL', dgl_repo_url)
    if repo_url[-1] != '/':
        repo_url += '/'
    data_name = 'wn18' if data_name.lower() == 'wn18' else data_name
    return repo_url + '{}.tgz'.format(data_name)

def get_from_aigraph_url(data_name):
    '''Get dataset from AIGraph online url for download.'''
    data_name = data_name.lower()
    data_name = 'wn18RR' if data_name == 'wn18rr' else data_name
    
    url = 'https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/{}.zip'.format(data_name)

    return url

def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Codes borrowed from mxnet/gluon/utils.py

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    if sha1.hexdigest() != sha1_hash:
        print(sha1.hexdigest())
    return sha1.hexdigest() == sha1_hash


def download(url, path=None, overwrite=False, retries=5, verify_ssl=True, log=True):
    """Download a given URL.

    Codes borrowed from mxnet/gluon/utils.py

    Parameters
    ----------
    url : str
        URL to download.
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with the same name as in url.
    overwrite : bool, optional
        Whether to overwrite the destination file if it already exists.
    retries : integer, default 5
        The number of times to attempt downloading in case of failure or non 200 return codes.
    verify_ssl : bool, default True
        Verify SSL certificates.
    log : bool, default True
        Whether to print the progress for download

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. ' \
            'Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        print(f'The benchmark datasets in this dir: {path}')
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries+1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                if log:
                    print('Downloading %s from %s...' % (fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s" % url)
                with open(fname, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    if log:
                        print("download failed, retrying, {} attempt{} left"
                              .format(retries, 's' if retries > 1 else ''))
        
    return fname


def extract_archive(file, target_dir):
    """Extract archive file.

    Parameters
    ----------
    file : str
        Absolute path of the archive file.
    target_dir : str
        Target directory of the archive to be uncompressed.
    """
    if os.path.exists(target_dir):
        return
    if file.endswith('.gz') or file.endswith('.tar') or file.endswith('.tgz'):
        archive = tarfile.open(file, 'r')
    elif file.endswith('.zip'):
        archive = zipfile.ZipFile(file, 'r')
    else:
        raise Exception('Unrecognized file type: ' + file)
    print('Extracting file to {}'.format(target_dir))
    archive.extractall(path=target_dir)
    archive.close()

def _triplets_as_list(data, entity_dict, relation_dict):
    l = []
    for triplet in data:
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l

def load_from_text(file_path, sep='\t', header=None, dtype=str):

    df = pd.read_csv(file_path, sep=sep, header=header,
                     names=None, dtype=dtype)
    df = df.drop_duplicates()
    return df.values


def load_from_csv(file_path, sep='\t', header=None):
    df = pd.read_csv(file_path, sep=sep, header=header,
                     names=None, dtype=str)
    df = df.drop_duplicates()
    return df.values

def write_to_csv(data, data_name, file_name, sep='\t', header=None):
    dir = os.path.join(os.environ.get('KG_DIR', os.getcwd()), data_name, file_name)
    with open(dir, 'w') as f:
        w = csv.writer(f)
        if header is not None:
            w.writerow(header)
        w.writerows(data)
            
def write_dataset(data_id, data_name , name='train2id.txt'):
    f = open(os.path.join(os.environ.get('KG_DIR'), data_name, name), 'w')
    f.write("%d\n"%(len(data_id)))
    for tripe in data_id:
        h, r, t = tripe
        h, r, t = int(h), int(r), int(t)
        f.write(f"{h}\t{r}\t{t}\n")
    f.close()

def write_str2id(data_dict, data_name, name='entity2id.txt'):
    f = open(os.path.join(os.environ.get('KG_DIR'), data_name, name), 'w')
    f.write("%d\n"%(len(data_dict)))
    for i in data_dict:
        f.write("%s\t%d\n"%(i, data_dict[i]))
    f.close()        

def load_and_check_original_data(fdir, data_name, data_sha1):
    check_train = os.path.join(fdir, data_name, 'train.txt')
    check_test = os.path.join(fdir, data_name, 'test.txt')
    check_valid = os.path.join(fdir, data_name, 'valid.txt')

    if not check_sha1(check_train, data_sha1['train']):
        print('The train set is error!')

    elif not check_sha1(check_test, data_sha1['test']):
        print('The test set is error!')
    elif not check_sha1(check_valid, data_sha1['valid']):
        print('The valid set is error!')
    else:
        train = load_from_text(check_train)
        valid = load_from_text(check_valid)
        test = load_from_text(check_test)
        
        total_train = len(train)
        total_valid = len(valid)
        total_test = len(test)

        return {'train': train, 'valid': valid, 'test': test, 
                'total_train': total_train,
                'total_valid': total_valid,
                'total_test': total_test,
                'name': data_name}

def clean_data(X):

    train = pd.DataFrame(X['train'], columns=['s', 'p', 'o'])
    valid = pd.DataFrame(X['valid'], columns=['s', 'p', 'o'])
    test = pd.DataFrame(X['test'], columns=['s', 'p', 'o'])

    train_ent = np.unique(np.concatenate((train.s, train.o)))
    train_rel = train.p.unique()
    
    all_ent = np.unique(np.concatenate((train.s, train.o, valid.s, valid.o, test.s, test.o)))
    all_rel = np.unique(np.concatenate((train.p, valid.p, test.p)))

    valid_idx = valid.s.isin(train_ent) & valid.o.isin(train_ent) & valid.p.isin(train_rel)
    test_idx = test.s.isin(train_ent) & test.o.isin(train_ent) & test.p.isin(train_rel)

    filtered_valid = valid[valid_idx].values
    filtered_test = test[test_idx].values

    # filtered_X = {'train': train.values, 'valid': filtered_valid, 'test': filtered_test,
    #               'name': X['name']}
    
    X['train'] = train.values
    X['valid'] = filtered_valid
    X['test'] = filtered_test
    X['total_entities'] = len(all_ent)
    X['total_relations'] = len(all_rel)

    return X


def str_to_idx(data):
    ent2id_dir = os.path.join(os.environ.get('KG_DIR'), data['name'], 'entity2id.txt')
    rel2id_dir = os.path.join(os.environ.get('KG_DIR'), data['name'], 'relation2id.txt')
    if os.path.exists(ent2id_dir) and os.path.exists(ent2id_dir):
        entities_idx_dict = {}
        relations_idx_dict = {}
        with open(ent2id_dir, 'r') as f:
            total_entities = (int)(f.readline().strip())
            for i in range(total_entities):
                line = f.readline()
                entity, ent_id = line.strip().split('\t')
                ent_id = int(ent_id)
                entities_idx_dict[entity] = ent_id
        
        with open(rel2id_dir, 'r') as f:
            total_relations = (int)(f.readline().strip())
            for i in range(total_relations):
                line = f.readline()
                relation, rel_id = line.strip().split('\t')
                rel_id = int(rel_id)
                relations_idx_dict[relation] = rel_id
    else:
        all_data = np.concatenate((data['train'], data['valid'], data['test']), 0)

        entities = Counter(np.concatenate((all_data[:, 0], all_data[:, 2]), 0)).most_common()
        relations = Counter(all_data[:, 1]).most_common()
        total_entities = len(entities)
        total_relations = len(relations)
        
        entities_idx_dict = {ent[0]: value for value, ent in enumerate(sorted(
            entities, key=lambda x: x[1], reverse=True
        ))}

        relations_idx_dict = {rel[0]: value for value, rel in enumerate(sorted(
            relations, key=lambda x: x[1], reverse=True
        ))}
    
    train2id = np.asarray(_triplets_as_list(data['train'], entities_idx_dict, relations_idx_dict))
    valid2id = np.asarray(_triplets_as_list(data['valid'], entities_idx_dict, relations_idx_dict))
    test2id = np.asarray(_triplets_as_list(data['test'], entities_idx_dict, relations_idx_dict))

    if not os.path.exists(os.path.join(os.environ.get('KG_DIR'), data['name'], 'constrain.txt')):
        write_to_csv(train2id, data['name'], file_name='train.csv')
        write_to_csv(valid2id, data['name'], file_name='valid.csv')
        write_to_csv(test2id, data['name'], file_name='test.csv')
        write_str2id(entities_idx_dict, data['name'], name='entity2id.txt')
        write_str2id(relations_idx_dict, data['name'], name='relation2id.txt')
        
        lef = {}    # (h, r) --> [t, ...]
        rig = {}    # (r, t) --> [h, ...]
        rel_lef = {}     # r --> {h: number}
        rel_rig = {}     # r --> {t: number}
        
        all_data = np.concatenate((train2id, valid2id, test2id), axis=0)
        for triplet in all_data:
            h, r, t = triplet
            if not (h, r) in lef:
                lef[(h, r)] = []
            if not (r, t) in rig:
                rig[(r, t)] = []
            lef[(h, r)] += [t]
            rig[(r, t)] += [h]
            if not r in rel_lef:
                rel_lef[r] = {}
            if not r in rel_rig:
                rel_rig[r] = {}
            rel_lef[r][h] = 1
            rel_rig[r][t] = 1
        
        # constrain file
        # rel_id_0     len(rel_lef[0])    head_id_0    head_id_1    ...
        # rel_id_0     len(rel_rig[0])    tail_id_0    tail_id_1    ...
        # rel_id_i     len(rel_lef[i])    head_id_0    head_id_1    ...
        # rel_id_i     len(rel_rig[i])    tail_id_0    tail_id_1    ...
        f = open(os.path.join(os.environ.get('KG_DIR'), data['name'], 'constraint.txt'), 'w')
        
        for i in rel_lef:
            f.write("%d\t%d"%(i,len(rel_lef[i])))
            for j in rel_lef[i]:
                f.write("\t%d"%(j))
            f.write("\n")
            f.write("%d\t%d"%(i,len(rel_rig[i])))
            for j in rel_rig[i]:
                f.write("\t%d"%(j))
            f.write("\n")
        f.close()
        
        rel_lef = {}     # r --> the out degree
        tot_lef = {}     # r --> total the number of (X, r) in the lef key.
        rel_rig = {}     # r --> the in degree
        tot_rig = {}     # r --> total the number of (r, X) in the rig key.
        # lef: {(h, r): t}
        # rig: {(r, t): h}
        for i in lef:
            if not i[1] in rel_lef:
                rel_lef[i[1]] = 0
                tot_lef[i[1]] = 0
            rel_lef[i[1]] += len(lef[i])
            tot_lef[i[1]] += 1.0

        for i in rig:
            if not i[0] in rel_rig:
                rel_rig[i[0]] = 0
                tot_rig[i[0]] = 0
            rel_rig[i[0]] += len(rig[i])
            tot_rig[i[0]] += 1.0

        # s11=0
        # s1n=0
        # sn1=0
        # snn=0
        # for triple in test2id:
        #     h, r, t = triple
        #     rig_n = rel_lef[r] / tot_lef[r]
        #     lef_n = rel_rig[r] / tot_rig[r]
        #     if (rig_n < 1.5 and lef_n < 1.5):
        #         s11+=1
        #     if (rig_n >= 1.5 and lef_n < 1.5):
        #         s1n+=1
        #     if (rig_n < 1.5 and lef_n >= 1.5):
        #         sn1+=1
        #     if (rig_n >= 1.5 and lef_n >= 1.5):
        #         snn+=1
        # # f.close()
        # print(s11)


        f11 = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "1-1.txt"), "w")
        f1n = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "1-n.txt"), "w")
        fn1 = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "n-1.txt"), "w")
        fnn = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "n-n.txt"), "w")
        fall = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "test2id_all.txt"), "w")
        for triple in test2id:
            h, r, t = triple
            content = f'{h}\t{r}\t{t}\n'
            rig_n = rel_lef[r] / tot_lef[r]
            lef_n = rel_rig[r] / tot_rig[r]
            if (rig_n < 1.5 and lef_n < 1.5):
                f11.write(content)
                fall.write("0"+"\t"+content)
            if (rig_n >= 1.5 and lef_n < 1.5):
                f1n.write(content)
                fall.write("1"+"\t"+content)
            if (rig_n < 1.5 and lef_n >= 1.5):
                fn1.write(content)
                fall.write("2"+"\t"+content)
            if (rig_n >= 1.5 and lef_n >= 1.5):
                fnn.write(content)
                fall.write("3"+"\t"+content)
        fall.close()
        f11.close()
        f1n.close()
        fn1.close()
        fnn.close()

    return {
        'train': train2id,
        'valid': valid2id,
        'test': test2id,
        'ent_dict': entities_idx_dict,
        'rel_dict': relations_idx_dict,
        'total_train': data['total_train'],
        'total_valid': data['total_valid'],
        'total_test': data['total_test'],
        'name': data['name']
    }, {
        'train': data['train'],
        'valid': data['valid'],
        'test': data['test'],
        'total_train': data['total_train'],
        'total_valid': data['total_valid'],
        'total_test': data['total_test'],
        'name': data['name']
        }, data['total_entities'], data['total_relations']


def load_table(dir):
    lines = []
    DataNames = set()
    with open(dir, 'r') as f:
        for line in f.readlines():
            if '-+-' in line or 'DataName' in line:
                continue
            line = line.strip().replace(' ', '').split('|')[1:-1]
            if line[0] not in DataNames:
                lines.append(line)
                DataNames.add(line[0])
    return lines, DataNames

def pprint(data, total_entities, total_relations, dataName=''):
    
    train_len = data['total_train']
    valid_len = data['total_valid']
    test_len = data['total_test']
    
    filename = os.path.join(os.environ['KG_DIR'], 'statistics.txt')
    row_data, data_names = load_table(filename) if os.path.exists(filename) else (None, [])
        
    tb = pt.PrettyTable()
    tb.field_names = ['DataName', 'TrainSet', 'ValidSet', 'TestSet', 'Entities', 'Relations']
    if row_data is not None:
        for line in row_data:
            tb.add_row(line)
    
    if dataName not in data_names:
        tb.add_row([dataName, train_len, valid_len, test_len, total_entities, total_relations])
    print(tb)
    print()
    
    with open(filename, 'w') as f:
        f.write(tb.get_string())
    
    return data, total_entities, total_relations



# #########################################################################################################

def build_graph(data_array):
    data_set = {}
    pairs = set()
    for triple in data_array:
        src, rel, dst = triple
        if (src, rel) in pairs:
            data_set[(src, rel)].add(dst)
        else:
            pairs.add((src, rel))
            data_set[(src, rel)] = {dst}
    graph = {}
    graph['pair->rel_set'] = data_set
    graph['pairs'] = np.array([[x[0], x[1]] for x in pairs])
    return graph

def src_T_dst(data_array):
    src, rel, dst = data_array.transpose(1, 0)
    return np.stack((dst, rel, src)).transpose(1, 0)

def get_triple_set(data_array):
    return set([(x[0], x[1], x[2]) for x in data_array])

def get_all_triples(data):
    train_set = get_triple_set(data['train'])
    valid_set = get_triple_set(data['valid'])
    test_set = get_triple_set(data['test'])
    return train_set | valid_set | test_set

def get_select_src_rate(data):
    left_entity = {}    # the entity on the left of relation in a triplet.
    right_entity = {}     # the entity on the right of relation in a triplet.
    rel_set = set()
    
    for fp in ['train', 'valid', 'test']:
        for triplet in data[fp]:
            src, rel, dst = triplet[0], triplet[1], triplet[2]
            rel_set.add(rel)
            if rel not in left_entity:
                left_entity[rel] = {}
            if src not in left_entity[rel]:
                left_entity[rel][src] = 0
            left_entity[rel][src] += 1
            
            if rel not in right_entity:
                right_entity[rel] = {}
            if dst not in right_entity[rel]:
                right_entity[rel][dst] = 0
            right_entity[rel][dst] += 1
    
    left_avg = {}
    for i in list(rel_set):
        left_avg[i] = sum(left_entity[i].values()) * 1.0 / len(left_entity[i])
    right_avg = {}
    for i in list(rel_set):
        right_avg[i] = sum(right_entity[i].values()) * 1.0 / len(right_entity[i])
    headSelector = {}
    for i in list(rel_set):
        headSelector[i] = 1.0 * left_avg[i] / (left_avg[i] + right_avg[i])
    return headSelector

