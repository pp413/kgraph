#!/usr/bin/env python
# coding=utf-8
import hashlib
import os
import tarfile
import warnings
import zipfile

import numpy as np
import pandas as pd
import prettytable as pt

from collections import Counter

try:
    import requests
except ImportError:
    class requests_failed_to_import(object):
        pass
    requests = requests_failed_to_import

root = os.path.join(os.path.expanduser('~'), '.KnowledgeGraphDataSets')
os.environ['KG_DIR'] = root
if not os.path.exists(root):
    os.makedirs(root)

all_data = {'wn18', 'wn18RR', 'fb15k', 'fb15k-237', 'YAGO3-10', 'wordnet11', 'freebase13'}


def down_url(data_name):
    url = 'https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/{}.zip'.format(data_name)
    return url


def set_root_dir(path=None):
    if path is not None:
        root = os.path.join(path, '.KnowledgeGraphDataSets')
    dirname = os.environ.get('KG_DIR')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


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


def load_from_csv(file_path, sep='\t', header=None):

    df = pd.read_csv(file_path, sep=sep, header=header,
                     names=None, dtype=str)
    df = df.drop_duplicates()
    return df.values


def _load_data(fdir, data_name, data_sha1):
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
        train = load_from_csv(check_train)
        valid = load_from_csv(check_valid)
        test = load_from_csv(check_test)

        return {'train': train, 'valid': valid, 'test': test, 'name': data_name}


def _clean_data(X):

    train = pd.DataFrame(X['train'], columns=['s', 'p', 'o'])
    valid = pd.DataFrame(X['valid'], columns=['s', 'p', 'o'])
    test = pd.DataFrame(X['test'], columns=['s', 'p', 'o'])

    train_ent = np.unique(np.concatenate((train.s, train.o)))
    train_rel = train.p.unique()

    valid_idx = valid.s.isin(train_ent) & valid.o.isin(train_ent) & valid.p.isin(train_rel)
    test_idx = test.s.isin(train_ent) & test.o.isin(train_ent) & test.p.isin(train_rel)

    filtered_valid = valid[valid_idx].values
    filtered_test = test[test_idx].values

    filtered_X = {'train': train.values, 'valid': filtered_valid, 'test': filtered_test,
                  'name': X['name']}

    return filtered_X


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


def _str_to_idx(data):
    if os.path.exists(os.path.join(os.environ.get('KG_DIR'), data['name'], 'entity2id.txt')):
        entities_idx_dict = {}
        relations_idx_dict = {}
        ent2id = os.path.join(os.environ.get('KG_DIR'), data['name'], 'entity2id.txt')
        with open(ent2id, 'r') as f:
            total_entities = (int)(f.readline().strip())
            for i in range(total_entities):
                line = f.readline()
                entity, ent_id = line.strip().split('\t')
                ent_id = int(ent_id)
                entities_idx_dict[entity] = ent_id
        rel2id = os.path.join(os.environ.get('KG_DIR'), data['name'], 'relation2id.txt')
        with open(rel2id, 'r') as f:
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
            entities, key=lambda x: x[1]
        ))}

        relations_idx_dict = {rel[0]: value for value, rel in enumerate(sorted(
            relations, key=lambda x: x[1]
        ))}
    
    train2id = np.asarray(_triplets_as_list(data['train'], entities_idx_dict, relations_idx_dict))
    valid2id = np.asarray(_triplets_as_list(data['valid'], entities_idx_dict, relations_idx_dict))
    test2id = np.asarray(_triplets_as_list(data['test'], entities_idx_dict, relations_idx_dict))

    if not os.path.exists(os.path.join(os.environ.get('KG_DIR'), data['name'], 'constrain.txt')):
        write_dataset(train2id, data['name'], name='train2id.txt')
        write_dataset(valid2id, data['name'], name='valid2id.txt')
        write_dataset(test2id, data['name'], name='test2id.txt')
        write_str2id(entities_idx_dict, data['name'], name='entity2id.txt')
        write_str2id(relations_idx_dict, data['name'], name='relation2id.txt')
        
        lef = {}    # (h, r) --> t
        rig = {}    # (r, t) --> h
        rellef = {}     # r --> the in degree of relation r in the graph
        relrig = {}     # r --> the in degree of relation r in the graph
        
        all_data = np.concatenate((train2id, valid2id, test2id), axis=0)
        for triplet in all_data:
            h, r, t = triplet
            if not (h, r) in lef:
                lef[(h, r)] = []
            if not (r, t) in rig:
                rig[(r, t)] = []
            lef[(h, r)] += [t]
            rig[(r, t)] += [h]
            if not r in rellef:
                rellef[r] = {}
            if not r in relrig:
                relrig[r] = {}
            rellef[r][h] = 1
            relrig[r][t] = 1
        
        f = open(os.path.join(os.environ.get('KG_DIR'), data['name'], 'constrain.txt'), 'w')
        f.write("%d\n"%(len(rellef)))
        for i in rellef:
            f.write("%d\t%d"%(i,len(rellef[i])))
            for j in rellef[i]:
                f.write("\t%d"%(j))
            f.write("\n")
            f.write("%d\t%d"%(i,len(relrig[i])))
            for j in relrig[i]:
                f.write("\t%d"%(j))
            f.write("\n")
        f.close()
        
        rellef = {}     # r --> the in degree
        totlef = {}     # r --> the number of in kinds of entities
        relrig = {}
        totrig = {}
        # lef: {h, r}
        # rig: {r, t}
        for i in lef:
            if not i[1] in rellef:
                rellef[i[1]] = 0
                totlef[i[1]] = 0
            rellef[i[1]] += len(lef[i])
            totlef[i[1]] += 1.0

        for i in rig:
            if not i[0] in relrig:
                relrig[i[0]] = 0
                totrig[i[0]] = 0
            relrig[i[0]] += len(rig[i])
            totrig[i[0]] += 1.0

        s11=0
        s1n=0
        sn1=0
        snn=0
        f = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "test2id.txt"), "r")
        tot = (int)(f.readline())
        for i in range(tot):
            content = f.readline()
            h, r, t = [int(x) for x in content.strip().split()]
            rign = rellef[r] / totlef[r]
            lefn = relrig[r] / totrig[r]
            if (rign < 1.5 and lefn < 1.5):
                s11+=1
            if (rign >= 1.5 and lefn < 1.5):
                s1n+=1
            if (rign < 1.5 and lefn >= 1.5):
                sn1+=1
            if (rign >= 1.5 and lefn >= 1.5):
                snn+=1
        f.close()


        f = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "test2id.txt"), "r")
        f11 = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "1-1.txt"), "w")
        f1n = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "1-n.txt"), "w")
        fn1 = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "n-1.txt"), "w")
        fnn = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "n-n.txt"), "w")
        fall = open(os.path.join(os.environ.get('KG_DIR'), data['name'], "test2id_all.txt"), "w")
        tot = (int)(f.readline())
        fall.write("%d\n"%(tot))
        f11.write("%d\n"%(s11))
        f1n.write("%d\n"%(s1n))
        fn1.write("%d\n"%(sn1))
        fnn.write("%d\n"%(snn))
        for i in range(tot):
            content = f.readline()
            h, r, t = [int(x) for x in content.strip().split()]
            rign = rellef[r] / totlef[r]
            lefn = relrig[r] / totrig[r]
            if (rign < 1.5 and lefn < 1.5):
                f11.write(content)
                fall.write("0"+"\t"+content)
            if (rign >= 1.5 and lefn < 1.5):
                f1n.write(content)
                fall.write("1"+"\t"+content)
            if (rign < 1.5 and lefn >= 1.5):
                fn1.write(content)
                fall.write("2"+"\t"+content)
            if (rign >= 1.5 and lefn >= 1.5):
                fnn.write(content)
                fall.write("3"+"\t"+content)
        fall.close()
        f.close()
        f11.close()
        f1n.close()
        fn1.close()
        fnn.close()

    return {
        'train': train2id,
        'valid': valid2id,
        'test': test2id,
        'ent_dict': entities_idx_dict, 'rel_dict': relations_idx_dict,
        'name': data['name']
    }, {
        'train': data['train'], 'valid': data['valid'], 'test': data['test'], 'name': data['name']
        }, total_entities, total_relations


def low_name(data_name_str):
    return data_name_str.lower()

def _pprint(data, total_entities, total_relations, dataName=''):
    
    train_len = len(data['train'])
    valid_len = len(data['valid'])
    test_len = len(data['test'])
    
    tb = pt.PrettyTable()
    tb.field_names = ['DataName', 'TrainSet', 'ValidSet', 'TestSet', 'Entities', 'Relations']
    tb.add_row([dataName, train_len, valid_len, test_len, total_entities, total_relations])
    print(tb)
    print()
    return data, total_entities, total_relations

def load_fb15k(original=False, clean_unseen=True):
    """Load the FB15k dataset

    .. warning::
        The dataset includes a large number of inverse relations that spilled to the test set, and its use in
        experiments has been deprecated. Use FB15k-237 instead.

    FB15k is a split of Freebase, first proposed by :cite:`bordes2013translating`.

    The FB15k dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    The dataset is divided in three splits:
    
    - ``train``
    - ``valid``
    - ``test``

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    FB15K     483,142   50,000  59,071  14,951        1,345
    ========= ========= ======= ======= ============ ===========
    Cleaned   483,142   50,000  59,071  14,951        1,345
    ========= ========= ======= ======= ============ ===========

    Parameters
    ----------
    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    """
    root = set_root_dir()
    data_name = 'fb15k'
    url = down_url(data_name)
    tgz_path = download(url, root)
    fdir = os.path.join(root, data_name)
    extract_archive(tgz_path, fdir)

    data_sha1 = {
        'train': '578bdb6b4311d22d4baf7da30aaadf03d687c84d',
        'test': '00d340728878df4f0b318fd1f488855e9b770425',
        'valid': '2694fe891109dea3470bd975dd55eeb12ef30cbd'
    }

    data = _load_data(fdir, data_name, data_sha1)

    data = _clean_data(data) if clean_unseen else data
    data, original_data, total_entities, total_relations = _str_to_idx(data)
    if original:
        return _pprint(original_data, total_entities, total_relations, 'FB15k')

    return _pprint(data, total_entities, total_relations, 'FB15k')


def load_fb15k237(original=False, clean_unseen=True):
    """Load the FB15k-237 dataset

    FB15k-237 is a reduced version of FB15K. It was first proposed by :cite:`toutanova2015representing`.

    The FB15k-237 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    The dataset is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    FB15K-237 272,115   17,535  20,466  14,541        237
    ========= ========= ======= ======= ============ ===========
    Cleaned   272,115   17,516  20,438  14,505        237
    ========= ========= ======= ======= ============ ===========


    Parameters
    ----------
    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    return data, total_entities, total_relations
    """

    root = set_root_dir()
    data_name = 'fb15k-237'
    url = down_url(data_name)
    tgz_path = download(url, root)
    fdir = os.path.join(root, data_name)
    extract_archive(tgz_path, fdir)

    data_sha1 = {
        'train': '1448a31a528da315217960edeca97f68209f8254',
        'test': '263a7bd582cf1d27961fc4143ca2bee474bf03fc',
        'valid': '732eb032787161e61b7bcc7ab3965b7569b405ce'
    }

    data = _load_data(fdir, data_name, data_sha1)

    data = _clean_data(data) if clean_unseen else data
    data, original_data, total_entities, total_relations = _str_to_idx(data)
    if original:
        return _pprint(original_data, total_entities, total_relations, 'FB15k-237')

    return _pprint(data, total_entities, total_relations, 'FB15k-237')


def load_wn18(original=False, clean_unseen=True):
    """Load the WN18 dataset

    .. warning::
        The dataset includes a large number of inverse relations that spilled to the test set, and its use in
        experiments has been deprecated. Use WN18RR instead.

    WN18 is a subset of Wordnet. It was first presented by :cite:`bordes2013translating`.

    The WN18 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    The dataset is divided in three splits:

    - ``train``: 141,442 triples
    - ``valid`` 5,000 triples
    - ``test`` 5,000 triples

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    WN18      141,442   5,000   5,000   40,943        18
    ========= ========= ======= ======= ============ ===========
    Cleaned   141,442   5,000   5,000   40,943        18 
    ========= ========= ======= ======= ============ ===========
    
    Parameters
    ----------
    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    return data, total_entities, total_relations
    """

    root = set_root_dir()
    data_name = 'wn18'
    url = down_url(data_name)
    tgz_path = download(url, root)
    fdir = os.path.join(root, data_name)
    extract_archive(tgz_path, fdir)

    data_sha1 = {
        'train': 'b78e956440e0b7631517aa1b230818f581281e6d',
        'test': 'e5308598809646ad01da6b4d9cede189f918aa31',
        'valid': '2433f787c8dcf3ac376d70ebd358ef3998313ea3'
    }

    data = _load_data(fdir, data_name, data_sha1)

    data = _clean_data(data) if clean_unseen else data
    data, original_data, total_entities, total_relations = _str_to_idx(data)
    if original:
        return _pprint(original_data, total_entities, total_relations, 'WN18')

    return _pprint(data, total_entities, total_relations, 'WN18')


def load_wn18rr(original=False, clean_unseen=True):
    """Load the WN18RR dataset

    The dataset is described in :cite:`DettmersMS018`.

    The WN18RR dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.


    It is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    WN18RR    86,835    3,034   3,134   40,943        11
    ========= ========= ======= ======= ============ ===========
    Cleaned   86,835    3,034   3,134   40,943        11
    ========= ========= ======= ======= ============ ===========

    .. warning:: WN18RR's validation set contains 198 unseen entities over 210 triples.
        The test set has 209 unseen entities, distributed over 210 triples.

    Parameters
    ----------
    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    return data, total_entities, total_relations
    """

    root = set_root_dir()
    data_name = 'wn18RR'
    url = down_url(data_name)
    tgz_path = download(url, root)
    fdir = os.path.join(root, data_name)
    extract_archive(tgz_path, fdir)

    data_sha1 = {
        'train': '078fd2890583f99d75342d0eeea6d0c4e6167c76',
        'test': 'da28a9c1759d66f87873d8e1ecc4884501211f83',
        'valid': '38bbe458b5f8e36310456cac5e0119f05d39cef5'
    }

    data = _load_data(fdir, data_name, data_sha1)

    data = _clean_data(data) if clean_unseen else data
    data, original_data, total_entities, total_relations = _str_to_idx(data)
    if original:
        return _pprint(original_data, total_entities, total_relations, 'WN18RR')

    return _pprint(data, total_entities, total_relations, 'WN18RR')


def load_yago3_10(original=False, clean_unseen=True):
    """Load the YAGO3-10 dataset
   
    The dataset is a split of YAGO3 :cite:`mahdisoltani2013yago3`,
    and has been first presented in :cite:`DettmersMS018`.

    The YAGO3-10 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    It is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    YAGO3-10  1,079,040 5,000   5,000   123,182       37
    ========= ========= ======= ======= ============ ===========
    Cleaned   1,079,040 4,978   4,982   123,142       37
    ========= ========= ======= ======= ============ ===========

    Parameters
    ----------
    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    return data, total_entities, total_relations
    """

    root = set_root_dir()
    data_name = 'YAGO3-10'
    url = down_url(data_name)
    tgz_path = download(url, root)
    fdir = os.path.join(root, data_name)
    extract_archive(tgz_path, fdir)

    data_sha1 = {
        'train': 'a27945951c79609313e4218b83b158b4660cc214',
        'test': 'b7d88415a9bdb54ef8fed5098c85f3e80bc05ae5',
        'valid': 'e56d645455d9b9751f1e38724d224be430f696b3'
    }

    data = _load_data(fdir, data_name, data_sha1)

    data = _clean_data(data) if clean_unseen else data
    data, original_data, total_entities, total_relations = _str_to_idx(data)
    if original:
        return _pprint(original_data, total_entities, total_relations, 'YAGO3-10')

    return _pprint(data, total_entities, total_relations, 'YAGO3-10')


def load_wn11(original=False, clean_unseen=True):
    """Load the WordNet11 (WN11) dataset

    WordNet was originally proposed in `WordNet: a lexical database for English` :cite:`miller1995wordnet`.

    WN11 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location, it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    It is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    Both the validation and test splits are associated with labels (binary ndarrays),
    with `True` for positive statements and `False` for  negatives:

    - ``valid_labels``
    - ``test_labels``

    ========= ========= ========== ========== ======== ======== ============ ===========
     Dataset  Train     Valid Pos  Valid Neg  Test Pos Test Neg Entities     Relations
    ========= ========= ========== ========== ======== ======== ============ ===========
    WN11      110361    2606       2609       10493    10542    38588        11
    ========= ========= ========== ========== ======== ======== ============ ===========

    Parameters
    ----------
    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    return data, total_entities, total_relations
    """

    root = set_root_dir()
    data_name = 'wordnet11'
    url = down_url(data_name)
    tgz_path = download(url, root)
    fdir = os.path.join(root, data_name)
    extract_archive(tgz_path, fdir)

    data_sha1 = {
        'train': 'dd2a8f65b75b0d96a9313498967a72f4f6c19c75',
        'test': 'f927599f38323450de3739d98cd0b2fced6cd3d4',
        'valid': '732eb032787161e61b7bcc7ab3965b7569b405ce'
    }

    data = _load_data(fdir, data_name, data_sha1)

    data = _clean_data(data) if clean_unseen else data
    if original:
        return data, len(set(data['train'][:, 0])), len(set(data['train'][:,1]))
    data = _str_to_idx(data)

    return (data,
            data['train'][:, 0].max()+1,
            data['train'][:, 1].max()+1)


def load_fb13(clean_unseen=True):
    """Load the Freebase13 (FB13) dataset

    FB13 is a subset of Freebase :cite:`bollacker2008freebase`
    and was initially presented in
    `Reasoning With Neural Tensor Networks for Knowledge Base Completion` :cite:`socher2013reasoning`.

    FB13 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location, it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    It is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    Both the validation and test splits are associated with labels (binary ndarrays),
    with `True` for positive statements and `False` for  negatives:

    - ``valid_labels``
    - ``test_labels``

    ========= ========= ========== ========== ======== ======== ============ ===========
     Dataset  Train     Valid Pos  Valid Neg  Test Pos Test Neg Entities     Relations
    ========= ========= ========== ========== ======== ======== ============ ===========
    FB13      316232    5908       5908       23733    23731    75043        13
    ========= ========= ========== ========== ======== ======== ============ ===========

    Parameters
    ----------
    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    return data, total_entities, total_relations
    """

    root = set_root_dir()
    data_name = 'freebase13'
    url = down_url(data_name)
    tgz_path = download(url, root)
    fdir = os.path.join(root, data_name)
    extract_archive(tgz_path, fdir)

    data_sha1 = {
        'train': '2db7c911c7f2e12896e30fa04e0ad699194f9d9d',
        'test': 'b10401858e49c1a67e1b8b607653faa9acd50672',
        'valid': '732eb032787161e61b7bcc7ab3965b7569b405ce'
    }

    data = _load_data(fdir, data_name, data_sha1)

    data = _clean_data(data) if clean_unseen else data
    data = _str_to_idx(data)

    return (data,
            data['train'][:, 0].max()+1,
            data['train'][:, 1].max()+1)


def load_all_datasets(clean_unseen=True):
    load_wn18(clean_unseen)
    load_wn18rr(clean_unseen)
    load_fb15k(clean_unseen)
    load_fb15k237(clean_unseen)
    load_yago3_10(clean_unseen)
    # load_wn11(clean_unseen)
    # load_fb13(clean_unseen)


if __name__ == "__main__":
    load_all_datasets(False)
