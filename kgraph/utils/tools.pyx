#cython: language_level=3
# distutils: language = c++
import os
import numpy as np
cimport numpy as np
from collections import Counter


def get_str2id(path_data, sep="\t", no_sort=True):
    relation2id = dict()
    id2relation = dict()
    entity2id = dict()
    id2entity = dict()
    entity_num = 0
    relation_num = 0

    element2id = ["relation2id_no_sort.txt", "entity2id_no_sort.txt"] if no_sort else ["relation2id_on_sort.txt", "entity2id_on_sort.txt"]

    for f in element2id:
        with open(os.path.join(path_data, f), "r") as f:
            for l in f.readlines():
                line = l.strip().split(sep)
                if f == "relation2id_no_sort.txt" or f == "relation2id_on_sort.txt":
                    if len(line) == 1:
                        relation_num = int(line[0])
                        continue
                    relation2id[line[0]] = int(line[1])
                    id2relation[int(line[1])] = line[0]
                else:
                    if len(line) == 1:
                        entity_num = int(line[0])
                        continue
                    entity2id[line[0]] = int(line[1])
                    id2entity[int(line[1])] = line[0]
    return (entity2id, id2entity), (relation2id, id2relation) (entity_num, relation_num)

def write_str2id(data, save_path, sep='\t'):
    with open(save_path, 'w') as f:
        f.write("%d\n" % len(data))
        for i in data:
            f.write("%s%s%d\n" % (i, sep, data[i]))

def load_triple_original_file(path_file, sep='\t'):
    data = []
    with open(path_file, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(sep)
            data.append(line[:3])
    return data

def write_triple_from_original_data(original_data, path_file, str2id, sep='\t'):

    with open(path_file, 'w') as f:
        num = len(original_data)
        f.write("%d\n" %num)
        for l in original_data:
            f.write("%d%s%d%s%d\n"%(
                str2id[0][l[0]],
                sep,
                str2id[1][l[1]],
                sep,
                str2id[0][l[2]]))

def generateTripleIdFile(from_original_data_dir, to_dir, sep='\t', no_sort=True):
    orignal_triple_files = ['train.txt', 'valid.txt', 'test.txt']
    to_id_files = [os.path.join(to_dir, x) for x in ['train2id.txt', 'valid2id.txt', 'test2id.txt']]

    entity2id_file_name = 'entity2id_no_sort.txt' if no_sort else 'entity2id_on_sort.txt'
    relation2id_file_name = 'relation2id_no_sort.txt' if no_sort else 'relation2id_on_sort.txt'
    
    if os.path.exists(os.path.join(from_original_data_dir, entity2id_file_name)):
        (ent2id, _), (rel2id, _), (entTotal, relTotal) = get_str2id(
            from_original_data_dir, sep=sep, no_sort=no_sort)
    else:
        data = []
        for file_name in orignal_triple_files:
            file_name = os.path.join(from_original_data_dir, file_name)
            data += load_triple_original_file(file_name, sep=sep)
        
        if no_sort:
            entities = []
            relations = []
            entities_set = set()
            relations_set = set()

            ent2id = {}
            rel2id = {}
            num_ent = 0
            num_rel = 0
            for element in data:
                if element[0] not in entities_set:
                    entities_set.add(element[0])
                    entities.append(element[0])
                    ent2id[element[0]] = num_ent
                    num_ent += 1
                
                if element[2] not in entities_set:
                    entities_set.add(element[2])
                    entities.append(element[2])
                    ent2id[element[2]] = num_ent
                    num_ent += 1
                
                if element[1] not in relations_set:
                    relations_set.add(element[1])
                    relations.append(element[1])
                    rel2id[element[1]] = num_rel
                    num_rel += 1
        else:
            data = []
            for file_name in orignal_triple_files:
                file_name = os.path.join(from_original_data_dir, file_name)
                data += load_triple_original_file(file_name, sep=sep)
            entities = []
            relations = []
            for element in data:
                entities.append(element[0])
                entities.append(element[2])
                relations.append(element[1])
            entities = Counter(entities).most_common()
            relations = Counter(relations).most_common()
            ent2id = {ent[0]: value_ for value_, ent in enumerate(sorted(entities, key=lambda x: x[1], reverse=True))}
            rel2id = {rel[0]: value_ for value_, rel in enumerate(sorted(relations, key=lambda x: x[1], reverse=True))}
        entTotal = len(entities)
        relTotal = len(relations)
        write_str2id(ent2id, os.path.join(from_original_data_dir, entity2id_file_name))
        write_str2id(rel2id, os.path.join(from_original_data_dir, relation2id_file_name))
        del data, entities, relations

    for i, file_name in enumerate(orignal_triple_files):
        file_name = os.path.join(from_original_data_dir, file_name)
        data = load_triple_original_file(file_name, sep=sep)
        write_triple_from_original_data(data, to_id_files[i], (ent2id, rel2id), sep=sep)

def clean(
    np.ndarray[int, ndim=2] train_array,
    np.ndarray[int, ndim=2] valid_array,
    np.ndarray[int, ndim=2] test_array
):
    cdef np.ndarray train_ent = np.unique(np.concatenate(train_array[:, 0], train_array[:, 2]))
    cdef np.ndarray train_rel = np.unique(train_array[:, 1])

    valid_index = valid_array[:, 0].isin(train_ent) & valid_array[:, 1].isin(train_rel) & valid_array[:, 2].isin(train_ent)
    test_index = test_array[:, 0].isin(train_ent) & test_array[:, 1].isin(train_rel) & test_array[:, 2].isin(train_ent)

    return valid_array[valid_index], test_array[test_index]

def generateN2N(
    np.ndarray[int, ndim=2] train_array,
    np.ndarray[int, ndim=2] valid_array,
    np.ndarray[int, ndim=2] test_array,
    save_path="", sep='\t'
):
    cdef float rig_n, lef_n
    lef = {}                # pair(h, r) -->tail   the lef pair of a triple
    rig = {}                # pair(r, t) -->head
    rel_lef = {}            # rel --> head --> num
    rel_rig = {}            # rel --> tail --> num
    data = np.concatenate((train_array, valid_array, test_array), axis=0)
    for triple in data:
        h, r, t = triple
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
    
    with open(os.path.join(save_path, 'constraint.txt'), 'w') as f:
        f.write("%d\n" % len(rel_lef))
        for i in rel_lef:
            f.write("%d %d" % (i, len(rel_lef[i])))
            for j in rel_lef[i]:
                f.write(" %d" % j)
            f.write("\n")
            f.write("%d %d" % (i, len(rel_rig[i])))
            for j in rel_rig[i]:
                f.write(" %d" % j)
            f.write("\n")

    rel_lef = {}     # r --> the in degree
    tot_lef = {}     # r --> total the number of (r, X) in the lef key.
    rel_rig = {}     # r --> the out degree
    tot_rig = {}     # r --> total the number of (X, r) in the rig key.

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

    f11 = open(os.path.join(save_path, "1-1.txt"), "w")
    f1n = open(os.path.join(save_path, "1-n.txt"), "w")
    fn1 = open(os.path.join(save_path, "n-1.txt"), "w")
    fnn = open(os.path.join(save_path, "n-n.txt"), "w")
    for triple in test_array:
        h, r, t = triple
        content = f'{h}\t{r}\t{t}\n'
        lef_n = rel_lef[r] / tot_lef[r]
        rig_n = rel_rig[r] / tot_rig[r]
        if (rig_n < 1.5 and lef_n < 1.5):
            f11.write(content)
            # fall.write("0"+"\t"+content)
        if (rig_n >= 1.5 and lef_n < 1.5):
            f1n.write(content)
            # fall.write("1"+"\t"+content)
        if (rig_n < 1.5 and lef_n >= 1.5):
            fn1.write(content)
            # fall.write("2"+"\t"+content)
        if (rig_n >= 1.5 and lef_n >= 1.5):
            fnn.write(content)
            # fall.write("3"+"\t"+content)
    # fall.close()
    f11.close()
    f1n.close()
    fn1.close()
    fnn.close()

