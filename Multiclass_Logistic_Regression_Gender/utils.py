'''
Author: Zhengxiang (Jack) Wang 
Date: 2021-07-04
GitHub: https://github.com/jaaack-wang 
About: Some utlity functions for the multi-class logistic 
regression classifier that predicts gender of Chinese names.
'''
from random import shuffle, seed
import numpy as np
from collections import defaultdict
import json


def load_char_dic(filepath='data/char_dic.json'):
    '''load the char dict that is made from the selected train set.
    
    Params:
        filepath: str --> path to the dict
    Rerturn:
        char_dic: dict
    '''
    dic = json.load(open(filepath, 'r'))
    dic = defaultdict(lambda :dic['size'], dic)
    return dic


def data_loader(filepath):
    '''loading the dataset and returns names and genders. 
    Params:
        filepath: str or list
    '''
    def readFile(path):
        f = open(path, 'r')
        next(f)
        out = []
        for line in f:
            line = line.split('\t')
            # line[0] = first name (full name for test set), line[1] = gender
            out.append([line[0], line[1].strip()])
        return out
    
    if isinstance(filepath, str):
        return readFile(filepath)
    elif isinstance(filepath, list):
        return [readFile(path) for path in filepath]
    else:
         raise TypeError('filepath must be either a str or a list.')
            
            
def name2vec(name, char_dic):
    '''convert a given name into vec with one-hot encoding. 
    
    Params:
        name: str
        char_dic: dict
            this dict contains the indices for chars in the selected train set that 
            can be used as an one-hot encoder
    Returns:
        name_vec: array-like (dim=(1, n+1)) where n = num of chars in the selected train set
            please the first columm for this row vector is equal to 1 (i.e., x0=1)
    '''
    name_vec = np.zeros((1, char_dic['size']+1))
    # x0 = 1
    name_vec[0,0] = 1
    for char in name:
        name_vec[0, char_dic[char]] = 1
    return name_vec


def convert_example(ds, char_dic):
    '''Converts the dataset and returns both names and gender as vectors.
    '''
    # m = num of examples, n = num of dimensions
    m, n = len(ds), char_dic['size']
    name_vec = np.zeros((m, n+1))
    gender_vec = np.zeros((m, 1))
    for i in range(m):
        name_vec[i] = name2vec(ds[i][0], char_dic)
        if ds[i][1] == 'F': gender_vec[i] = 1
        elif ds[i][1] == 'U': gender_vec[i] = 2
    
    return name_vec, gender_vec


# load the Chinese Last Names and 
# create bisyllabic last name and monosyllabic last names
f = open('data/ChineseLastNames.txt', 'r')
next(f)
lastnames = [line.split('\t')[0] for line in f]    
uni_nam, bi_nam = (), ()
for n in lastnames:
    if len(n) == 1: uni_nam += (n, )
    else: bi_nam += (n, )
        

def getFirstName(name, bi_nam=bi_nam, uni_nam=uni_nam):
    '''Returns the first name of a given name.
    '''
    if name.startswith(bi_nam): return name[2:]
    elif name.startswith(uni_nam): return name[1:]
    else: return name
