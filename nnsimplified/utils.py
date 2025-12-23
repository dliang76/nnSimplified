import os
import itertools
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def save_pickle(obj: object, path: str, **kwargs):
    '''pickle an object and save it to storage
    
       Author(s): dliang1122@gmail.com
    '''

    pickled_obj = pickle.dumps(obj)

    save_dir = os.path.dirname(path) # get save directory
    os.makedirs(save_dir,exist_ok= True) # create the directory if not found

    with open(path, 'wb') as  f:
        f.write(pickled_obj)


def load_pickle(path: str, **kwargs):
    '''load an object from storage
    
       Author(s): dliang1122@gmail.com
    '''

    with open(path, 'rb') as f:
        pickled_obj = f.read()

    return pickle.loads(pickled_obj)


def save_json(obj: object, path: str, **kwargs):
    '''save an object in json format to storage
    
       Author(s): dliang1122@gmail.com
    '''
    json_obj =  json.dumps(obj)

    save_dir = os.path.dirname(path) # get save directory
    os.makedirs(save_dir,exist_ok= True) # create the directory if not found

    with open(path, 'w') as f:
        f.write(json_obj)


def load_json(path: str, **kwargs):
    '''load json file from storage

       Author(s): dliang1122@gmail.com
    '''

    with open(path, 'r') as f:
        obj = f.read()

    return json.loads(obj)


def flatten_list(lst: list, remove_none = False):
    flattened = []

    for i in lst:
        if not (remove_none and i is None):
            if type(i) in (tuple, list):
                flattened.extend(flatten_list(i))
            else:
                flattened.append(i)
    
    return flattened
