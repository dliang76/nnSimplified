import os
import itertools
import boto3
import json
import pickle
import numpy as np
import pandas as pd
import ray
from typing import List
# from .dataloading.helper import load_single_parquet_ray, list_batching
from awstools.s3 import s3_object_exists, parse_s3_path, put_object_to_s3, get_s3_object
#from .emr import get_internal_cluster_id
from awstools.helper import get_aws_credentials
from botocore.config import Config
import matplotlib.pyplot as plt
from collections import defaultdict

def save_pickle(obj: object, path: str, **kwargs):
    '''pickle an object and save it to storage
    
       Author(s): dliang1122@gmail.com
    '''

    pickled_obj = pickle.dumps(obj)

    if path.startswith('s3://'):
        aws_credential = kwargs.get('aws_credential', None)
        put_object_to_s3(obj = pickled_obj, s3_path = path, aws_credential = aws_credential)
    else:
        save_dir = os.path.dirname(path) # get save directory
        os.makedirs(save_dir,exist_ok= True) # create the directory if not found

        with open(path, 'wb') as  f:
            f.write(pickled_obj)


def load_pickle(path: str, **kwargs):
    '''load an object from storage
    
       Author(s): dliang1122@gmail.com
    '''
    if path.startswith('s3://'):
        aws_credential = kwargs.get('aws_credential', None)
        pickled_obj = get_s3_object(s3_path = path, aws_credential = aws_credential).read()
    else:
        with open(path, 'rb') as f:
            pickled_obj = f.read()

    return pickle.loads(pickled_obj)


def save_json(obj: object, path: str, **kwargs):
    '''save an object in json format to storage
    
       Author(s): dliang1122@gmail.com
    '''
    json_obj =  json.dumps(obj)

    if path.startswith("s3://"):
        aws_credential = kwargs.get('aws_credential', None)
        put_object_to_s3(obj = json_obj, s3_path = path, aws_credential = aws_credential)
    else:
        save_dir = os.path.dirname(path) # get save directory
        os.makedirs(save_dir,exist_ok= True) # create the directory if not found

        with open(path, 'w') as f:
            f.write(json_obj)


def load_json(path: str, **kwargs):
    '''load json file from storage

       Author(s): dliang1122@gmail.com
    '''
    if path.startswith("s3://"):
        aws_credential = kwargs.get('aws_credential', None)
        obj = get_s3_object(s3_path = path, aws_credential = aws_credential).read()
    else:
        with open(path, 'r') as f:
            obj = f.read()

    return json.loads(obj)


def flatten_list(lst: List, remove_none = False):
    flattened = []

    for i in lst:
        if not (remove_none and i is None):
            if type(i) in (tuple, list):
                flattened.extend(flatten_list(i))
            else:
                flattened.append(i)
    
    return flattened
