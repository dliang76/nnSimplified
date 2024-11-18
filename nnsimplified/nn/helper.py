import io
import os
from awstools.s3 import put_object_to_s3, get_s3_object
import torch
from typing import Dict, Tuple, Union, Callable
from itertools import product
import pandas as pd
import numpy as np
from collections import Counter
import inspect
import functools

def _get_call_args(call: Callable) -> dict:
    return list(inspect.signature(call).parameters.keys())


def _get_call_default_args(call: Callable) -> dict:

    return {k:v.default for k,v in inspect.signature(call).parameters.items() if v.default is not inspect.Parameter.empty}


def _extract_object_config_info(object_name_args: Union[str, Dict[str, dict]]) -> Tuple[str, dict]:
    ''' Simple helper func for parsing configurations for loss, optmizer or lr scheduler.
        The function takes object_config (a dictionary of the form {name: arg_dict})
        e.g
            {'CrossEntropyLoss': {'weight': torch.FloatTensor([1,2,3]), 'reduction': 'none'}}
        and returns name and argument dictionary
           'CrossEntropyLoss' ,  {'weight': torch.FloatTensor([1,2,3]), 'reduction': 'none'}}

        If object_config only contains str name, then name and an empty {} are returned; this will force default argument values

        Author(s): dliang1122@gmail.com

        Arg
        -----
        object_name_args (str, dict): configuration input. Can be simply name (e.g loss function name, optimizer name etc)
                                  or {name: {arg: value}} dictionary for specifying arguments loss, optmizer or lr scheduler.
                                  If only name is specified, the object will just use default values.

        Return
        -----
        name (str) and config (dict)
    '''
    # extract info
    if isinstance(object_name_args, dict):
        name = list(object_name_args)[0]
        arg_dict = object_name_args[name]
    elif isinstance(object_name_args, str):
        name = object_name_args
        arg_dict = {}
    else:
        raise ValueError(
            f"Format not recognized. The object_config argument must be either a str name or a dictionary with the str name as the key and its arguments (dict) as the value")

    return name, arg_dict


class _weight_init():
    # available pytorch weight initialization methods
    available = [k[:-1] for k, v in torch.nn.init.__dict__.items() if
                        k.endswith('_') and callable(v) and not k.startswith('_')]

    def __init__(self, weight_init: str | Dict[str, dict]):

        self.weight_init = weight_init

    @property
    def weight_init(self):
        '''getter for weight_init method name'''
        return self._weight_init

    @weight_init.setter
    def weight_init(self, weight_init: str | Dict[str, dict]):
        '''setter for weight_init method'''

        # check input format
        if isinstance(weight_init, str):
            name = weight_init
            give_config = {}

        elif isinstance(weight_init, dict):
            if len(weight_init) > 1:
                raise ValueError('Cannot specify more than 1 weight initialization method!')

            name = list(weight_init.keys())[0]    
            give_config = weight_init[name]

            if not isinstance(give_config, dict):
                raise ValueError('Weight initialization arguments must be specified in python dict format!')
        else:
            raise ValueError('weight_init accepts only string (name of init method) or dict ({"name of init method": dictionary of init method args})')

        # check input validity
        if name not in self.available:
            raise ValueError(f'{name} is not a valid initialization method. Available: {self.available}')       

        # get function
        call = eval(f'torch.nn.init.{name}_')
        # get function default arguments
        default_config = _get_call_default_args(call)

        ### Check config arguments 
        # check whether the function accepts keyword arguments
        accept_kwargs = '**' in list(inspect.signature(call).parameters.values())[-1].__str__()

        if not accept_kwargs:
            unexpected_args = set(give_config) - set(default_config)

            if unexpected_args :
                raise TypeError(f"'{name}' method has no argument {', '.join(unexpected_args)}!")
        
        # store config
        self._config  = default_config | give_config

        # create weight init function
        self._weight_init = functools.partial(call, **self._config)
        functools.update_wrapper(self._weight_init, call)

    def __call__(self, tensor: torch.Tensor):
        
        return self._weight_init(tensor)
    
    def __repr__(self):
        config_str = ', '.join([f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}" for k,v in self._config.items()])

        result = f'{self._weight_init}\nInit Method: {self._weight_init.__name__}\nConfig: ({config_str})'

        return result


def load_torch_object(path: str, device: Union[str, torch.device] = 'cpu'):
    ''' method for loading object that was saved using torch; load into cpu device only for safety

        Author(s): dliang1122@gmail.com

        Args
        ----------
        path (str): object path 

        Returns
        -------
        torch objects
    '''

    if path.startswith('s3://'):
        pickled_object = get_s3_object(path)
        f = io.BytesIO(pickled_object.read())
    else:
        f = path

    return torch.load(f = f, map_location = device)


def save_torch_object(torch_obj, path: str):
    ''' method for saving torch object

        Author(s): dliang1122@gmail.com
       
        Args
        ----------
        torch_obj (anything picklable): torch objects. E.g. model or model/optimizer parameter (state dict)
        path (str): save path 

        Returns
        -------
        None
    '''
    if path.startswith('s3://'):
        # save object to memory buffer first
        buffer = io.BytesIO()
        torch.save(torch_obj, buffer)

        # move object in buffer to s3
        put_object_to_s3(obj = buffer.getvalue(), s3_path = path)
    elif path.startswith('hdfs://'):
        raise RuntimeError("Hadoop file system is not currently supported!")
    else:
        # create directory if not found
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # save to local
        torch.save(torch_obj, path)


def load_torch_model(path: str,
                     device: Union[str, torch.device] = 'cpu'):
    ''' wrapper for loading torch model

        Author(s): dliang1122@gmail.com

        Args
        ----------
        path (str): object path 

        Returns
        -------
        torch.nn.module
    '''

    # load model object
    model = load_torch_object(path = path, device = device)

    # set to eval mode for safety
    model.eval()

    return model


def agg_cm_counts(truth: torch.Tensor,
                  prediction: torch.Tensor,
                  prev_cm_count_dict: Dict[tuple, int] = {},
                  live_update = False) -> Dict[tuple, int]:
    '''function for aggregating counts (true positive, false posive, true negative, false negative)
       for generating confusion matrix; useful for mini-batch training where one cannot get
       all the counts at once.

       Author(s): dliang1122@gmail.com

       Arg
       ----
       truth (torch.Tensor): true values
       prediction (torch.Tensor): predicted values
       prev_cm_count_dict (dict): existing count dictionary
       live_update (bool): whether to update the count dictionary in place

       return
       ------
       dict: dictionary of counts with key the label pair combo and value the counts. e.g. {(0,0): 128, (0,1): 234, (1,0): 123, (1,1): 321}
    '''

    if live_update:
        cm_count_dict = prev_cm_count_dict # assign by reference; will update the input (prev_cm_count_dict)
    else:
        cm_count_dict = prev_cm_count_dict.copy() # make a copy so we don't modify the input

    # flatten the tensors
    if truth.ndim > 1:
        truth = truth.squeeze()

    if prediction.ndim > 1:
        prediction = prediction.squeeze()

    # gather the total counts of each truth-prediction pair
    counts = dict(Counter(zip(truth.cpu().numpy(), prediction.cpu().numpy())))

    # update counts
    for k,v in counts.items():

        if cm_count_dict.get(k):
            cm_count_dict[k] += v
        else:
            cm_count_dict[k] = v

    return cm_count_dict


def add_cm_counts(cm_1: Dict[tuple, int] = {}, cm_2: Dict[tuple, int] = {}) -> Dict[tuple, int]:
    ''' function for adding two confusion matrices in dictioanary format (e.g {(P,PP): 10, (P, PN): 20, (N, PP): 5, (N, PN): 10})

        Author(s): dliang1122@gmail.com
    
        Arg:
        --------
        cm_1 (dict): confusion matrix 1
        cm_2 (dict): confusion matrix 2

        Return:
        ---------
        dict: resulting confusion matrix in dict format
    '''

    # find keys in cm_1
    keys1 = cm_1.keys()

    # find keys in cm_2
    keys2 = cm_2.keys()

    # find all unique keys
    keys = set(list(keys1) + list(keys2))

    # initialize aggregate dictionary
    aggr_dict = dict(zip(keys, np.zeros(len(keys), dtype=int)))

    # aggregate
    for k in keys:
        if k in keys1:
            aggr_dict[k] += cm_1[k]
        if k in keys2:
            aggr_dict[k] += cm_2[k]

    return aggr_dict


def convert_cm_counts_to_cm(cm_count_dict:Dict[Tuple[int,int], int],
                            n_classes: int) -> pd.DataFrame:
    ''' function for converting label pair counts to confuion matrix in tabular format (pandas) for better readability

        Author(s): dliang1122@gmail.com
    
        Arg:
        --------
        cm_count_dict (dict): confusion matrix in dict format
        n_classes (dict): number of classes in the label; this is necessary as sometimes there is no prediction for certain class labels

        Return:
        ---------
        pandas dataframe
    '''
    all_classes = range(n_classes)
    truth_pred_ind = product(all_classes, all_classes)

    # get confusion matrix
    cm = pd.DataFrame({'counts': cm_count_dict.values()},
                      index= cm_count_dict.keys()).reindex(truth_pred_ind)\
                                               .sort_index()\
                                               .fillna(0)\
                                               .astype(int)\
                                               .reset_index()\
                                               .rename({'level_0': 'label', 'level_1': 'prediction'}, axis = 1)\
                                               .pivot_table(index = 'label', columns = 'prediction', values = 'counts', sort = True)

    return cm