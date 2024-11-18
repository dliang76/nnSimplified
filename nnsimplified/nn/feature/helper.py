from typing import List, Union, ClassVar, Dict, Optional, Callable
import numpy as np
import pandas as pd

def infer_ftype(data):
    # raise a error if no data for type inference found
    if data is None:
        raise ValueError("<ftype> cannot be None unless data is provided for ftype inference.")

    # infer ftype from the data provided
    if isinstance(data, pd.Series):
        # if pandas series
        if data.dtype.name == 'category':
            ftype = 'categorical'
        elif data.dtype.name == 'object':
            n_unique = len(pd.unique(data))

            if n_unique <= 2:
                ftype = 'binary'
            else:
                ftype = 'categorical' # force string or mixed type into categories
        elif 'float' in data.dtype.name:
            ftype = 'numeric'
        elif 'bool' in data.dtype.name:
            ftype = 'binary'
        elif 'int' in data.dtype.name:
            n_unique = len(pd.unique(data))

            if n_unique <= 2:
                ftype = 'binary'
            elif n_unique/len(data) < 0.2:
                ftype = 'categorical'
            else:
                ftype = 'numeric'
        else:
            raise ValueError(f'Unable to infer ftype from data. Data of type {data.dtype.name} is currently not supported.')
    elif isinstance(data, (list, tuple, np.ndarray)):
        # if other data input (e.g. list, numpy array)
        data = np.array(data)
        if isinstance(data[0], str):
            ftype = 'categorical'
        elif isinstance(data[0], (np.floating, float)):
            ftype = 'numeric'
        elif isinstance(data[0], (np.bool_, bool)):
            ftype = 'binary'
        elif isinstance(data[0], (np.integer, int)):
            n_unique = len(pd.unique(data))

            if n_unique <= 2:
                ftype = 'binary'
            elif n_unique/len(data) < 0.2:
                ftype = 'categorical'
            else:
                ftype = 'numeric'
        else:
            raise ValueError(f'Unable to infer ftype from data. Data of type {type(data[0])} is currently not supported.')
    else:
        raise ValueError('Unable to infer ftype from data. Currently, only the following types are support for data input: list, tuple, numpy array and pandas series')
    
    return ftype

def infer_categories_from_data(data: Union[list, tuple, np.ndarray, pd.core.series.Series]):
    '''method to infer unique categories from data'''
    if data is None:
        raise RuntimeError('No data to infer categories. Catetories cannot be None.')
    else:
        # input check
        if not isinstance(data, (list, tuple, np.ndarray, pd.core.series.Series)) or not hasattr(data, '__iter__'):
            raise ValueError("The <data> argument needs to be of one of the following type: list, tuple, np.ndarray or pandas.core.series.Series")

        # get unique categorical values from data
        data = np.array(data)
        categories = sorted(pd.unique(data).tolist())

        return categories

def get_list_data_types(lst: list):
    return set(type(i).__name__ for i in lst)
