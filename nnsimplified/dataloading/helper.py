import random
import numpy as np
import torch
from typing import List, Union, Dict, Tuple, Optional
from ..misc import ray_fix, hash_sort
import pandas as pd
import os
from itertools import chain
from math import ceil
import pyarrow as pa
import pyarrow.dataset
from concurrent.futures import ThreadPoolExecutor


def list_batching(lst: list, batch_size: int = None, n_batch: int = None):
    ''' Function for divide a list into batches.

        Author(s): dliang1122@gmail.com

        Args
        ----------
        lst (list): a list of values
        batch_size (int): size of the batch
        n_batch (int): number of batches. If batch_size is defined, this won't have any effect.

        Returns
        -------
        list of list
    '''
    if not batch_size:
        # calculate batch_size from n_batch
        if n_batch > len(lst):
            raise Exception(
                f'Not enough data to be distributed into {n_batch} groups. Please reduce the desired number of groupings.')

        batch_size = int(len(lst) / n_batch)
    else:
        n_batch = None

    batches = [lst[i: (i + batch_size)] for i in range(0, len(lst), batch_size)]

    if n_batch:
        # to ensure equal number of elements in each batch, we might drop some elements.
        batches = batches[0:n_batch]

    return batches


def create_slice_batches(end, start = 0, slice_size: int = None, n_slice: int = None):
    ''' Function for divide a list into batches of slices.

        Author(s): dliang1122@gmail.com

        Args
        ----------
        lst (list): a list of values
        batch_size (int): size of the batch
        n_batch (int): number of batches. If batch_size is defined, this won't have any effect.

        Returns
        -------
        list of slices
    '''
    if not slice_size:
        # calculate batch_size from n_batch
        if n_slice > (end - start):
            raise Exception(f'Not enough data to be distributed into {n_slice} slices. Please reduce the desired number of slices.')

        slice_size = int((end - start) / n_slice)
    else:
        n_slice = None

    slices = []

    for i in range(start, end, slice_size):
        slice_end = i + slice_size
        slice_end = slice_end if slice_end < end else end

        slices.append(slice(i, slice_end))

    if n_slice:
        # to ensure equal number of elements in each batch, we might drop some elements.
        slices = slices[0:n_slice]

    return slices


def list_local_files(path):
    ''' Function for listing local files in a given path (dir or file)

        Author(s): dliang1122@gmail.com

        Args
        ----------
        path (str): local file or dir path

        Returns
        -------
        list of full file paths
    '''
    path = os.path.abspath(path) # get abolute path

    if os.path.isdir(path):
        return [os.path.join(dir_path, f) for dir_path, direc, files in os.walk(path) for f in files]
    elif os.path.isfile(path):
        return os.path.abspath(path)
    else:
        raise FileNotFoundError(f"No such file or directory: '{path}'")


def remove_every_nth_element(lst, n, offset=0, sort_seed=None):
    ''' funtion for removing every nth element from a list; useful for implementing cross-validation

        Author(s): dliang1122@gmail.com
    '''
    # make a copy to avoid modifying the original list

    if sort_seed is None:
        lst = lst.copy()
    else:
        lst = hash_sort(lst=lst, seed=sort_seed)

    del lst[offset::n]
    return lst


def extract_every_nth_element(lst, n, offset=0, sort_seed=None):
    ''' Funtion for extracting every nth element from a list; useful for implementing cross-validation

        Author(s): dliang1122@gmail.com
    '''

    if sort_seed is not None:
        lst = hash_sort(lst=lst, seed=sort_seed)

    return lst[offset::n]


def split_list(lst: List[int],
               split_ratio: List[Union[int, float]],
               randomize: bool = True,
               seed: int = None) -> Tuple[list]:
    ''' function to split a list into multiple segments using the split ratio provided
        e.g lst = [0,1,2,3,4,5,6,7,8,9],  split_ratio = [2,3,5] and randomize = False
            Returns [0,1], [2,3,4], [5,6,7,8,9]

        Author(s): dliang1122@gmail.com

        args
        ------
        lst (list): input list
        split_ratio (list of int or float): split ratio. E.g [1,1,1] indicates 1:1:1; will split the input list into 3 equal-sized segments
        randomize (bool): whether to randomize the input list before splitting
        seed (int): random seed for reproducible splitting

        returns
        ------
        tuple of lists
    '''
    lst = lst.copy() # don't want to modify the original list

    if randomize:
        rng = np.random.default_rng(seed)
        rng.shuffle(lst)

    # normalize split_ratio so the they add up to 1
    split_ratio = np.array(split_ratio)
    split_ratio = split_ratio/split_ratio.sum()

    # get array of upper-bound indices
    upper_indices = np.round(split_ratio.cumsum() * len(lst)).astype(int)

    # obtain intervals by adding 0 to the upper-bound array. e.g [0, 1, 3, 4] for 3 intervals (0,1), (1,3), (3,4)
    interval_indices = np.insert(upper_indices, 0, 0)

    return [lst[interval_indices[i]: interval_indices[i+1]] for i in range(len(split_ratio))]


class list_split():
    ''' class for splitting a list

        Author(s): dliang1122@gmail.com

        init args
        ------
        split_ratio (list of int or float): split ratio. E.g [1,1,1] indicates 1:1:1; will split the input list into 3 equal-sized segments
        randomize (bool): whether to randomize the input list before splitting
        seed (int): random seed for reproducible splitting
    '''
    def __init__(self,
                 split_ratio: List[Union[int, float]],
                 randomize: bool = True,
                 seed: int = None):
        self.split_ratio = split_ratio
        self.randomize = randomize
        self.seed = seed if seed else random.randint(0, 1e10)

    def split(self, lst):
        '''method for splitting a list'''

        return split_list(lst = lst,
                          split_ratio = self.split_ratio,
                          randomize = self.randomize,
                          seed = self.seed)

    def get_segment(self,
                    lst: list,
                    segment: int) -> list:
        '''method of getting a specifig segment from the splitted list
           args
           ------
           lst (list): input list
           segment (int): segment to return.
                          E.g if splitted list = ([0,1,2], [3,4], [5,6]), segments 0, 1, 2 are [0,1,2], [3,4], [5,6] respectively

           returns
           ------
           list
        '''

        splitted_list = self.split(lst)

        if segment >= len(splitted_list):
            raise ValueError('No such segment!')

        return splitted_list[segment]

    def exclude_segment(self,
                lst: list,
                segment: int) -> list:
        '''method of getting a specifig segment from the splitted list
           args
           ------
           lst (list): input list
           segment (int): segment to return.
                          E.g if splitted list = ([0,1,2], [3,4], [5,6]), segments 0, 1, 2 are [0,1,2], [3,4], [5,6] respectively

           returns
           ------
           list
        '''

        splitted_list = self.split(lst)

        if segment >= len(splitted_list):
            raise ValueError('No such segment!')

        del splitted_list[segment]

        return list(chain(*splitted_list))


class get_splitted_list_segment():
    ''' Class for getting a specific segment of a spliited list

        Author(s): dliang1122@gmail.com

        init args
        ------
        split_ratio (list of int or float): split ratio. E.g [1,1,1] indicates 1:1:1; will split the input list into 3 equal-sized segments
        keep_or_exclude (str): whether to 'keep' or 'exclude' the segment specified. 'exclude' is useful for creating train data for CV folds 
        segment (int): segment to return.
                        E.g if splitted list = ([0,1,2], [3,4], [5,6]), segments 0, 1, 2 are [0,1,2], [3,4], [5,6] respectively
        list_split_object (list_split): a list_split object that contains list-splitting process
    '''
    def __init__(self,
                 list_split_object: list_split,
                 keep_or_exclude: str,
                 segment: int):

        self.keep_or_exclude = keep_or_exclude
        self.segment = segment
        self.list_split = list_split_object
        self.split_ratio = list_split_object.split_ratio

    def __call__(self, lst):
        '''get splitted list segment'''

        if self.keep_or_exclude == 'keep':
            return self.list_split.get_segment(lst, segment= self.segment)
        elif self.keep_or_exclude == 'exclude':
            return self.list_split.exclude_segment(lst, segment= self.segment)


class sampling():
    '''Class for sampling a list in a controllable way'''
    def __init__(self, 
                 sample_rate: int|float = 1, 
                 sample_seed: Optional[int|str] = None, 
                 with_replacement = False):
        self.sample_rate = sample_rate
        self.sample_seed = sample_seed
        self.with_replacement = with_replacement

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate: int|float):
        # data check
        if not isinstance(sample_rate, (int, float)):
            raise ValueError('Sample_rate has to be a number between 0 and 1')
        elif sample_rate > 1 or sample_rate < 0:
            raise ValueError('Sample_rate has to be a number between 0 and 1')

        self._sample_rate = sample_rate

    @property
    def sample_seed(self):
        return self._sample_seed

    @sample_seed.setter
    def sample_seed(self, sample_seed: int|str):

        if sample_seed is None:
            sample_seed = random.randint(0, 1e12) # use random integer for seed if not provided

        self._sample_seed = sample_seed

    def __call__(self, lst: list) -> list:
        arr = np.array(lst)
        rng = np.random.default_rng(self.sample_seed)
        arr = rng.choice(arr,
                         replace = self.with_replacement,
                         size = ceil(len(arr) * self.sample_rate))

        return arr


def load_pa_fragment(pa_fragment: pa._dataset.FileFragment | pa._dataset.Dataset,
                     indices: List[int] = None,
                     columns: List[str] = None) -> pd.DataFrame:
    '''function for loading a pyarrow fragment into pandas dataframe

        Author(s): dliang1122@gmail.com

        Args
        ----------
        pa_fragment (pa._dataset.FileFragment or pa._dataset.Dataset): pyarrow fragment or dataset
        indices (list): list of data indices; if None, use all data
        columns (list): columns to return; if None, use all columns

        Returns
        -------
        Pandas DataFrame
    '''
    if indices is not None:
        # take only data from speficifed indices
        df = pa_fragment.take(indices = indices, columns = columns)
    else:
        df = pa_fragment.to_table(columns = columns)

    # conver to pandas; set split_blocks=True and self_destruct=True in to_pandas() method to avoid doubling memeory usage
    # see https://arrow.apache.org/docs/python/pandas.html for details
    df = df.to_pandas(split_blocks=True, self_destruct=True)

    return df


def load_pa_fragemnts(pa_fragments: List[pa._dataset.FileFragment | pa._dataset.Dataset],
                           columns: List[str] = None,
                           n_workers = 10) -> pd.DataFrame:
    '''static method for loading multiple pyarrow fragments into memory as pandas dataframe using multi-threading

        Author(s): dliang1122@gmail.com

        Args
        ----------
        pa_fragments (list): list of pyarrow fragments
        indices (list): list of data indices; if None, use all data
        n_workers (int): number of threads for multi-threaded data pull

        Returns
        -------
        Pandas DataFrame
    '''

    # Use 10 threads to pull data from pyarrow fragments
    with ThreadPoolExecutor(n_workers) as exe:
        df_futures = [exe.submit(load_pa_fragment,
                                 pa_fragment = pa_fragment,
                                 columns = columns) for pa_fragment in pa_fragments]

    return pd.concat([f.result() for f in df_futures], ignore_index = True)


def _load_pa_fragment_split(pa_fragment: pa._dataset.FileFragment | pa._dataset.Dataset,
                                 columns: List[str] = None,
                                 split_pipeline: List[get_splitted_list_segment] = []) -> pd.DataFrame:
    ''' function for loading partial data from a single pyarrow fragment with specified data splitting pipeline.

        Author(s): dliang1122@gmail.com

        Args
        ----------
        pa_fragment (path): path of the file
        columns (list): columns to return; if None, use all columns
        split_pipeline (list): list of get_splitted_list_segment objects that indicate how to split the data

        Returns
        -------
        Pandas DataFrame
    '''
    # get all indices
    data_indices = np.arange(pa_fragment.count_rows())

    # get desired indices based on splitting pipeline specfied
    if split_pipeline:
        for split in split_pipeline:
            data_indices = split(data_indices)

    # get partial data from the segment
    df = load_pa_fragment(pa_fragment = pa_fragment,
                               indices = data_indices,
                               columns = columns)

    return df


def _load_pa_fragemnts_split(pa_fragments: List[pa._dataset.FileFragment | pa._dataset.Dataset],
                                  columns: List[str] = None,
                                  split_pipeline: List[get_splitted_list_segment] = [],
                                  n_workers = 10) -> pd.DataFrame:
    '''static method for loading multiple pyarrow fragments into memory as pandas dataframe using multi-threading

        Author(s): dliang1122@gmail.com

        Args
        ----------
        pa_fragments (list): list of pyarrow fragments
        indices (list): list of data indices; if None, use all data
        n_workers (int): number of threads for multi-threaded data pull

        Returns
        -------
        Pandas DataFrame
    '''

    # Use 10 threads to pull data from pyarrow fragments
    with ThreadPoolExecutor(n_workers) as exe:
        df_futures = [exe.submit(_load_pa_fragment_split,
                                    pa_fragment = pa_fragment,
                                    columns = columns,
                                    split_pipeline = split_pipeline)
                      for pa_fragment in pa_fragments]

    return pd.concat([f.result() for f in df_futures], ignore_index = True)


def load_pyarrow_dataset(pa_dataset: pyarrow._dataset.FileSystemDataset,
                         indices: List[int] = None,
                         columns: List[str] = None) -> pd.DataFrame:
    ''' function for loading partial data from a pyarrow dataset with specified data indices.

        Author(s): dliang1122@gmail.com

        Args
        ----------
        pa_dataset (pyarrow._dataset.FileSystemDataset): pyarrow dataset
        indices (list): list of data indices; if None, use all data
        columns (list): columns to return; if None, use all columns

        Returns
        -------
        Pandas DataFrame
    '''

    if indices is not None:
        # take only data from speficifed indices
        df = pa_dataset.take(indices = indices, columns = columns)
    else:
        df = pa_dataset.to_table(columns = columns)

    # conver to pandas
    df = df.to_pandas(split_blocks=True, self_destruct=True)

    return df

def _is_nested(lst):
    '''check if nested list or tuple'''
    for elem in lst:
        if isinstance(elem, (list, tuple)):
            return True
    return False


class _sliceable_dataset():
    '''class for generating a sliceable dataset'''
    def __init__(self, data):
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        '''setter for data attribute'''
        data = self._get_data(data)

        if isinstance(data, (list, tuple)):
            # make sure each data source has the same length
            lengths = [len(i) for i in data]
            if len(set(lengths)) > 1:
                raise RuntimeError('Composite data (e.g list of datasets) found but not all datasets have the same length (number of data points!')

        self._data = data

    def _get_data(self, data):
        '''recursive function for getting data.
           If the data is nested (e.g. list of list of data), will convert the nested ones to _slicable_dataset
        '''
        if isinstance(data, (pd.DataFrame, torch.Tensor, np.ndarray, self.__class__)):
            return data
        elif isinstance(data, (list, tuple)):
            result = []
            for d in data:
                if isinstance(d, (list, tuple)):
                    result.append(self._get_data(self.__class__(d)))
                else:
                    result.append(self._get_data(d))

            return result
        else:
            raise RuntimeError('Unrecognized data format! Supported formats: pd.DataFrame, torch.Tensor, np.ndarray, List[pd.DataFrame | torch.Tensor | np.ndarray]')


    @property
    def shape(self):
        if isinstance(self.data, (pd.DataFrame, torch.Tensor, np.ndarray)):
            return self.data.shape
        elif isinstance(self.data, (list, tuple)):
            return tuple(d.shape for d in self.data)

    def __len__(self):
        if isinstance(self.data, (pd.DataFrame, torch.Tensor, np.ndarray)):
            length = len(self.data)
        else:
            length = len(self.data[0])

        return length

    def _concat(self, data1, data2):
        '''concatenate data in the 0th dimension/axis'''
        if isinstance(data1, pd.DataFrame):
            return pd.concat([data1, data2], axis = 'rows')
        elif isinstance(data1, np.ndarray):
            return np.concatenate([data1, data2], axis = 0)
        elif isinstance(data1, torch.Tensor):
            return torch.concat([data1, data2], dim = 0)
        else:
            # nested list of data
            result = []

            for i in range(len(data1)):
                result.append(self._concat(data1[i], data2[i]))

            return tuple(result)

    def __add__(self, sliceable_dataset):

        new_data = self._concat(self.data, sliceable_dataset.data)

        instance = self.__class__.__new__(self.__class__)

        instance.__init__(data = new_data)

        return instance

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def _getitem(self, data, idx):
        if isinstance(data, pd.DataFrame):
            if isinstance(idx, int):
                return data.iloc[[idx],:] # iloc with integer input produces a pandas series; write this way to force returning dataframe
            else:
                return data.iloc[idx]
        elif isinstance(data, (torch.Tensor, np.ndarray, self.__class__)):
            return data[idx]

        elif isinstance(data, (list, tuple)):
            # for composite data (e.g list of nested dataset), extract the desired data point from each nested dataset
            # e.g. _sliceable_dataset([df1, df2, df3])[0] will return [df1.iloc[0], df2.iloc[0.iloc[0]]

            result = []

            for d in data:
                result.append(self._getitem(d, idx))

            return tuple(result)

    def create_batches(self,
                       batch_size: int,
                       shuffle: bool = False,
                       shuffle_seed: int = None) -> list:
        '''method for breaking the sliceable dataset into multiple sliceable datasets of the same size
            Author(s): dliang1122@gmail.com

            Args
            ----------
            batch_size (int): number of data point in a batch
            shuffle (bool): whether to shuffle data before split
            shuffle_seed (int): random seed for shuffling data

            Returns
            -------
            List of _sliceable_dataset
        '''
        indices = np.arange(len(self)) # get array of indices

        if shuffle:
            # shuffle data if specified
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(indices)

        # create mini batches
        index_batches = list_batching(lst = indices, batch_size = batch_size)

        batches = [self[indices] for indices in index_batches]

        return batches

    def __getitem__(self, idx):
        return self._getitem(self.data, idx)

    def __repr__(self):
        repr_text = str(self.__class__) + '\n'
        repr_text += str(self.data)

        return repr_text


def sample_data(data: _sliceable_dataset | pd.DataFrame,
                sampling_rate: float = 1,
                sampling_seed: int = None,
                with_replacement = False):
    '''method for sampling a dataframe using numpy random generator for consistent sampling 
    (independent of computing platform) with random seed; uncertain how consistent pandas sample() method is'''

    if isinstance(data, pd.DataFrame):
        dataset = _sliceable_dataset(data)
    else:
        dataset = data

    data_length = len(dataset)
    rng = np.random.default_rng(sampling_seed)
    indices = rng.choice(np.arange(data_length),
                         replace = with_replacement,
                         size = ceil(data_length * sampling_rate))

    return _sliceable_dataset(dataset[indices]) if isinstance(data, _sliceable_dataset) else dataset[indices]


def _batch_sliceable_dataset(sliceable_dataset: _sliceable_dataset,
                             batch_size: int,
                             shuffle: bool = False,
                             shuffle_seed: int = None) -> list:
    '''function for breaking a sliceable dataset into multiple sliceable dataset of the same size
        Author(s): dliang1122@gmail.com

        Args
        ----------
        sliceable_dataset (_sliceable_dataset): sliceable dataset
        batch_size (int): number of data point in a batch
        shuffle (bool): whether to shuffle data before split
        shuffle_seed (int): random seed for shuffling data

        Returns
        -------
        List of _sliceable_dataset
    '''
    indices = np.arange(len(sliceable_dataset)) # get array of indices

    if shuffle:
        # shuffle data if specified
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(indices)

    # create mini batches
    index_batches = list_batching(lst = indices, batch_size = batch_size)

    batches = [sliceable_dataset[indices] for indices in index_batches]

    return batches