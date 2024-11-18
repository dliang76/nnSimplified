import time
import random
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from awstools.s3 import list_s3_objects
from awstools.helper import get_aws_credentials
from .helper import list_local_files, sample_data, _batch_sliceable_dataset, _sliceable_dataset, sampling
from .helper import load_pa_fragemnts, _load_pa_fragemnts_split
from .helper import get_splitted_list_segment, split_list, list_split, list_batching
from ..utils import flatten_list
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import Future
from copy import copy, deepcopy
import os
from typing import Union, List, Tuple, Optional, Callable
from math import ceil
import pyarrow as pa
import pyarrow.dataset
import s3fs
import re
import gc

import platform
if platform.python_version() < '3.11':
    from typing_extensions import Self
else:
    from typing import Self


class _DataLoader(ABC):
    def __init__(self,
                 data_source: Union[str, List[str]],
                 columns: List[str] = None,
                 batch_size: int = 512,
                 data_partition: Optional[str | List[str]] = None,
                 data_transformations: List[Callable] = [],
                 batch_transformations: List[Callable] = [],
                 drop_last: bool = False,
                 **kwarg):

        # store additional arguments (used for some of the methods)
        self._addtional_arguments = kwarg

        ### data source handling
        self._data_partition = [data_partition] if isinstance(data_partition, str) else data_partition
        self.data_source = data_source # get data source (e.g a directory or a list of file path(s))
        self.columns = columns # stores column used

        ### batch parameters
        self.batch_size = batch_size
        self.drop_last = drop_last

        # data transformations
        self._data_transformations = data_transformations
        self._batch_transformations = batch_transformations

        # shuffle parameters
        self._shuffle = False
        self._shuffle_seed = None # used for deterministic data shuffle

        ### initialize operational attributes
        self._memory_buffer_deque = deque() # store prefetched data
        self._minibatch_serving_deque = deque() # store mini-batches
        self._carryover = deque() # keep track of the incomplete minibatches (size < batch_size) that we are going to carryover and add to the next data load
        self._batch_number = -1 # keep track of number of batches

    @property
    def data_source(self):
        return self._data_source

    @property
    def data_source_type(self):
        return self._data_source_type

    @property
    def data_partition(self):
        return self._data_partition

    @property
    def schema(self):
        '''return schema'''
        if self.data_source_type == 'pandas.DataFrame':
            schema = self.data_source.dtypes
        elif self.data_source_type == '_sliceable_dataset':
            schema = None
        else:
            schema = self._pa_dataset.schema

        return schema

    @property
    def columns(self):

        if self._columns is None:
            if self.data_source_type == 'pandas.DataFrame':
                columns = list(self.schema.index)
            elif self.data_source_type == '_sliceable_dataset':
                columns = None
            else:
                columns = [i for i in self.schema.names if not i.startswith('__index_level')]
        else:
            columns = self._columns

        return columns

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def drop_last(self):
        return self._drop_last

    @property
    def batch_number(self):
        return self._batch_number if self._batch_number >= 0 else None

    @property
    def data_transformations(self):
        return self._data_transformations

    @property
    def batch_transformations(self):
        return self._batch_transformations

    @data_source.setter
    @abstractmethod
    def data_source(self, data_source: Union[str, List[str]]):
        '''method for handling data source'''
        pass

    @columns.setter
    def columns(self, columns):
        # value check
        if not isinstance(columns, (str, list, type(None))):
            raise ValueError('"columns" arg has to be either a str or a list of str.')

        if columns is None:
            self._columns = None
        else:
            if isinstance(self.data_source, pd.DataFrame):
                all_columns = self._data_source.columns.to_list()
            else:
                all_columns = [name for name in self.schema.names if not re.match('__index_level_[0-9]+__', name)]

            if set(columns).issubset(set(all_columns)): # check if columns specified are in the data
                self._columns = columns

            else:
                invalid_cols = set(columns) - set(all_columns)

                raise ValueError(f'Column(s) {invalid_cols} not found in data.')

    @batch_size.setter
    def batch_size(self, batch_size: int):
        # data check
        if (not isinstance(batch_size, int)) or (batch_size <= 0):
            raise ValueError('batch_size has to be a positive integer.')

        self._batch_size = batch_size

    @drop_last.setter
    def drop_last(self, drop_last):
        if not isinstance(drop_last, bool):
            raise ValueError('drop_last has to be of boolean type (True or False). ')

        self._drop_last = drop_last

    @abstractmethod
    def _load_data_into_memory(self, source) -> _sliceable_dataset:
        '''method for loading data into memory'''
        pass

    def _fill_memory_buffer(self):
        '''method for filling memory buffer deque with data'''
        try:
            # use thread for background loading (non-blocking)
            exe = ThreadPoolExecutor(1)

            self._memory_buffer_deque.append(exe.submit(self._load_data_into_memory))

        except Exception as e:
            exe.shutdown(wait = False, cancel_futures= True)
            raise e
        finally:
            exe.shutdown(wait = False)

    @staticmethod
    def _transform(data, transformations: list):
        '''method for transforming data'''
        data = copy(data)

        for tr in transformations:
            data = tr(data)

        return data

    def _create_minibatches(self, dataset):
        '''method for generating mini-batch data from pyarrow fragment pool'''

        # break data into data batches; perform second stage shuffle before batching if self._shuffle = True
        mini_dataset_batches = dataset.create_batches(batch_size = self._batch_size,
                                                      shuffle = self._shuffle,
                                                      shuffle_seed = self._shuffle_seed)

        # if the last mini-batch is not complete (smaller than the desired batch size),
        # store it in _carryover so we can add to the top of the next batch
        if len(mini_dataset_batches[-1]) < self._batch_size:
            self._carryover.append(mini_dataset_batches.pop())

        return mini_dataset_batches # This returns a list of _sliceable_dataset

    def __iter__(self):
        '''make object iterable'''
        return self

    @abstractmethod
    def __next__(self):
        pass

    def head(self, n = 5):
        '''method for getting first n batches'''
        result = []

        for i in range(n):
            try:
                batch = next(self)
                result.append(batch)
            except StopIteration:
                break

        self.reset_iteration()

        return result

    def shuffle(self, seed: int | None = None):
        '''method of storing info used for shuffling data.'''

        if not (isinstance(seed, int) or (seed is None)):
            raise ValueError('seed has to be either an integer or None.')

        # turn shuffle on; set shuffle_seed
        self._shuffle = True
        self._shuffle_seed = seed

        # reset iterator
        self.reset_iteration()

    @abstractmethod
    def reset_iteration(self):
        pass

    def reset(self):
        '''method for re-initialize necessary structure for iteration.'''

        self._shuffle = False
        self._shuffle_seed = None

        # reset operational deque
        self.reset_iteration()

    @abstractmethod
    def copy(self):
        pass

    def __len__(self):
        '''method for getting the number of batches in the loader; ignore sampling for now'''
        self.reset_iteration()
        total_batches = sum([1 for i in self])

        return total_batches

    def collect(self, combine_batches = False):
        '''method for collection all batches to a list; might run into memory issue for large data; use with caution'''

        # reset batch iteration
        self.reset_iteration()

        if combine_batches:
            result = sum([_sliceable_dataset(data) for data in self]).data
        else:
            result = [data for data in self]

        return result

    @abstractmethod
    def sample(self, sample_rate: float, seed: int | None = None, with_replacement: bool = False) -> Self:
        '''sample source data and return a new data loader'''
        pass

    def split(self,
              split_ratio: List[float | int],
              split_method: str = 'data',
              randomize: bool = True,
              random_seed: int = None) -> Tuple[Self]:
        '''method for splitting a dataloader into multiple dataloader with split_ratio specified
           E.g. For split_ratio = (1, 1, 2), this method will return 3 dataloaders with 25%, 25% and 50% of the original data
        '''

        if isinstance(self.data_source, pd.DataFrame):
            split_func = _dataloader_split_memory_lvl

        elif split_method == 'data':
            split_func = _dataloader_split_data_lvl

        elif split_method == 'file':
            split_func = _dataloader_split_partition_lvl

        else:
            raise ValueError("split_method has to be either 'data' or 'file'.")

        splitted = split_func(dataloader = self,
                              split_ratio = split_ratio,
                              randomize = randomize,
                              random_seed = random_seed)

        return splitted

    def splitCV(self,
                n_fold: int = 3,
                randomize: bool = True,
                random_seed: int = None) -> Tuple[Tuple[Self]]:
        '''method for splitting a dataloader into multiple dataloader for n-fold CV
           E.g. For n_fold = 3, this method will return 3 pairs for train and test dataloaders
        '''
        if isinstance(self.data_source, pd.DataFrame):
            splitCV_func = _dataloader_splitCV_memory_lvl

        else:
            splitCV_func =  _dataloader_splitCV_data_lvl

        splitted = splitCV_func(dataloader = self,
                                n_fold = n_fold,
                                randomize = randomize,
                                random_seed = random_seed)

        return splitted


class FileLoader(_DataLoader):
    '''
    Class for generating data batchs from files. Creates an iterable instance.

    Author(s): dliang1122@gmail.com

    init args
    ----------
    data_source (str or list(str)): data directory or a list of file paths; supported file format: .csv or .parquet
    columns (List(str)): columns to load. If none, load all columns.
    batch_size (int): number of data points in a batch for batch processing
    data_partition (str, list): Default: None. partitions used to group the data; use Hive partition mechanism (e.g '.../partition=1/...).
                                E.g. Setting data_partition = 'group' will group all the files with the same 'group' value in the file path together.
                                     If None, each partition is a single file.
    n_part_per_fetch (int): The number of partitions (1 partition = an individual file or a group of files; see data_partition above) to load into a memory_buffer_deque slot using parallel processing.
                            Shuffling/sampling operation is performed on all the data in a memory buffer. Thus, n_part_per_fetch will increase the randomness of these operation.
                            In general, increasing the number will
                            1) increase randomness of the data for shuffling or sampling operation.
                            2) improve dataloading as the system need not load data frequently
                            3) increase memory usage
    data_transformations (list of callable): chained transformations performed during data loading. If data_partition is specified, these transformations will be performed at the partition level; otherwise, the transformations will be performed for each data load.
                                             These transformations are pre-batching (performed before breaking data into mini-batches) and act upon individual data partition only (not across data partitions);
                                             e.g. if we want to have a transformation to calculate some stats (e.g. mean, max), these stats will be calculated using only data in a partition.
    batch_transformations (list of callable): chained transformations for mini-batches. These transformations are post-batching and act upon individual mini-batch.
    drop_last (bool): whether to drop the last data batch that is incomplete (size < specified batch_size).
                      E.g. When breaking 4096 data points into batches of size 100, the last batch will only have 96 data points. drop_last = True will drop this last batch with 96 data points.

    return
    --------
    FileLoader object
    '''
    def __init__(self,
                 data_source: Union[str, List[str]],
                 columns: List[str] = None,
                 batch_size: int = 512,
                 data_partition: Optional[str | List[str]] = None,
                 n_part_per_fetch: int = 5,
                 data_transformations: List[Callable] = [],
                 batch_transformations: List[Callable] = [],
                 drop_last: bool = False,
                 **kwarg):

        ### set pyarrow resources. Tried multiple settings; read performance plateaued at n_cpu ~ 10
        pa.set_cpu_count(min(10, os.cpu_count() - 1))
        pa.set_io_thread_count(min(10, (os.cpu_count() - 1) * 3))

        super().__init__(data_source = data_source,
                         columns = columns,
                         batch_size = batch_size,
                         data_partition = data_partition,
                         data_transformations = data_transformations,
                         batch_transformations = batch_transformations,
                         drop_last = drop_last,
                         **kwarg)

        # get files from data_source
        print('Collecting files... ', end = '')
        self._files = self._get_files()
        self._file_format = self._get_file_format()
        print(f'{len(self._files)} file{"s" if len(self._files)>1 else ""} found.')

        # create pyarrow objects; we'll utilize pyarrow for fast dataloading
        self._create_pyarrow_object()
        # store pyarrow fragment futures to be used for data loading for out-of-core data loading
        self._create_pa_fragment_deque()

        # dataloading setting
        self.n_part_per_fetch = n_part_per_fetch

        # initialize pipeline for splitting data
        self._split_pipeline = [] # store how we want to split the data
        self._split_data_portion = 1

    @property
    def data_source(self):
        return self._data_source

    @property
    def file_format(self):
        return self._file_format

    @property
    def files(self):
        return self._files

    @property
    def n_part_per_fetch(self):
        return self._n_part_per_fetch

    def __del__(self):

        gc.collect()

    def get_data_source_type(self, data_source: Union[str, List[str]]):
        '''method for getting data source type (S3, local_files)'''
        if isinstance(data_source, (list, tuple)):
            # extract info from the first element if given a list
            data_source = data_source[0]

        if isinstance(data_source, str):
            path_name = data_source

        elif isinstance(data_source, (list, tuple)):
            path_name = data_source[0] # only check the first file path

        else:
            raise ValueError('data_source has to be one of the following: a directory or file path (string), a list of file paths (list).')

        if 's3:' in path_name:
            source_type = 's3'

            # get aws credential
            if 'aws_credential' in self._addtional_arguments:
                self._aws_credentials = self._addtional_arguments['aws_credential']
            else:
                # obtain aws_credentials from AWS environment
                self._aws_credentials = get_aws_credentials()

        elif 'hdfs:' in path_name:
            source_type = 'hdfs'
        else:
            source_type = 'local_files'

        return source_type

    @data_source.setter
    def data_source(self, data_source: Union[str, List[str]]):
        self._data_source_type = self.get_data_source_type(data_source)

        self._data_source = data_source

    def _get_files(self):
        '''method for getting the list of files from a data source.'''
        if isinstance(self.data_source, (list, tuple)):
            files = self.data_source

        elif self.data_source_type == 's3':
            files = [f for f in list_s3_objects(self.data_source, aws_credential = self._aws_credentials)
                     if '.parquet' in f]

        elif self.data_source_type == 'hdfs':
            raise Exception('Not Implemented yet!')

        elif self.data_source_type == 'local_files':

            files = [f for f in list_local_files(self.data_source) if '.parquet' in f or '.csv' in f]

        return files

    def _get_file_format(self):
        '''method for inferring file format (csv or parquet)'''
        if self.files[0].endswith('.csv'):
            return 'csv'
        elif self.files[0].endswith('.parquet'):
            return 'parquet'
        else:
            # assume parquet if no extension found
            return 'parquet'

    def _get_partitioned_files(self):
        '''method of getting directories for each data_partition'''
        if self.data_partition:
            partitioned_files = dict()
            for file in self.files:
                key = tuple(re.search(f'/{part}=(.*?)/', file).group(0) for part in self.data_partition)
                if key in partitioned_files:
                    partitioned_files[key].append(file)
                else:
                    partitioned_files[key] = [file]

            return partitioned_files

    def _create_pyarrow_object(self):
        '''method for creating pyarrow objects (pyarrow datasets and fragments) from files'''

        # extract base directory
        # e.g. /dir/group1=0/group2=0 -> /dir/
        partition_base_dir = re.search('/[^=]+/', self.files[0]).group(0)

        if self._data_partition:
            # if using data partiotions, re-order partition strings so that they match the order of appearance in the file path
            self._data_partition = [i[0].replace('=','') for i in re.finditer('|'.join([f'{i}=' for i in self._data_partition]), self.files[0])]

            # group data based on the partitions provided. Create 1 pyarrow dataset for each partition (can contain multiple files).
            self._pa_data_partition = [self._create_pa_dataset_from_source(source = files,
                                                                           format = self._file_format,
                                                                           partition_base_dir = partition_base_dir,
                                                                           partitioning = pa.dataset.partitioning(flavor = 'hive')) # use hive partition format
                                       for partition, files in self._get_partitioned_files().items()]

            # create a pyarrow dataset for the entire data by combining pyarrow datasets from each partition.
            self._pa_dataset = pa.dataset.dataset(source = self._pa_data_partition)

        else:
            # if no partition given, create a pyarrow dataset for the entire data from files
            self._pa_dataset = self._create_pa_dataset_from_source(source = self.files,
                                                                   format = self._file_format,
                                                                   partition_base_dir = partition_base_dir,
                                                                   partitioning = pa.dataset.partitioning(flavor = 'hive')) # use hive partition format

            # self._pa_data_partition = list(self._pa_dataset.get_fragments())
            # no data grouping. Store pyarrow fragments as pyarrow FileSystemDataset (each fragment = 1 file) to ensure partition info extraction
            schema = self._pa_dataset.schema
            format = self._pa_dataset.format
            self._pa_data_partition = [pa._dataset.FileSystemDataset([frag], schema = schema, format = format)  for frag in self._pa_dataset.get_fragments()]

    @n_part_per_fetch.setter
    def n_part_per_fetch(self, n_part_per_fetch: int | None):
        '''setter with data check'''

        # data check
        if (not isinstance(n_part_per_fetch, int)) or (n_part_per_fetch <= 0):
            raise ValueError('n_part_per_fetch has to be a positive integer.')

        self._n_part_per_fetch = n_part_per_fetch

    def _create_pa_dataset_from_source(self,
                                       source,
                                       format = None,
                                       partition_base_dir = None,
                                       partitioning = None) -> pa._dataset.FileSystemDataset:
        '''Create pyarrow dataset from a list of files to gain quick access to schema and metadata; data are not loaded into memory at this point.'''

        if self._data_source_type == 's3':
            # initiate s3 file system for pyarrow
            filesystem = s3fs.S3FileSystem(key = self._aws_credentials['access_key'],
                                           secret = self._aws_credentials['secret_key'],
                                           token = self._aws_credentials['token'])
        elif self._data_source_type == 'hdfs':
            raise Exception('Not Implemented yet!')
        else:
            filesystem = None

        # load the files as a pyarrow dataset
        pa_dataset = pa.dataset.dataset(source = source,
                                        filesystem = filesystem,
                                        format = format,
                                        partition_base_dir = partition_base_dir,
                                        partitioning = partitioning)

        return pa_dataset

    @staticmethod
    def _scan_fragment_metadata(frag_or_dataset, file_format):
        '''helper function for pyarrow fragment or dataset scan; once scanned, the fragment/dataset's metadata is cached and can be retrieve with almost no computation cost.
           This is only for parquet and not for csv files as csv has no metadata.
           Note: Make this a static method to avoid circulating referencing problem.
        '''
        # do a simple operation on the fragment or dataset once to gain faster access
        if file_format == 'parquet':
            frag_or_dataset.count_rows() # use count_rows() operation to force metadata reading

        return frag_or_dataset

    def _create_pa_fragment_deque(self):
        '''method for filling self._pa_fragment_deque using multi-threading
        '''
        self._pa_fragment_deque = deque(self._pa_data_partition.copy()) # don't want to alter the order of original partition data

        # shuffle data
        if self._shuffle:
            # shuffle pyarrow fragments; first stage shuffle
            rng = np.random.default_rng(self._shuffle_seed)
            rng.shuffle(self._pa_fragment_deque)

        try:
            # scan through pyarrow fragments; use 5 thread for non-blocking background operation (Due to GIL, using more thread will slow down the data loader)
            exe = ThreadPoolExecutor(20)
            [exe.submit(self._scan_fragment_metadata,
                        frag_or_dataset = d,
                        file_format = self.file_format)
                        for d in self._pa_fragment_deque]

        except Exception as e:
            print(f"Exception: {e}")
            # shutdown to stop accepting takes and cancel all unfinished tasks/threads
            exe.shutdown(wait = False, cancel_futures = True)
        finally:
            exe.shutdown(wait=False)

    def _load_data_into_memory(self) -> _sliceable_dataset:
        '''method for loading pyarrow fragments or a pyarrow dataset into memory as pandas dataframe'''

        n_parts = min(self._n_part_per_fetch, len(self._pa_fragment_deque))

        frags = [self._pa_fragment_deque.popleft() for i in range(n_parts)]

        # read pyarrow fragments into memory
        n_workers = 10
        if self._split_pipeline:
            data = _load_pa_fragemnts_split(pa_fragments = frags,
                                            columns = self._columns,
                                            split_pipeline = self._split_pipeline,
                                            n_workers = n_workers)
        else:
            data = load_pa_fragemnts(pa_fragments = frags,
                                     columns = self._columns,
                                     n_workers = n_workers)

        # transform data if specified
        if self.data_transformations:
            # data = self._transform(data = data, transformations = self.data_transformations)

            if self._data_partition:
                data = data.groupby(self.data_partition, observed = True)\
                           .apply(lambda x: _sliceable_dataset(
                                            self._transform(data = x, transformations = self.data_transformations)
                                                              )
                                 )
                data = sum(data) # combine all _sliceable_dataset
            else:
                data = _sliceable_dataset(self._transform(data = data, transformations = self.data_transformations))
        else:
            data = _sliceable_dataset(data)

        return data

    @staticmethod
    def _transform(data, transformations: list):
        '''method for transforming data'''
        data = copy(data)

        for tr in transformations:
            data = tr(data)

        return data

    def _return_next_batch(self):
        '''method for returning the next mini-batch'''
        # fill memory buffers in the beginning of the run (batch_number = -1)
        if self.batch_number is None:
            self._fill_memory_buffer() # fill memory buffer

        # create mini-batches and fill minibatch_serving_deque for serving mini-batches
        while len(self._minibatch_serving_deque) < 50 and len(self._memory_buffer_deque) > 0:
            # trigger minibatch generation machanism if the number of minibatches falls below a certain amount and if there is any data left in the memory_buffer_deque
            dataset = self._memory_buffer_deque.popleft().result() # get dataset from the memory buffer

            if len(self._pa_fragment_deque) > 0:
                self._fill_memory_buffer() # replenish buffer to maintain constant memory buffer size

            if self._carryover:
                # add the incomplete minibatch (< than desired batch size) from the previous dataset to the current dataset
                dataset = self._carryover.pop() + dataset

            minibatches =self._create_minibatches(dataset = dataset) # create mini-batches from the dataset
            self._minibatch_serving_deque.extend(minibatches) # append mini-batches to minibatch_serving_deque

        if len(self._memory_buffer_deque) == 0:
            # add the remaining incomplete data (size < batch_size) in the carryover to the minibatch_serving_deque if
            # (1) not more data in the memory buffer,
            # (2) and we don't want to drop the last incomplete minibatch (drop_last == False)
            if self._carryover and not self._drop_last:

                self._minibatch_serving_deque.append(self._carryover.pop())

        # stop iteration if no more mini-batches left
        if len(self._minibatch_serving_deque) == 0:
            self.reset_iteration()
            raise StopIteration

        # serve mini-batches
        minibatch = self._minibatch_serving_deque.popleft()
        self._batch_number += 1 # update batch number

        if isinstance(minibatch, Future):
            minibatch = minibatch.result()

        return minibatch

    def __next__(self):
        '''method for getting next mini-batch'''

        minibatch =  self._return_next_batch()

        # transform minibatch if batch transformation methods given
        if self._batch_transformations:
            minibatch = self._transform(data = minibatch, transformations=self._batch_transformations)

        if isinstance(minibatch, pd.DataFrame):
            start_index = self.batch_number * self.batch_size
            end_index = start_index + len(minibatch)
            minibatch.index = np.arange(start_index, end_index)

        return minibatch

    def reset_iteration(self):
        '''method for resetting batch iteration'''
        # reset operational deque
        self._create_pa_fragment_deque()

        self._memory_buffer_deque = deque()
        self._minibatch_serving_deque = deque()
        self._carryover = deque()
        self._batch_number = -1

        gc.collect()

    def copy(self):
        '''method for cloning the object with the same settings'''

        # create a new empty instance
        instance = self.__class__.__new__(self.__class__)

        # copy settings over to the new instance
        for k,v in self.__dict__.items():
            if k in ('_pa_fragment_deque', '_memory_buffer_deque', '_carryover', '_minibatch_serving_deque'):
                # do not copy transient data
                instance.__dict__[k] = deque()
            else:
                instance.__dict__[k] = copy(v)

        # reset to initial state
        instance.reset()

        return instance

    def __repr__(self):
        return '\n'.join([str(self.__class__)]+['Settings:']+[f'{k}: {v}' for k,v in self.__dict__.items() if k not in ['_data_source', '_files',
                                                 '_pa_dataset', '_pa_data_partition','_pa_fragment_deque',
                                                 '_scan_executor', '_memory_buffer_deque', '_minibatch_serving_deque', '_carryover',
                                                 '_batch_number']])

    def sample(self, sample_rate: float, seed: int | None = None, with_replacement: bool = False) -> Self:
        '''sample source data and return a new data loader'''

        # data check
        if  (not isinstance(sample_rate, (float,int))) or (sample_rate > 1) or (sample_rate <=0):
            raise ValueError('sample_rate has to be a number between 0 and 1.')

        if not (isinstance(seed, int) or (seed is None)):
            raise ValueError('seed has to be either an integer or None.')

        dl_sampled = self.copy()

        sample = sampling(sample_rate= sample_rate, sample_seed = seed, with_replacement = with_replacement)

        dl_sampled._split_pipeline.append(sample)
        dl_sampled._split_data_portion *= sample_rate

        dl_sampled.reset()

        return dl_sampled


class inCoreDataLoader(_DataLoader):
    '''
    Class for generating fixed-size data batches from data in memory. Creates an iterable instance.

    Author(s): dliang1122@gmail.com

    init args
    ----------
    data_source (pd.DataFrame): source of data
    columns (List(str)): columns to load. If none, load all columns.
    batch_size (int): number of data points in a batch for batch processing
    data_partition (str, list): Partitions used to group the data. data_transformations are performed at the partition level;
                                if data_partition is None, data_transformations are performed on the entire dataset; .Default: None.
    data_transformations (list of callable): chained transformations for data. If data_partition is specified, the transformations are applied at the partition level; otherwise, the transformation are applied to the entire dataset.
                                                These transformations are pre-batching (performed before breaking data into mini-batches) and act upon individual data partition only (not across data partitions);
                                                e.g. if we want to have a transformation to calculate some stats (e.g. mean, max), these stats will be calculated using only data in a partition.
    batch_transformations (list of callable): chained transformations for mini-batches. These transformations are post-batching and act upon individual mini-batch.
    drop_last (bool): whether to drop the last data batch that is incomplete (size < specified batch_size).
                      E.g. When breaking 4096 data points into batches of size 100, the last batch will only have 96 data points. drop_last = True will drop this last batch with 96 data points.

    return
    --------
    inCoreDataLoader object
    '''
    def __init__(self,
                 data_source: pd.DataFrame | _sliceable_dataset,
                 columns: List[str] = None,
                 data_partition: Optional[str | List[str]] = None,
                 batch_size: int = 512,
                 data_transformations: List[Callable] = [],
                 batch_transformations: List[Callable] = [],
                 drop_last: bool = False,
                 **kwarg):

        super().__init__(data_source = data_source,
                         columns = columns,
                         batch_size = batch_size,
                         data_partition = data_partition,
                         data_transformations = data_transformations,
                         batch_transformations = batch_transformations,
                         drop_last = drop_last,
                         **kwarg)

        # initial data fill
        self._fill_memory_buffer() # fill memory buff

    @staticmethod
    def get_data_source_type(data_source):
        '''get source type using recursion'''
        if isinstance(data_source, (list, tuple)):
            source_type_list = [inCoreDataLoader.get_data_source_type(d) for d in data_source]
            all_source_types = set(flatten_list(source_type_list))
            if len(all_source_types) > 1:
                raise ValueError('Multiple source types found. Currently, only pandas.DataFrame is supported.')
            else:
                return source_type_list[0]

        elif isinstance(data_source, pd.DataFrame):
            return 'pandas.DataFrame'
        
        elif isinstance(data_source, _sliceable_dataset):
            return '_sliceable_dataset'

        else:
            raise ValueError('data_source has to be a pandas dataframe.')

    @property
    def data_source(self):
        return self._data_source

    @data_source.setter
    def data_source(self, data_source: Union[str, List[str], pd.DataFrame]):
        '''method for determining the source of files'''

        self._data_source_type = self.get_data_source_type(data_source)

        self._data_source = data_source

    def _load_data_into_memory(self) -> _sliceable_dataset:

        if self.columns:
            data = self._data_source[self.columns]
        else:
            data = self._data_source

        if self.data_transformations:
            if isinstance(data, pd.DataFrame):
                if self._data_partition:
                    data = data.groupby(self.data_partition, observed = True)\
                            .apply(lambda x: _sliceable_dataset(
                                                    self._transform(data = x, transformations = self.data_transformations)
                                                                            )
                                                )
                    data = sum(data) # combine all _sliceable_dataset
                else:
                    data = _sliceable_dataset(self._transform(data = data, transformations = self.data_transformations))
            elif isinstance(data, _sliceable_dataset):
                data = self._transform(data = data, transformations = self.data_transformations)
        
        if not isinstance(data, _sliceable_dataset):
            data = _sliceable_dataset(data)

        return data

    def _return_next_batch(self):

        # create all mini-batches at the beginning of the operation
        if self.batch_number is None:
            dataset = self._memory_buffer_deque[0]
            if isinstance(dataset, Future):
                dataset = dataset.result()

            mini_batches = self._create_minibatches(dataset = dataset)
            self._minibatch_serving_deque.extend(mini_batches)

        # if mini-batches deque is empty
        if len(self._minibatch_serving_deque) == 0:
            # check if there are data in _carryover deque
            if self._carryover and not self._drop_last:
                self._minibatch_serving_deque.append(self._carryover.pop())
            else:
                # stop if no more mini-batches left
                self.reset_iteration()
                raise StopIteration

        self._batch_number += 1 # increment batch number

        # serve mini-batch
        minibatch = self._minibatch_serving_deque.popleft()

        if isinstance(minibatch, Future):
            minibatch = minibatch.result()

        return minibatch

    def __next__(self):
        '''method for getting next mini-batch'''

        minibatch =  self._return_next_batch()

        # transform minibatch if batch transformation methods given
        if self._batch_transformations:
            minibatch = self._transform(data = minibatch, transformations=self._batch_transformations)

        if isinstance(minibatch, pd.DataFrame):
            start_index = self.batch_number * self.batch_size
            end_index = start_index + len(minibatch)
            minibatch.index = np.arange(start_index, end_index)

        return minibatch

    def reset_iteration(self):
        # reset operational deque

        self._minibatch_serving_deque = deque()
        self._batch_number = -1

        gc.collect()

    def copy(self):
        '''method for cloning the object with the same settings; no data copy unless in_core = True'''

        return self.__class__(data_source = self.data_source,
                              columns = self._columns,
                              data_partition = self._data_partition,
                              batch_size = self.batch_size,
                              data_transformations = self.data_transformations,
                              batch_transformations = self.batch_transformations,
                              drop_last = self.drop_last,
                              **self._addtional_arguments)

    def __repr__(self):
        return '\n'.join([str(self.__class__)]+['Settings:']+[f'{k}: {v}' for k,v in self.__dict__.items() if k not in ['_data_source', '_memory_buffer_deque', '_minibatch_serving_deque', '_batch_number']])

    def sample(self, sample_rate: float, seed: int | None = None, with_replacement: bool = False) -> Self:
        '''sample source data and return a new data loader'''

        # data check
        if  (not isinstance(sample_rate, (float,int))) or (sample_rate > 1) or (sample_rate <=0):
            raise ValueError('sample_rate has to be a number between 0 and 1.')

        if not (isinstance(seed, int) or (seed is None)):
            raise ValueError('seed has to be either an integer or None.')

        if self._data_partition:
            # if using partition, do stratified sampling
            data_source = self.data_source.groupby(self.data_partition, observed = True)\
                                          .apply(lambda x: sample_data(data = x,
                                                                       sampling_rate = sample_rate,
                                                                       sampling_seed = seed,
                                                                       with_replacement = with_replacement))\
                                          .reset_index(drop = True)
        else:
            data_source = sample_data(data = self.data_source,
                                      sampling_rate = sample_rate,
                                      sampling_seed = seed,
                                      with_replacement = with_replacement)

        dl_sampled = self.__class__(data_source = data_source,
                                    columns = self._columns,
                                    data_partition = self._data_partition,
                                    batch_size = self.batch_size,
                                    data_transformations = self.data_transformations,
                                    batch_transformations = self.batch_transformations,
                                    drop_last = self.drop_last,
                                    **self._addtional_arguments)

        return dl_sampled


def _dataloader_split_memory_lvl(dataloader: inCoreDataLoader,
                                 split_ratio: List[Union[int, float]],
                                 randomize: bool = True,
                                 random_seed: int = None):
    ''' Function for splitting a dataloader at memory level
        Author(s): dliang1122@gmail.com

        Args
        ------
        dataloader (inCoreDataLoader): dataloader object
        split_ratio (list of int or float): split ratio. E.g [1,1,1] indicates 1:1:1; will split the input list into 3 equal-sized segments
        randomize (bool): whether to randomize the input list before splitting
        random_seed (int): random seed for reproducible splitting

        returns
        ------
        list of dataloaders
    '''
    # split pyarrow fragments according to ratio specified
    splitted_indices = split_list(lst = np.arange(len(dataloader.data_source)),
                                  split_ratio = split_ratio,
                                  randomize = randomize,
                                  seed = random_seed)

    # total number of resulting dataloaders
    n_dataloaders = len(split_ratio)

    # track resulting dataloaders
    splitted = []

    for i in range(n_dataloaders):

        if isinstance(dataloader._data_source, pd.DataFrame):
            data_source = dataloader._data_source.iloc[splitted_indices[i]]
        elif isinstance(dataloader._data_source, _sliceable_dataset):
            data_source = dataloader._data_source[splitted_indices[i]]

        dl = inCoreDataLoader(data_source = data_source,
                              columns = dataloader._columns,
                              data_partition = dataloader._data_partition,
                              batch_size = dataloader.batch_size,
                              data_transformations = dataloader.data_transformations,
                              batch_transformations = dataloader.batch_transformations,
                              drop_last = dataloader.drop_last,
                              **dataloader._addtional_arguments)

        splitted.append(dl)

    return tuple(splitted)


def _dataloader_splitCV_memory_lvl(dataloader: inCoreDataLoader,
                                   n_fold: int = 3,
                                   randomize: bool = True,
                                   random_seed: int = None) -> Tuple[Tuple[inCoreDataLoader]]:
    ''' Function for performing n-fold CV split at memory level

        Author(s): dliang1122@gmail.com

        args
        ------
        dataloader (inCoreDataLoader): dataloader object
        n_fold (int): number of folds to create
        randomize (bool): whether to randomize the input list before splitting
        random_seed (int): random seed for reproducible splitting

        returns
        ------
        list of dataloader train-test pairs (onr pair for each fold)
    '''

    # splitting data using equal weights
    split_ratio = np.ones(n_fold)

    # determin how to split
    list_split_object = list_split(split_ratio = split_ratio, randomize = randomize, seed = random_seed)

    # total number of resulting dataloader train-test pairs
    n_dataloaders = len(split_ratio)

    all_data_indices = np.arange(len(dataloader.data_source))

    # track resulting dataloader train-test pairs
    splitted_pairs = []

    for fold in range(n_dataloaders):
        # determine how to split and which segment to return

        fold_dls = [] # store train and test dataloader

        for i in ('train', 'test'):

            # want to exclude the segment for train and keep only the segment for test.
            # E.g. for 3-fold, we split the data 3 ways; keep 2 (i.e remove 1 segment) for train and 1 for test
            keep_or_exclude_segment = 'exclude' if i == 'train' else 'keep'

            split_method = get_splitted_list_segment(list_split_object = list_split_object,
                                                     keep_or_exclude = keep_or_exclude_segment,
                                                     segment = fold)

            indices = split_method(all_data_indices)

            if isinstance(dataloader._data_source, pd.DataFrame):
                data_source = dataloader._data_source.iloc[indices]
            elif isinstance(dataloader._data_source, _sliceable_dataset):
                data_source = dataloader._data_source[indices]

            dl = inCoreDataLoader(data_source = data_source,
                            columns = dataloader._columns,
                            data_partition = dataloader._data_partition,
                            batch_size = dataloader.batch_size,
                            data_transformations = dataloader.data_transformations,
                            batch_transformations = dataloader.batch_transformations,
                            drop_last = dataloader.drop_last,
                            **dataloader._addtional_arguments)

            fold_dls.append(dl)

        splitted_pairs.append(fold_dls)

    return tuple(splitted_pairs)

def _dataloader_split_partition_lvl(dataloader: FileLoader,
                               split_ratio: List[Union[int, float]],
                               randomize: bool = True,
                               random_seed: int = None) -> Tuple[FileLoader]:
    ''' function for splitting a dataloader at file level

        Author(s): dliang1122@gmail.com

        args
        ------
        dataloader (FileLoader): FileLoader object
        split_ratio (list of int or float): split ratio. E.g [1,1,1] indicates 1:1:1; will split the input list into 3 equal-sized segments
        randomize (bool): whether to randomize the input list before splitting
        random_seed (int): random seed for reproducible splitting

        returns
        ------
        list of dataloaders
    '''

    pa_data_partitions = np.array(dataloader._pa_data_partition.copy()) # get pq_fragments from the original dataloader

    # shuffle pq_fragments if shuffle seed found
    if dataloader._shuffle_seed:
        rng = np.random.default_rng(dataloader._shuffle_seed)
        rng.shuffle(pa_data_partitions)

    # split pyarrow fragments according to ratio specified
    splitted_indices = split_list(lst = np.arange(len(pa_data_partitions)),
                                  split_ratio = split_ratio,
                                  randomize = randomize,
                                  seed = random_seed)

    splitted = []

    for i in range(len(split_ratio)):
        fl = dataloader.copy() # make a copy of the original dataloader
        fl._pa_data_partition = deque(pa_data_partitions[splitted_indices[i]])
        fl._files = list(np.array(fl.files)[splitted_indices[i]])

        if isinstance(fl._pa_data_partition, pa._dataset.Dataset):
            fl._pa_dataset = pa.dataset.dataset(fl._pa_data_partition)
        else:
            fl._pa_dataset = pa.dataset.FileSystemDataset(fragments = fl._pa_data_partition,
                                                          schema = dataloader._pa_dataset.schema,
                                                          format = dataloader._pa_dataset.format)

        fl._shuffle_seed = None
        fl.reset()

        splitted.append(fl)

    return tuple(splitted)

def _dataloader_split_data_lvl(dataloader: FileLoader,
                               split_ratio: List[Union[int, float]],
                               randomize: bool = True,
                               random_seed: int = None) -> Tuple[FileLoader]:
    ''' Function for splitting a dataloader at partition level

        Author(s): dliang1122@gmail.com

        Args
        ------
        dataloader (FileLoader): dataloader object
        split_ratio (list of int or float): split ratio. E.g [1,1,1] indicates 1:1:1; will split the input list into 3 equal-sized segments
        randomize (bool): whether to randomize the input list before splitting
        random_seed (int): random seed for reproducible splitting
        adjust_load (bool): whether to adjust number of batches to load into memory to counter reduced data size from individual file

        returns
        ------
        list of dataloaders
    '''
    list_split_object = list_split(split_ratio, randomize = randomize, seed = random_seed)

    # total number of resulting dataloaders
    n_dataloaders = len(split_ratio)

    # track resulting dataloaders
    splitted = []

    for target_segment in range(n_dataloaders):
        # determine how to split and which segment to return
        split_method = get_splitted_list_segment(list_split_object = list_split_object,
                                                 keep_or_exclude = 'keep',
                                                 segment = target_segment)

        # make a copy of the original dataloader
        dl = dataloader.copy()

        # add split method to the pipeline
        dl._split_pipeline.append(split_method)

        # keep track of portion of data retained
        portion = split_ratio[target_segment]/sum(split_ratio)
        dl._split_data_portion *= portion

        # # adjust load size to accommodate change in data load
        # dl._n_batches_in_memory = int(dataloader._n_batches_in_memory / portion)

        dl.reset()

        splitted.append(dl)

    return tuple(splitted)


def _dataloader_splitCV_data_lvl(dataloader: FileLoader,
                                 n_fold: int = 3,
                                 randomize: bool = True,
                                 random_seed: int = None) -> Tuple[Tuple[FileLoader]]:
    ''' Function for performing n-fold CV split at individual file data level

        Author(s): dliang1122@gmail.com

        args
        ------
        dataloader (FileLoader): FileLoader object
        n_fold (int): number of folds to create
        randomize (bool): whether to randomize the input list before splitting
        random_seed (int): random seed for reproducible splitting

        returns
        ------
        list of dataloader train-test pairs (a pair for each fold)
    '''

    # splitting data using equal weights
    split_ratio = np.ones(n_fold)

    # determin how to split
    list_split_object = list_split(split_ratio = split_ratio, randomize = randomize, seed = random_seed)

    # total number of resulting dataloader train-test pairs
    n_dataloaders = len(split_ratio)

    # track resulting dataloader train-test pairs
    splitted_pairs = []

    for fold in range(n_dataloaders):
        # determine how to split and which segment to return

        split_method_dict = dict() # keep track of split mechanism for train and test dataloaders in a CV fold
        fold_dls = [] # store train and test dataloader

        for i in ('train', 'test'):

            # want to exclude the segment for train and keep only the segment for test.
            # E.g. for 3-fold, we split the data 3 ways; keep 2 (i.e remove 1 segment) for train and 1 for test
            keep_or_exclude_segment = 'exclude' if i == 'train' else 'keep'

            split_method = get_splitted_list_segment(list_split_object = list_split_object,
                                                     keep_or_exclude = keep_or_exclude_segment,
                                                     segment = fold)

            # make a copy of original dataloader
            dl = dataloader.copy()

            # add split mechanism to the split pipeline
            dl._split_pipeline.append(split_method)

            # calculate portion of data retained
            if i == 'train':
                portion = sum(split_ratio[1:])/sum(split_ratio)
            else:
                portion = split_ratio[0]/sum(split_ratio)

            dl._split_data_portion *= portion

            # adjust n_batches_in_memory to account for data size change
            dl._n_batches_in_memory = int(dataloader._n_batches_in_memory / portion)

            dl.reset()

            fold_dls.append(dl)

        splitted_pairs.append(fold_dls)

    return tuple(splitted_pairs)


def DataLoader(data_source: Union[str, List[str]],
               columns: List[str] = None,
               batch_size: int = 256,
               data_partition: Optional[str | List[str]] = None,
               n_part_per_fetch: int = 20,
               data_transformations: List[Callable] = [],
               batch_transformations: List[Callable] = [],
               drop_last: bool = False,
               **kwarg):

    if isinstance(data_source, (pd.DataFrame, _sliceable_dataset)):
        dataloader = inCoreDataLoader(data_source = data_source,
                                      columns = columns,
                                      data_partition = data_partition,
                                      batch_size = batch_size,
                                      data_transformations = data_transformations,
                                      batch_transformations = batch_transformations,
                                      drop_last = drop_last,
                                      **kwarg)

    elif isinstance(data_source, (str, list, tuple)):
        dataloader = FileLoader(data_source = data_source,
                                columns = columns,
                                data_partition = data_partition,
                                batch_size = batch_size,
                                n_part_per_fetch = n_part_per_fetch,
                                data_transformations = data_transformations,
                                batch_transformations = batch_transformations,
                                drop_last = drop_last,
                                **kwarg)
    else:
        raise ValueError('<data_source> must be one of the following: a pandas dataframe, a file directory or path or a list of file paths.')

    return dataloader