from dataclasses import dataclass
from typing import List, Union, Any, Dict, Optional, Callable
import torch
from math import ceil, sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from ...preprocessing.encoding import labelEncoder, binaryEncoder
from ..module import embeddingModule


@dataclass
class categoricalFeature:
    '''a dataclass for storing categorical feature attributes and operations
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        name (str): feature name
        categories (list): a list of unique values for categories. E.g. ['cat1', 'cat2', 'cat3']
        inference_data (list, array, pandas series): data for inferring unique categorical values
    '''
    name: str
    categories: str

    def __init__(self,
                 name: str,
                 categories: List[Any]):
        self._name = name
        self.categories = categories

        self.create_cat_encoder()
        self._cat_embedding = None

    @property
    def name(self):
        return self._name

    @property
    def ftype(self):
        return 'categorical'

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, categories):
        # input check
        if categories:
            if not (isinstance(categories, (list, tuple)) and len(categories) > 0):
                raise ValueError("The <categories> argument must be a list or a tuple of unique values.")
            elif len(categories) > len(set(categories)):
                raise ValueError('<categories> cannot contain duplicated values.')

        self._categories = categories

    @property
    def n_categories(self):
        return len(self.categories)

    @property
    def cat_encoder(self):
        return self._cat_encoder

    @property
    def cat_embedding(self):
        return self._cat_embedding

    def create_cat_encoder(self, unseen_value_label: Union[str, int] = '__unseen__') -> labelEncoder:
        '''method for creating a label encoder for encoding the categorical values to integers'''

        # create categorical encoder
        le = labelEncoder(classes = self.categories, unseen_value_label = unseen_value_label)

        self._cat_encoder = le

        return le

    def create_cat_embedding(self,
                             embedding_dim: Optional[int|Callable] = lambda x: ceil(sqrt(sqrt(x))),
                             include_missing_embedding: bool = True,
                             input_data_shape: torch.Size = torch.Size([-1]),
                             **kwargs) -> embeddingModule:
        '''method for generating embedding layer for nn training'''
        n_categories = len(self.categories) # get number of categories

        if callable(embedding_dim):
            # generate embedding dimension using a specified function
            embedding_dim = embedding_dim(n_categories)
        elif not isinstance(embedding_dim, int):
            # embedding dimension has to be an integer
            raise ValueError('<embedding_dim> has to be either an integer or a function that takes number of categories as input and generate an integer for embedding dimension.')

        if include_missing_embedding:
            # including an additional vector to represent missing values (weights are 0)
            n_categories = n_categories + 1 # +1 for including an additional vector to represent missing values (weights are 0)
            padding_idx = -1 # set the position of the embedding vector to the last
        else:
            padding_idx = None

        embedding = embeddingModule(n_categories = n_categories, 
                                    embedding_dim = embedding_dim,
                                    padding_idx = padding_idx,
                                    input_data_shape = input_data_shape,
                                    **kwargs)

        self._cat_embedding = embedding

        return embedding


@dataclass
class numericFeature:
    '''a dataclass for storing numeric feature attributes and operations
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        name (str): feature name
    '''
    name: str

    def __init__(self,
                 name: str):
        ''''''
        self._name = name
        self._num_scaler = None

    @property
    def name(self):
        return self._name

    @property
    def ftype(self):
        return 'numeric'

    @property
    def num_scaler(self):
        return self._num_scaler

    def create_num_scaler(self,
                          data: np.ndarray | List[int|float] | pd.Series | pd.DataFrame,
                          scaler_type: str = 'StandardScaler',
                          **kwargs):
        '''method for creating a scaler to scale the numeric data

            Args
            ----------
            data (list or numpy array or pandas): data used to train the scaer
            scaler_type (str or dict): Type of scaler. Use the scalers from sklearn. Supported scaler type: 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler'
                                       Accept string (e.g. 'StandardScaler') or a dictionary with configurations (e.g. {'StandardScaler': {'copy': True, 'with_mean': True}}, see sklearn for addtional args)
            **kwargs: additional arguments for the scaler type specified. See scikit-learn's preprocessing scaler for option (shttps://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
        '''

        # check scaler argument value
        if scaler_type not in ('StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler'):
            raise ValueError("<scaler_type> must be one of the following (sklearn scalers): 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler'")

        scaler_config = kwargs

        # check data shape
        # The input has to be either a pandas dataframe or 2d array
        data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError('<data> has to be in the form of a 2D array. E.g. [[0], [1], [2], ...]')

        elif len(data.shape) == 1:
            data = data.reshape(-1,1)

        # create scaler
        num_scaler = eval(scaler_type)(**scaler_config)
        num_scaler.fit(data)

        self._num_scaler = num_scaler

        return num_scaler


@dataclass
class binaryFeature:
    '''a dataclass for storing binary feature attributes and operations
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        name (any): feature name
        binary_values (list of int): a list of 2 integers; (negative class, positive class) -> (int1, int2)
        pos_label (any): value to be designated the positive class
    '''
    name: Any
    binary_values: List[int]
    pos_label: Any

    def __init__(self,
                 name: Any,
                 binary_values: List[int] = (0,1),
                 pos_label: Any = 1):

        self._name = name
        self.binary_values = binary_values
        self._pos_label = pos_label

    @property
    def name(self):
        return self._name

    @property
    def ftype(self):
        return 'binary'

    @property
    def binary_values(self):
        return self._binary_values

    @binary_values.setter
    def binary_values(self, binary_values: List[int]):
        # input check
        if not (isinstance(binary_values, (list, tuple)) and len(binary_values) == 2):
            raise ValueError("<binary_values> has to be a list of 2 unqiue integers.")
        
        self._binary_values = binary_values

    @property
    def pos_label(self):
        return self._pos_label

    @property
    def binary_encoder(self):
        return binaryEncoder(binary_values = self.binary_values, pos_label = self.pos_label)

