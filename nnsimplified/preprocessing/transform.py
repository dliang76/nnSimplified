import torch
import numpy as np
from typing import List, Optional
from .encoding import label_encoder
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

''' The following transformation classes are typically used for transforming data on the fly during model training'''

class pd_to_torch:
    ''' class for transforming pandas dataframe to torch tensors (y, X) for model training

        Author(s): dliang1122@gmail.com

        init args
        ----------
        train_type (str): options: 'autoencoder', 'regression', 'multiclass', 'binary'
        feature_cols (list of str): list of feature column names
        label_col (str): label column name
    '''
    # class variable; supported train type
    supported_train_type = ['autoencoder', 'regression', 'multiclass', 'binary']

    def __init__(self, train_type: str = None, feature_cols: List[str] = None, label_col: str = None):
        self.train_type = train_type
        self.feature_cols = feature_cols
        self.label_col = label_col

    @property
    def train_type(self):
        '''get method'''
        return self._train_type

    @train_type.setter
    def train_type(self, train_type: str):
        '''Specify training type; currently supported type: 'autoencoder', 'regression', 'multiclass', 'binary'

            init args
            ----------
            train_type (str): options: 'autoencoder', 'regression', 'multiclass', 'binary'
        '''

        # validate input
        if train_type:
            if train_type not in self.supported_train_type:
                raise ValueError(
                    f'The train_type ({train_type}) is not supported. Supported options: {self.supported_train_type}')

        self._train_type = train_type

    @property
    def label_col(self):
        '''get method'''
        return self._label_col

    @label_col.setter
    def label_col(self, label_col: str):
        '''Specify label column
        
            init args
            ----------
            label_col (str): label/target column name
        '''

        # validate input
        if self.train_type:
            if self.train_type != 'autoencoder': # no need to specify label columns for autoencoder as input = target
                if label_col is None:
                    raise ValueError(f"For train_type = '{self.train_type}', 'label_col' is required.")

        self._label_col = label_col

    @property
    def feature_cols(self):
        '''get method'''
        return self._feature_cols

    @feature_cols.setter
    def feature_cols(self, feature_cols: List[str]):
        '''Specify feature columns to use
        
            init args
            ----------
            feature columns (list of str): list of feature columns to use
        '''
        # validate input
        if self.train_type:
            if self.train_type != 'autoencoder':
                if not feature_cols:
                    raise ValueError("For train_type = '{self.train_type}', feature_cols cannot be null!")

        self._feature_cols = feature_cols

    def __call__(self, df: pd.DataFrame):
        ''' dunder method for making the class object callable (act as a function); transform pandas dataframe into torch tensors'''

        if self.feature_cols is None:
            # use the whole dataset
            X = df.to_numpy(dtype=np.float32)
        else:
            # use selected columns
            X = df[self.feature_cols].to_numpy(dtype=np.float32)

        # convert to torch
        X = torch.from_numpy(X)

        # return based on the train type
        if self.train_type is None:
            return X

        elif self.train_type == 'autoencoder': # for autoencoder, input = output
            return X, X

        elif self.train_type == 'regression':
            # convert label data
            y = df[self.label_col].to_numpy(dtype=np.float32)
            y = torch.from_numpy(y).unsqueeze(1)

        elif self.train_type == 'multiclass':
            # convert label data
            y = df[self.label_col].to_numpy(dtype=np.int64)
            y = torch.from_numpy(y)

        elif self.train_type == 'binary':
            # convert label data
            y = df[self.label_col].to_numpy(dtype=np.float32)
            y = torch.from_numpy(y).unsqueeze(1)

        return y, X


class encode_label:
    ''' class for encoding the label column in a pandas df

        Author(s): dliang1122@gmail.com

        init args
        ----------
        label_col (str): label column name
        ordered_labels (list(str)): label values in specified order
        target_class: (str): class used as the positive class; for transforming multi-class into binary
    '''

    def __init__(self,
                 label_col: str,
                 ordered_labels: List[str] = None,
                 target_class: str = None):
        self.label_col = label_col
        self.ordered_labels = ordered_labels
        self.target_class = target_class
        self.le = label_encoder()

    def __call__(self, df: pd.DataFrame):
        ''' dunder method for making the class object callable (act as a function); encode target column values into integer indices'''
        df = df.copy(deep=True)
        df[self.label_col] = self.le.fit_transform(df[self.label_col], target_class=self.target_class,
                                                   ordered_labels=self.ordered_labels)

        return df


class TfIdf_to_Torch:
    """
    Author(s): kumarsajal49@gmail.com

    TF-idf to torch conversion

    Member variables
    --------------
    feature_dims: total number of features

    Init Args
    --------------
    feature_dims: total number of features

    """

    def __init__(self,
                 start_dim_ind: int = 0,
                 feature_dims: int = None,
                 col_var: str = None,
                 value_var: str = None,
                 labelCol: str = None,
                 out_type: str = 'DataOnly'):
        self.start_dim_ind = start_dim_ind
        self.feature_dims = feature_dims
        if col_var is not None:
            self.col_var = col_var
        else:
            self.col_var = 'asin_list'
        if value_var is not None:
            self.value_var = value_var
        else:
            self.value_var = 'value_list'
        self.labelCol = labelCol
        self.out_type = out_type

    def __call__(self, df: pd.DataFrame):
        df = df.copy(deep=True)

        # row variable does not matter (we use index)
        row = np.arange(0, df.shape[0])
        row = np.repeat(row, df[self.col_var].apply(lambda x: len(x)))

        # explode col and value variables
        col = np.array(df[self.col_var].explode(self.col_var), dtype=np.int32) - self.start_dim_ind
        value = np.array(df[self.value_var].explode(self.value_var), dtype=np.float32)

        # convert customer id to string
        cids = df['customer_id'].astype('str')
        cids = cids.to_list()

        # get labels if available
        if self.labelCol is not None:
            labels = df[self.labelCol].to_list()
        else:
            labels = None

        # convert to sparse tensor
        i = torch.LongTensor(np.array([row, col]))
        v = torch.FloatTensor(value)
        data = torch.sparse.FloatTensor(i, v, torch.Size([df.shape[0], self.feature_dims])).to_dense()

        if self.out_type == "DataOnly":
            return data
        elif self.out_type == "DataWithCids":
            return cids, data
        elif self.out_type == "AllData":
            return cids, data, labels
        else:
            raise ValueError("Invalid out_type specified.")


class sparse_df_to_dense_torch:
    ''' convert a pandas dataframe in sparse format (indices, values) into a dense torch tensor

        Author(s): dliang1122@gmail.com

        init args
        ----------
        indices_col (str): name of column containing indices of non-zero values
        values_col (str): name of column containing values for the corresponding indices
        n_features (int): total number of features (zeroes or non-zeroes)
    '''

    def __init__(self, indices_col: str, values_col: str, n_features: int):
        self.values_col = values_col
        self.indices_col = indices_col
        self.n_features = n_features

    def __call__(self, sparse_df: pd.DataFrame):
        ''' dunder method for making the class object callable (act as a function); convert sparse format (indices, values) into a dense torch tensor'''

        # get list of indices
        col_indices_list = sparse_df[self.indices_col].values
        row_indices_list = [np.full(len(v), i) for i, v in enumerate(col_indices_list)]

        # get list of values
        values_list = sparse_df[self.values_col].values

        dense_tensor = torch.sparse_coo_tensor(np.array([np.concatenate(row_indices_list), np.concatenate(col_indices_list)]),
                                               np.concatenate(values_list),
                                               (len(values_list), self.n_features)).float().to_dense()

        return dense_tensor