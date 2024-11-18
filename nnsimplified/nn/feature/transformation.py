from typing import List, Tuple, Union, Any, Dict, Optional
import pandas as pd
import torch
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .feature_collection import featureSchema

class encode_and_scale:
    '''
    class for performing data encoding (e.g categorical and binary data) and scaling (e.g numerical data) using feature schema generated

    Author(s): dliang1122@gmail.com

    init args
    ---------
    feature_schema (featureSchema): a featureSchema object that contains the feature information (data type, unique categorical values etc) and transformation methods
    '''
    def __init__(self,
                 feature_schema: featureSchema,
                 transform_target: bool = False):
        self.feature_schema = feature_schema # The featureSchema object that provide info and methods for data transformation

        # extract target feature names if target features exist.
        if self.feature_schema.target:
            self.target = self.feature_schema.target.feature_names

        # flag for whether to transform targets
        self._transform_target = transform_target

    @property
    def feature_schema(self):
        '''attribute getter'''
        return self._feature_schema

    @feature_schema.setter
    def feature_schema(self, feature_schema: featureSchema):
        '''attribute setter'''

        # input check
        if not isinstance(feature_schema, featureSchema):
            raise ValueError('<feature_schema> has to of type featureSchema!')

        self._feature_schema = feature_schema # assign value to attribute

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        '''dunder method for encoding categorical features and scaling numerical features; make object callable (can be invoke like a function)'''

        result = pd.DataFrame()

        for fs in self.feature_schema:
            if fs:
                # if we don't want to transform the target, pass the target value directly to the resulting df and skip the remaing code in the loop
                if (fs.name == 'target') and not self._transform_target:
                    for f in fs:
                        result[f.name] = df[f.name].values # pass the value without modification

                    continue # skip the remaing code in this loop

                for f in fs:
                    if f.ftype == 'categorical':
                        result[f.name] = f.cat_encoder.transform(df[f.name].values) # convert to integers
                    elif f.ftype == 'numeric':
                        result[f.name] = f.num_scaler.transform(df[[f.name]].values).reshape(-1) # Beware. num_scalers (sklearn implementation) requires 2D array as inputs. That's why we are using double brackets for selecting feature data
                    elif f.ftype == 'binary':
                        result[f.name] = f.binary_encoder.transform(df[f.name].values) # convert to binary integers

        return result

# class encode_and_scale:
#     '''
#     class for performing data encoding (e.g categorical and binary data) and scaling (e.g numerical data) using feature schema generated

#     Author(s): dliang1122@gmail.com

#     init args
#     ---------
#     feature_schema (featureSchema): a featureSchema object that contains the feature information (data type, unique categorical values etc) and transformation methods
#     '''
#     def __init__(self,
#                  feature_schema: featureSchema,
#                  transform_target: bool = False):
#         self.feature_schema = feature_schema # The featureSchema object that provide info and methods for data transformation

#         # extract target feature names if target features exist.
#         if self.feature_schema.target:
#             self.target = self.feature_schema.target.feature_names

#         # flag for whether to transform targets
#         self._transform_target = transform_target

#     @property
#     def feature_schema(self):
#         '''attribute getter'''
#         return self._feature_schema

#     @feature_schema.setter
#     def feature_schema(self, feature_schema: featureSchema):
#         '''attribute setter'''

#         # input check
#         if not isinstance(feature_schema, featureSchema):
#             raise ValueError('<feature_schema> has to of type featureSchema!')

#         self._feature_schema = feature_schema # assign value to attribute

#     def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
#         '''dunder method for encoding categorical features and scaling numerical features; make object callable (can be invoke like a function)'''

#         result = dict()

#         for fs in self.feature_schema:
#             if fs:
#                 result[fs.name] = pd.DataFrame()

#                 # if we don't want to transform the target, pass the target value directly to the resulting df and skip the remaing code in the loop
#                 if (fs.name == 'target') and not self._transform_target:
#                     for f in fs:
#                         result[fs.name][f.name] = df[f.name].values # pass the value without modification

#                     continue # skip the remaing code in this loop

#                 for f in fs:
#                     if f.ftype == 'categorical':
#                         result[fs.name][f.name] = f.cat_encoder.transform(df[f.name].values) # convert to integers
#                     elif f.ftype == 'numeric':
#                         result[fs.name][f.name] = f.num_scaler.transform(df[[f.name]].values).reshape(-1) # Beware. num_scalers (sklearn implementation) requires 2D array as inputs. That's why we are using double brackets for selecting feature data
#                     elif f.ftype == 'binary':
#                         result[fs.name][f.name] = f.binary_encoder.transform(df[f.name].values) # convert to binary integers

#         return result

    # def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
    #     '''dunder method for encoding categorical features and scaling numerical features; make object callable (can be invoke like a function)'''

    #     columns_used = self.feature_schema.all_feature_names # extract required columns

    #     df = df[columns_used].copy() # copy to avoid modifying the original dataframe in place

    #     # loop through features and transform numerical (scaling) and categorical data (label encoding)
    #     if self.feature_schema.target and self._transform_target:
    #         features = self.feature_schema.all_features
    #     else:
    #         features = self.feature_schema.features

    #     for f in features:
    #         if f.ftype == 'categorical':
    #             df[f.name] = f.cat_encoder.transform(df[f.name].values) # convert to integers
    #         elif f.ftype == 'numeric':
    #             df[f.name] = f.num_scaler.transform(df[[f.name]].values) # Beware. num_scalers (sklearn implementation) requires 2D array as inputs. That's why we are using double brackets for selecting feature data
    #         elif f.ftype == 'binary':
    #             df[f.name] = f.binary_encoder.transform(df[f.name].values) # convert to binary integers

    #     return df

class df_to_torch:
    '''
    class for performaing data transformation and conversion (to torch tensors)

    Author(s): dliang1122@gmail.com

    init args
    ----------
    feature_schema (featureSchema): a featureSchema object that contains the information (data type, unique categorical values etc) and ordering of features
    '''

    supported_training_type = ('regression', 'binary', 'multiclass', 'multilabel', 'autoencoder') # class variable to store supported training type

    def __init__(self,
                 feature_schema: featureSchema = featureSchema(),
                 training_type: str = 'regression'):
        self.feature_schema = feature_schema # store feature collection object
        self.training_type = training_type # training type for determining tensor type and output format

    @property
    def feature_schema(self):
        '''attribute getter'''
        return self._feature_schema

    @feature_schema.setter
    def feature_schema(self, feature_schema: featureSchema):
        '''attribute setter'''

        # input check
        if not isinstance(feature_schema, featureSchema):
            raise ValueError('<feature_schema> has to be of type featureSchema!')

        self._feature_schema = feature_schema # assign value to attribute

    @property
    def training_type(self):
        '''attribute getter'''
        return self._training_type

    @training_type.setter
    def training_type(self, training_type):
        '''attribute setter'''
        # input check
        if training_type not in self.supported_training_type:
            raise ValueError(f'Only the following options are supported for <training_type>: {self.supported_training_type}')

        self._training_type = training_type # assign value to attribute

    def __call__(self, data: Dict[str, pd.DataFrame]) -> torch.Tensor | Tuple[torch.Tensor | List[torch.Tensor]]:
        '''dunder method for transforming pandas dataframe into torch tensors; make object callable (can be invoke like a function)'''
        X = [] # used to keep track of feature tensors

        # loop through feature_sets in feature collection
        # need to track int and float tensors separately. Int tensors (cat features) are added to the list individually as cat embedding inputs cannot be concated even if they are adjecent.
        # adjecent float tensors (numeric or binary features) can be combined into one single float tensor
        for f_set in self.feature_schema:
            if f_set:
                if f_set.name == 'target':
                    if not self.training_type == 'autoencoder':
                        y = torch.tensor(data[f_set.feature_names].to_numpy('float32')) # convert labels to float32 tensors
                    elif self.training_type == 'multiclass':
                        y = torch.tensor(data[f_set.feature_names].to_numpy('int64')) # convert labels to long tensors
                else:
                    num_group = [] # used to keep track of numeric tensors that we want to use torch.cat to combine to a single tensor;

                    # loop thorugh features in feature_sets
                    for f in f_set:
                        if f.ftype == 'categorical':
                            # use int tensor as a divider
                            # concat all adjecent float tensors in num_group if the next tensor is an int tensor
                            # reset num_group afterwards
                            if len(num_group) > 0:
                                X.append(torch.concat(num_group, dim = -1)) # concat all adjecent float tensors into one
                                num_group = [] # reset num_group to track the next batch of adgecent float tensors

                            # categorical features are of type integer. Need to Add them to the list one by one.
                            X.append(torch.IntTensor(data[f.name].values))

                        elif f.ftype in ('numeric', 'binary'):
                            # Add numeric tensors to the num group to be combined into a single float tensor later.
                            num_group.append(torch.FloatTensor(data[[f.name]].values))

                    # concat all remaining float tensors in the feature set
                    if len(num_group) > 0:
                        X.append(torch.concat(num_group, dim = -1))
                        num_group = []

        # if the list only contains one tensor, return the tensor only
        if len(X) == 1:
            X = X[0]

        if self.training_type == 'autoencoder':
            # for autoencoder, label = output
            return X, X

        elif self.feature_schema.target:
            # return both label and feature data if label is provide
            return y, X
        else:
            # return feature data only if label is missing
            return X



# class df_to_torch:
#     '''
#     class for performaing data transformation and conversion (to torch tensors)

#     Author(s): dliang1122@gmail.com

#     init args
#     ----------
#     feature_schema (featureSchema): a featureSchema object that contains the information (data type, unique categorical values etc) and ordering of features
#     '''

#     supported_training_type = ('regression', 'binary', 'multiclass', 'multilabel', 'autoencoder') # class variable to store supported training type

#     def __init__(self,
#                  feature_schema: featureSchema = featureSchema(),
#                  training_type: str = 'regression'):
#         self.feature_schema = feature_schema # store feature collection object
#         self.training_type = training_type # training type for determining tensor type and output format

#     @property
#     def feature_schema(self):
#         '''attribute getter'''
#         return self._feature_schema

#     @feature_schema.setter
#     def feature_schema(self, feature_schema: featureSchema):
#         '''attribute setter'''

#         # input check
#         if not isinstance(feature_schema, featureSchema):
#             raise ValueError('<feature_schema> has to be of type featureSchema!')

#         self._feature_schema = feature_schema # assign value to attribute

#     @property
#     def training_type(self):
#         '''attribute getter'''
#         return self._training_type

#     @training_type.setter
#     def training_type(self, training_type):
#         '''attribute setter'''
#         # input check
#         if training_type not in self.supported_training_type:
#             raise ValueError(f'Only the following options are supported for <training_type>: {self.supported_training_type}')

#         self._training_type = training_type # assign value to attribute

#     def __call__(self, data: Dict[str, pd.DataFrame]) -> torch.Tensor | Tuple[torch.Tensor | List[torch.Tensor]]:
#         '''dunder method for transforming pandas dataframe into torch tensors; make object callable (can be invoke like a function)'''
#         X = [] # used to keep track of feature tensors

#         # loop through feature_sets in feature collection
#         # need to track int and float tensors separately. Int tensors (cat features) are added to the list individually as cat embedding inputs cannot be concated even if they are adjecent.
#         # adjecent float tensors (numeric or binary features) can be combined into one single float tensor
#         for f_set in self.feature_schema:
#             if f_set:
#                 if f_set.name == 'target':
#                     if not self.training_type == 'autoencoder':
#                         y = torch.tensor(data[f_set.name].to_numpy('float32')) # convert labels to float32 tensors
#                     elif self.training_type == 'multiclass':
#                         y = torch.tensor(data[f_set.name].to_numpy('int64')) # convert labels to long tensors
#                 else:
#                     num_group = [] # used to keep track of numeric tensors that we want to use torch.cat to combine to a single tensor;

#                     # loop thorugh features in feature_sets
#                     for f in f_set:
#                         if f.ftype == 'categorical':
#                             # use int tensor as a divider
#                             # concat all adjecent float tensors in num_group if the next tensor is an int tensor
#                             # reset num_group afterwards
#                             if len(num_group) > 0:
#                                 X.append(torch.concat(num_group, dim = -1)) # concat all adjecent float tensors into one
#                                 num_group = [] # reset num_group to track the next batch of adgecent float tensors

#                             # categorical features are of type integer. Need to Add them to the list one by one.
#                             X.append(torch.IntTensor(data[f_set.name][f.name].values))

#                         elif f.ftype in ('numeric', 'binary'):
#                             # Add numeric tensors to the num group to be combined into a single float tensor later.
#                             num_group.append(torch.FloatTensor(data[f_set.name][[f.name]].values))

#                     # concat all remaining float tensors in the feature set
#                     if len(num_group) > 0:
#                         X.append(torch.concat(num_group, dim = -1))
#                         num_group = []

#         # if the list only contains one tensor, return the tensor only
#         if len(X) == 1:
#             X = X[0]

#         if self.training_type == 'autoencoder':
#             # for autoencoder, label = output
#             return X, X

#         elif self.feature_schema.target:
#             # return both label and feature data if label is provide
#             return y, X
#         else:
#             # return feature data only if label is missing
#             return X


class sliding_window_transform:
    '''class for generating multiple dataset by sliding along the row axis (1 row at a time);
       use numpy's efficient implementation to generate views of the original data (a windowed reference) instead of new copies and thus do not use much memory.
       useful for generating time series data on the fly'''
    def __init__(self,
                 window_size: int,
                 groupby_cols: Optional[str | List[str]] = None,
                 sort_cols: Optional[str | List[str]] = None,
                 include_sort_cols: bool = False,
                 include_groupby_cols: bool = False,
                 ascending = False):

        self._window_size = window_size
        self._groupby_cols = groupby_cols
        self._sort_cols = sort_cols
        self._include_sort_cols = include_sort_cols
        self._include_groupby_cols = include_groupby_cols
        self._ascending = ascending

    @property
    def window_size(self):
        return self._window_size

    @property
    def groupby_cols(self):
        return self._groupby_cols

    @property
    def sort_cols(self):
        return self._sort_cols

    @property
    def include_sort_cols(self):
        return self._include_sort_cols

    @property
    def include_groupby_cols(self):
        return self._include_groupby_cols

    @property
    def ascending(self):
        return self._ascending

    def __call__(self, df: pd.DataFrame | np.ndarray) -> np.ndarray:
        ''' dunder method for making the class object callable (act as a function); encode target column values into integer indices'''
        if self.sort_cols:
            df = df.sort_values(self.sort_cols, ascending = self.ascending)

            if not self.include_sort_cols:
                df.drop(self.sort_cols, axis = 1, inplace = True)

        if self.groupby_cols:
            groups = df.groupby(self.groupby_cols)

            if self.include_groupby_cols:
                result = df.groupby(self.groupby_cols)\
                        .apply(lambda x: sliding_window_view(x, window_shape=(self.window_size, df.shape[1])).squeeze(1))
            else:
                result = df.groupby(self.groupby_cols)\
                        .apply(lambda x: sliding_window_view(x.drop(self.groupby_cols, axis = 1), window_shape=(self.window_size, x.shape[1] - 1)).squeeze(1))
            result = np.concatenate(result.to_list())
        else:
            result = sliding_window_view(df, window_shape=(self.window_size, df.shape[1]))\
                            .squeeze(1) # some how this will generate 1 extra dimension; use squeeze to remove that extra dimension

        return result