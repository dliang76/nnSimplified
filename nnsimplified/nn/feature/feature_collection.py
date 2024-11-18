from math import ceil, sqrt
import numpy as np
import pandas as pd
import copy
import warnings
import torch
from .features import categoricalFeature, numericFeature, binaryFeature
from .helper import infer_categories_from_data, infer_ftype, get_list_data_types
from ...utils import save_pickle
import platform
from typing import List, Union, Dict, Optional, Callable

if platform.python_version() < '3.11':
    from typing_extensions import Self
else:
    from typing import Self


class featureSet:
    ''' A class for storing a set/collection of feature objects and their operations (e.g. adding or removing features, filtering by feature types, creating categorical embeddings etc).
        A featureSet can only contain uniquely named feature objects (hence the name featureSet).

        init args
        ----------
        name (str): featureSet name; used for identifying featureSets
    '''

    valid_ftype = ('categorical', 'numeric', 'binary') # feature types supported

    def __init__(self,
                 name: str):
        self._name = name
        self._features = []

    @property
    def name(self):
        '''getter for featureSet name'''
        return self._name

    @property
    def ftypes(self):
        '''getter for feauture types'''
        return [f.ftype for f in self.features]

    @property
    def features(self):
        '''getter for feature objects stored in the featureSet'''
        return self._features

    @property
    def feature_names(self):
        '''getter for names of feature objects stored in the featureSet'''
        return [f.name for f in self._features]

    @staticmethod
    def check_inference_df(inference_df: pd.DataFrame | None):
        '''method for checking validity of dataframe used for inferring feature types and cateogories in categorical features'''
        if inference_df is not None:
            # make sure input is of type pandas dataframe
            if not isinstance(inference_df, pd.DataFrame):
                raise ValueError("<inference_df> has to either be None or a pandas dataframe.")

    def add_feature(self,
                    feature: str | categoricalFeature | numericFeature | binaryFeature = None,
                    ftype: Optional[str] = None,
                    inference_data: list | tuple | np.ndarray | pd.Series | pd.DataFrame = None,
                    **kwargs):
        '''
        method for adding a feature to the featureSet
        Args
        ----------
        feature (str or feature object): name of the feature or a feature object
        ftype (str): feature type (numeric, binary or categorical)
        inference_data (list, array, pandas series): data for inferring feature type and unique categories in a categorical feature
        **kwargs: additional arguments to specify a feature object (e.g. <binary_values> for binary features or <categories> for categorical features)

        Return
        ----------
        None
        '''
        # case 1: feature added is already a feature object; just add the feature object directly to the featureSet
        if isinstance(feature, (categoricalFeature, numericFeature, binaryFeature)):
            # raise an error if the feature has been already added
            if feature.name in self.__dict__.keys():
                raise ValueError(f"A feature with name '{feature}' already exists in featureSet '{self.name}'!")

            # add to feature set directly if the feature is already a feature object
            self._features.append(feature)
            self.__dict__[feature.name] = feature

        # case 2: feature added is a str (feature name); might need to infer certain attributes (e.g feature type, categories) using data
        elif isinstance(feature, str):
            # raise an error if the feature has been already added
            if feature in self.__dict__.keys():
                raise ValueError(f"A feature with name '{feature}' already exists in featureSet '{self.name}'!")

            # check inference_data
            if inference_data is not None:
                if isinstance(inference_data, pd.DataFrame):
                    feature_data = inference_data.get(feature)
                elif isinstance(inference_data, (list, tuple, np.ndarray, pd.Series)):
                    feature_data = inference_data
                else:
                    raise ValueError("Unsupport data type for <inference_data>! <inference_data> only accepts a list, a tuple , a np.ndarray, a pd.Series or a pd.DataFrame.")
            else:
                feature_data = None

            # infer ftype if not provided
            if not ftype:
                ftype = infer_ftype(feature_data)

            # create feature objects based on feature types
            if ftype == 'categorical':
                categories = kwargs.get('categories')

                if not categories:
                    # infer categories using feature_data if not provide throught the argument 'categories'
                    if feature_data is not None:
                        categories = infer_categories_from_data(data = feature_data)
                    else:
                        raise RuntimeError("No categories found! Please either specify the <categories> arguement or provide <inference_df> to infer categories.")

                # create categorical feature
                feature_obj = categoricalFeature(name = feature, categories = categories)

            elif ftype == 'numeric':
                # create numerical feature
                feature_obj = numericFeature(name = feature)

            elif ftype == 'binary':
                # extract binary values and positive label from keyword arguments (if provided)
                binary_values = kwargs.get('binary_values')
                pos_label = kwargs.get('pos_label')

                if not binary_values:
                    # use default binary_values (0,1) if not binary_values specified
                    binary_values = (0,1)

                if not pos_label:
                    if feature_data is not None:
                        unique_values = sorted(feature_data) # get unique values sorted in alphanumeric order
                        pos_label = unique_values[-1] # choose the last value (max value) as the positive label.

                        if len(set(unique_values)) > 2:
                            message = f"More than 2 unique values found! Set the max value {pos_label} as the positive class ({binary_values[1]})."
                            warnings.warn(message)
                    else:
                        raise RuntimeError(f"No <pos_label> specified for binary feature '{feature}'! Please either specify the <pos_label> argument or provide <inference_df> to infer.")

                # create binary feature
                feature_obj = binaryFeature(name = feature, binary_values = binary_values, pos_label = pos_label)
            else:
                raise ValueError("Unsupported ftype! <ftype> has to be one of the following: 'categorical', 'numeric' or 'binary'.")

            # add to feature list
            self._features.append(feature_obj)

            # register feature into __dict__ so users can access it easily using dot notation. E.g instance.feature_name
            self.__dict__[feature] = feature_obj

        else:
            raise ValueError("The <feature> argument only accepts a string or a feature object.")

    def add_features(self,
                     features: Union[str, categoricalFeature, numericFeature, binaryFeature,
                                     List[categoricalFeature | numericFeature | binaryFeature],
                                     List[str],
                                     None] = None,
                     ftypes: Optional[str | List[str] | Dict[str, str]] = None,
                     inference_df: pd.DataFrame | None = None,
                     **kwargs):
        '''
        method for adding multiple features to the featureSet.

        Args
        ----------
        features (list of str or feature objects): a list of features or feature objects
        ftypes (list of str): a list of feature types (numeric, binary or categorical) for the added features.
        inference_df (pandas dataframe): data for inferring feature types and unique categories in categorical features
        **kwargs: additional arguments to specify feature objects (e.g. <binary_values> for binary features or <categories> for categorical features)

        Return
        ----------
        None
        '''

        #### input check
        # check features
        if features is None:
            if inference_df is None:
                raise ValueError("<features> cannot be None unless <inference_df> is provided for feature generation.")
        elif isinstance(features, (str, categoricalFeature, numericFeature, binaryFeature)):
            # convert to a list if a string or a feature object is provide
            features = [features]

        elif isinstance(features, (list, tuple, self.__class__)) or hasattr(features, '__iter__'):
            # raise an error if both str and feature type are found in features provided; no mixed data types for now.
            data_types = get_list_data_types(lst = features)

            if data_types - set(['str', 'categoricalFeature', 'numericFeature', 'binaryFeature']):
                raise ValueError("<features> can only contain one of the following: a featureSet object, a list of feature names (str) or a list of feature objects.")

            if ('str' in data_types) and (set(['categoricalFeature', 'numericFeature', 'binaryFeature']) & data_types):
                raise ValueError("Both string and feature type (categoricalFeature, numericFeature or binaryFeature) are present in <features>; no mixed data types.")

            if features and isinstance(features[0], str):
                # if <features> provided is a list of str, make sure at least one of <ftypes> and <inference_df> is specified.
                if ftypes is None:
                    if inference_df is None:
                        raise ValueError("<ftypes> cannot be None unless <inference_df> is provided for ftypes inference.")

        else:
            raise ValueError("<features> can only contain the following: a list of feature names (str) or a list of feature objects.")

        if isinstance(ftypes ,dict):
            ftype_list = []
            for f in features:
                if isinstance(f, str):
                    ftype_list.append(ftypes.get(f))
                elif isinstance(f, (categoricalFeature, numericFeature, binaryFeature)):
                    ftype_list.append(ftypes.get(f.name))

            ftypes = ftype_list

        elif not isinstance(ftypes, (list, tuple)):
            # convert to list if ftypes is not a list or tuple (e.g string)
            ftypes = [ftypes]

        # check inference_df
        self.check_inference_df(inference_df)

        #### Start adding features
        if features is None:
            # adding all features in inference df
            for c in inference_df.columns:
                self.add_feature(feature = c, inference_data = inference_df.get(c), **kwargs)

        elif isinstance(features, self.__class__) or isinstance(features[0], (categoricalFeature, numericFeature, binaryFeature)):
            ### The 'features' argument is a list of feature objects( type = categoricalFeature, numericFeature or binaryFeature)
            # add to the feature set directly
            for f in features:
                self.add_feature(feature = f)

        else:
            ### The 'features' argument is a list of strings (feature names)
            feature_names = features

            if inference_df is not None:
                features_not_found = set(feature_names) -  set(inference_df.columns)
                if len(features_not_found) > 0:
                    raise RuntimeError(f"The features {features_not_found} provided cannot be found in <inference_df>.")

            # check if ftypes and features lengths match.
            if len(ftypes) > 1:
                if len(ftypes) != len(feature_names):
                    raise ValueError("The lengths of <ftypes> and <feature_names> have to match.")

            elif len(ftypes) == 1:
                ftypes = ftypes * len(feature_names) # if ftypes is single valued, repeat the value to match the feature_names size

            # adding feature to the feature set
            for idx, f in enumerate(feature_names):
                self.add_feature(feature = f, ftype = ftypes[idx], inference_data = inference_df, **kwargs)

    def create_num_scalers(self,
                           df: pd.DataFrame,
                           scaler_type: str = 'StandardScaler',
                           **kwargs):
        '''
        method for creating numerical scalers for numerical features in the featureSet

        Args
        ----------
        df (pandas dataframe): a dataframe containing data used for creating scalers
        scaler_type (str or dict): Type of scaler. Use the scalers from sklearn. Supported scaler type: 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler'
        **kwargs: additional arguments for the scaler type specified. See scikit-learn's preprocessing scaler for option (shttps://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)

        Return
        ----------
        None
        '''

        # get available features in the schema
        available_num_features = self.filter_features(ftypes = 'numeric')

        for f in available_num_features:
            f.create_num_scaler(data = df[f.name], scaler_type = scaler_type, **kwargs)

    def create_cat_embeddings(self,
                              embedding_dims: Optional[List[int] | Callable] = lambda x: ceil(sqrt(sqrt(x))),
                              include_missing_embedding: bool = True,
                              input_data_shape: torch.Size = torch.Size([-1]),
                              **kwargs):
        '''method for generating embedding layers for categorical features

            Args
            ----------
            embedding_dims (list of int or callable): a list of embedding dimensions (one for each categorical feature) or a function to generate embedding dimensions from input dimensions
            include_missing_embedding (bool): whether to include an additional embedding to represent missing values.
            **kwargs: additional arguments for specifying embeddings (see PyTorch. https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

            Return
            ----------
            None
        '''

        # <embedding_dims> input check
        if isinstance(embedding_dims, list):
            if len(embedding_dims) != len(cat_features):
                raise ValueError("The number of embedding dimensions provided does not match the number of features. Need to specify 1 per feature.")
        elif not callable(embedding_dims):
            raise ValueError("The <emedding_dims> argument has to be either a list of integers (embedding dimension for each feature) or a function that takes in the number of categories and outputs embedding dimension.")

        cat_features = self.filter_features('categorical')

        for idx, f in enumerate(cat_features):
            if isinstance(embedding_dims, list):
                f.create_cat_embedding(embedding_dim = embedding_dims[idx],
                                    include_missing_embedding = include_missing_embedding,
                                    input_data_shape = input_data_shape,
                                    **kwargs)
            else:
                f.create_cat_embedding(embedding_dim = embedding_dims,
                                    include_missing_embedding = include_missing_embedding,
                                    input_data_shape = input_data_shape,
                                    **kwargs)

    def remove(self, feature_name: str):
        '''remove a feature from a featureSet by name'''

        # remove from __dict__
        if self.__dict__.get(feature_name) and isinstance(self.__dict__.get(feature_name), (categoricalFeature, numericFeature, binaryFeature)):
            self.__dict__.pop(feature_name)
        else:
            raise ValueError(f"'{feature_name}' feature not found!")

        # remove from self._schema
        for i in range(len(self.features) - 1, -1, -1):
            f = self.features[i]

            if f.name == feature_name:
                self.features.pop(i)
                break

    def pop(self, index):
        '''pop a feature in a featureSet by index'''
        f = self.features.pop(index)

        # remove from __dict__
        self.__dict__.pop(f.name)

        return f

    def filter_features(self,
                        ftypes: List[str] = None) -> Self:
        '''method for collecting features of specified ftype(s)'''

        if not ftypes:
            return self
        else:
            # enforce list type
            if not isinstance(ftypes, (list, tuple)):
                ftypes = [ftypes]

            # data check
            for ftype in ftypes:
                if not ftype in self.valid_ftype:
                    raise ValueError(f"The <ftype> argument must be one of the following: {self.valid_ftype}")

            #### collect features with specified ftype
            # initiate a new featureSet instance
            collected = self.__class__.__new__(self.__class__)
            collected.__init__(name = self.name)

            for f in self:
                if f.ftype in ftypes:
                    collected.add_feature(feature = f)

            return collected

    def __add__(self, feature_set: Self) -> Self:
        '''overload + operation to allow combining two featureSets into 1. No duplicated features allowed.'''

        result = self.__class__.__new__(self.__class__)
        result.__init__(name = f'{self.name}+{feature_set.name}')

        result.add_features(features = self)
        result.add_features(features = feature_set)

        return result

    def __repr__(self) -> str:
        feature_desc_list = [f'     ({i}) {str(f)}' for i, f in enumerate(self.features)]
        repr_text = str(self.__class__) + '\n'

        if feature_desc_list:
            repr_text += f'{self.name}:\n'
            repr_text += '\n'.join(feature_desc_list)
        else:
            repr_text += 'None'

        return repr_text

    def __getitem__(self, index):
        '''Enables selecting a feature in a featureSet by index'''
        return self.features[index]

    def __iter__(self):
        '''make object iterable'''
        for f in self.features:
            yield f

    def __len__(self):
        return len(self.features)


class featureSchema():
    '''A class for storing information and operations of a schema of features (composed of grouped features or featureSets)'''
    def __init__(self):
        self._schema = []

    @property
    def schema(self):
        '''getter for the schema'''
        return self._schema

    @property
    def features(self):
        '''getter for feature objects in the schema'''
        return [f for f_set in self for f in f_set if f_set.name != 'target']

    @property
    def feature_names(self):
        '''getter for feature names in the schema'''
        return [f.name for f_set in self for f in f_set if f_set.name != 'target']

    @property
    def target_features(self):
        '''getter for target feature object(s) in the schema'''
        if 'target' in self.featureSet_names:
            return [f for f_set in self for f in f_set if f_set.name == 'target']

    @property
    def target_names(self):
        '''getter for target feature name(s) in the schema'''

        if 'target' in self.featureSet_names:
            return [f.name for f_set in self for f in f_set if f_set.name == 'target']

    @property
    def all_features(self):
        '''getter for all feature objects in the schema (including target)'''
        return [f for f_set in self for f in f_set]

    @property
    def all_feature_names(self):
        '''getter for all feature names in the schema (including target)'''
        return [f.name for f_set in self for f in f_set]

    @property
    def featureSet_names(self):
        '''getter for returning all featureSet names'''

        return [f_set.name for f_set in self]

    def add_featureSet(self,
                        name: str,
                        features: Union[str, categoricalFeature, numericFeature, binaryFeature,
                                        List[categoricalFeature | numericFeature | binaryFeature],
                                        List[str],
                                        None] = None,
                        ftypes: List[str] | None = None,
                        inference_df: pd.DataFrame = None,
                        **kwargs):
        '''method for adding a group of features (featureSet) to the schema

            Args
            ----------
            name (str): name of featureSet
            features (list): list of feature names to be added or a list of feature objects to be added
            ftypes (list): list or feature types
            inference_df (pandas df): dataframe used for inferring feature type or categorical values

            Return
            ----------
            None
        '''
        if name in self.__dict__.keys():
            # if the featureSet already exists (identified by name), add the additonal features to the featureSet
            self.__dict__[name].add_features(features = features, ftypes = ftypes, inference_df = inference_df, **kwargs)
        else:
            # create a new featureSet and add features to the featureSet
            fset = featureSet(name = name)
            fset.add_features(features = features, ftypes = ftypes, inference_df = inference_df, **kwargs)

            self._schema.append(fset) # add to feature list
            self.__dict__[name] = fset # register into __dict__

    def remove(self, name: str):
        '''method for removing a feature set from the schema by name'''

        # remove from __dict__
        if self.__dict__.get(name) and isinstance(self.__dict__.get(name), featureSet):
            self.__dict__.pop(name)
        else:
            raise ValueError(f"Feature set '{name}' not found!")

        # remove from self._schema
        for i in range(len(self.schema) - 1, -1, -1):
            f_set = self.schema[i]

            if f_set.name == name:
                self.schema.pop(i)
                break

    def pop(self, index):
        '''method for popping a featureSet in schema by index'''
        f_set = self.schema.pop(index)

        self.__dict__.pop(f_set.name)

        return f_set

    def filter_features(self,
                        ftypes: Optional[List[str]] = None) -> Self:
        '''method for collecting features of specified ftype(s); returns a new featureSchema'''
        if not ftypes:
            # if ftypes is not provide, return all
            return self
        else:
            collected = self.__class__.__new__(self.__class__)
            collected.__init__()

            for f_set in self:
                filtered_f_set = f_set.filter_features(ftypes = ftypes)

                if filtered_f_set:
                    collected.add_featureSet(name = f_set.name, features = filtered_f_set)

            return collected

    def create_num_scalers(self,
                           df: pd.DataFrame,
                           scaler_type: str = 'StandardScaler',
                           **kwargs) -> dict:
        '''
        method for creating numerical scalers for numerical features in the featureSchema

        Args
        ----------
        df (pandas dataframe): a dataframe containing data used for creating scalers
        scaler_type (str or dict): Type of scaler. Use the scalers from sklearn. Supported scaler type: 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler'
        **kwargs: additional arguments for the scaler type specified. See scikit-learn's preprocessing scaler for option (shttps://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)

        Return
        ----------
        None
        '''

        for f_set in self:
            f_set.create_num_scalers(df = df, scaler_type = scaler_type, **kwargs)

    def create_cat_embeddings(self,
                              embedding_dims: Optional[List[int] | Callable] = lambda x: ceil(sqrt(sqrt(x))),
                              include_missing_embedding: bool = True,
                              input_data_shape: torch.Size = torch.Size([-1]),
                              **kwargs):
        '''method for generating embedding layers for categorical features

            Args
            ----------
            embedding_dims (list of int or callable): a list of embedding dimensions (one for each categorical feature) or a function to generate embedding dimensions from input dimensions
            include_missing_embedding (bool): whether to include an additional embedding to represent missing values.
            **kwargs: additional arguments for specifying embeddings (see PyTorch. https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

            Return
            ----------
            None
        '''
        for f_set in self:
            f_set.create_cat_embeddings(embedding_dims = embedding_dims,
                                        include_missing_embedding = include_missing_embedding,
                                        input_data_shape = input_data_shape,
                                        **kwargs)

    def create_nn_input_structure(self, passthrough_extra_dim = None):
        '''method for creating input structure for neural net based on the feature schema specified'''

        if passthrough_extra_dim:
            if not isinstance(passthrough_extra_dim, (list, tuple)):
                passthrough_extra_dim = [passthrough_extra_dim]

        input_structure_dict = dict()

        # loop through features (ignore target) and create inputs based on feature types (e.g embedding for categorical features)
        for feature_set in self:
            if feature_set.name != 'target':
                structure = []
                for f in feature_set:
                    # use embedding for categorical feature
                    if f.ftype == 'categorical':
                        embedding = f.cat_embedding
                        if embedding:
                            structure.append(embedding)
                        else:
                            raise RuntimeError(f'No embedding found for the categorical feature {f.name}; might want to run create_cat_embeddings() method first.')

                    elif f.ftype in ('numeric', 'binary'):
                        # combine adjacent numeric and binary feature into a single numeric tensor for efficiency
                        # if structure and isinstance(structure[-1], int):
                        #     structure[-1] += 1
                        if structure and isinstance(structure[-1], torch.Size):
                            structure[-1] = structure[-1][0:-1] + torch.Size([structure[-1][-1]+ 1])
                        else:
                            if passthrough_extra_dim:
                                structure.append(torch.Size([-1, *passthrough_extra_dim, 1]))
                            else:
                                structure.append(torch.Size([-1, 1]))

                input_structure_dict[feature_set.name] = structure[0] if len(structure) == 1 else structure

        return input_structure_dict

    def clear(self):
        '''method for removing all features'''
        # figure out feature set to remove in __dict__
        feature_set_to_remove = [k for k,v in self.__dict__.items() if isinstance(v, featureSet)]

        [self.__dict__.pop(f) for f in feature_set_to_remove] # remove from __dict__

        self._schema = []

    def save(self, path):
        '''save method'''
        save_pickle(self, path)

    def __add__(self, feature_schema):
        '''overload + operation to allow combining two feature schema into 1'''

        # make a copy of self
        result = copy.deepcopy(self)

        for f_set in feature_schema:
            result.add_featureSet(name = f_set.name, features = f_set)

        return result

    def __getitem__(self, index: int):
        '''Retrieve a feature set using indices'''
        return self.schema[index]

    def __iter__(self):
        '''make object iterable'''
        for feature_set in self.schema:
            yield feature_set

    def __len__(self):
        return len(self.schema)

    def __repr__(self):
        repr_text = str(self.__class__) + '\n'

        if self.schema:
            for i, f_set in enumerate(self.schema):
                repr_text += f'({i}) featureSet {f_set.name}:\n'
                if f_set:
                    for j, f in enumerate(f_set):
                        repr_text += f'     ({j}) {f}\n'
                else:
                    repr_text += 'None\n'
        else:
            repr_text += 'None'

        return repr_text
