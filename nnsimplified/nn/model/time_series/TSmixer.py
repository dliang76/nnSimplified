from typing import Optional
from ...module.base import nnModule
from ...helper import _weight_init, _get_call_args
from ..utils import _construct_activation, _initialize_linear_layer
import torch

'''Our implementation of the time series model TSmixer (see https://arxiv.org/pdf/2303.06053.pdf)'''

class _BN2d(torch.nn.BatchNorm2d):
    '''Batch normalization 2D for TSmixer;
       Modified the implementation from https://github.com/smrfeld/tsmixer-pytorch/blob/main/utils/model.py

       Authors(s): denns.liang@hilton.com

       Args (from torch.nn.BatchNorm2d)
       -----
       eps (float): a value added to the denominator for numerical stability. Default: 1e-5
       momentum (float): the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
       affine (bool): a boolean value that when set to True, this module has learnable affine parameters. Default: True
       track_running_stats (bool): a boolean value that when set to True, this module tracks the running mean and variance,
                                   and when set to False, this module does not track such statistics,
                                   and initializes statistics buffers running_mean and running_var as None.
                                   When these buffers are None, this module always uses batch statistics.
                                   in both training and eval modes. Default: True
    '''
    def __init__(self,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device: Optional[str | torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        # The Torch's BatchNorm2d requirs a 4-D data input of the form (batch_size, # of features/channels, height, width).
        # The statistic is computed on (batch_size, height, width) over the # of features/channels dimension.
        # For TSmixer, we only require (batch_size, height, width) where height is # of time steps and width is the number of input features (time series).
        # Since there is no additional channel dimension, we set <num_features> argument to 1.
        super().__init__(num_features = 1, # for this implementaion we don't use
                         eps = eps,
                         momentum = momentum,
                         affine = affine,
                         track_running_stats = track_running_stats,
                         device = device,
                         dtype = dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: (batch_size, time sequence, features)

        # reshape input_data to (batch_size, channel = 1, timepoints, features) to be compatible with torch.nn.batchnorm2d
        x = x.unsqueeze(1)

        # Forward pass
        x = super().forward(x)

        # restore the output back to (batch_size, time sequence, features) for our use
        x = x.squeeze(1)

        return x

    def reset(self):
        self.reset_running_stats()

class _TimeMixing(torch.nn.Module):
    ''' Class for constructing a time mixing component of TSmixer

        Authors(s): denns.liang@hilton.com

        init args
        ----------
        in_time_steps(int): number of time steps in time series used for training
        activation_setting (str or dict): activation function setting.
                                          Available choice: ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
                                                                  'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
                                                                  'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention',
                                                                  'PReLU', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'LeakyReLU': {'negative_slope': 0.01}}.
        batchnorm2d_setting (dict): a dict for storing setting for batchnorm2d. e.g {'eps': 1e-05,
                                                                                     'momentum': 0.1,
                                                                                     'affine': True,
                                                                                     'track_running_stats': True}.
                                    For details, please see arguments for _BN2d
        bn_before_or_after (str): 'before' or 'after'. indicate whether to add batchnorm2d before or after the MLP layers
        weight_init (str or dict): nn weights initialization method. Available choice: ['uniform', 'normal', 'trunc_normal', 'constant',
                                                                                 'ones', 'zeros', 'eye', 'dirac', 'xavier_uniform',
                                                                                 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
                                                                                 'orthogonal', 'sparse']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'kaiming_normal': {'a': 0, 'nonlinearity': 'leaky_relu'}}.
        dropout_rate (float): between 0 and 1; 0 for not using dropout
    '''
    def __init__(self,
                 in_time_steps: int,
                 activation_setting: str | dict[str, dict] = "ReLU",
                 batchnorm2d_setting: Optional[dict] = {'eps': 1e-05,
                                                        'momentum': 0.1,
                                                        'affine': True,
                                                        'track_running_stats': True},
                 bn_before_or_after: str = 'before',
                 weight_init: str | dict[str, dict] = "kaiming_normal",
                 dropout_rate: float = 0.3):
        # store init argument values so we can use them to reset/re-initialize object
        self._init_args = {k:v for k,v in locals().items() if k in _get_call_args(self.__class__)}

        super().__init__()

        self.bn_before_or_after = bn_before_or_after

        #### initialize layers
        # initialize batchnorm 2d layer
        self.batchnorm2d = _BN2d(**batchnorm2d_setting) if batchnorm2d_setting else None

        # construct weight initialization object
        self.weight_init = _weight_init(weight_init)

        #### initialize layers
        # construct and register linear layers
        self.linear = _initialize_linear_layer(in_features = in_time_steps,
                                               out_features = in_time_steps,
                                               weight_init = self.weight_init)

        # construct activation
        self.activation = _construct_activation(activation_setting) if activation_setting else None

        # dropout setup
        self.dropout = torch.nn.Dropout(p = dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor):
        '''forward method for defining network structure (how data flows)'''

        # apply batchnorm2d BEFORE mlp if specified
        if self.bn_before_or_after == 'before':
            if self.batchnorm2d:
                y = self.batchnorm2d(x)
            else:
                y = x
        else:
            y = x

        # move data through linear layers
        y = self.linear(y)

        # apply activation
        y = self.activation(y)

        # apply dropout
        if self.dropout:
            y = self.dropout(y)

        # add residual connection
        y = y + x

        # apply batchnorm2d AFTER mlp if specified
        if self.bn_before_or_after == 'after':
            if self.batchnorm2d:
                y = self.batchnorm2d(y)

        return y

    def reset(self):
        '''method for re-initializing the model'''

        self.__init__(**self._init_args)

class _FeatureMixing(torch.nn.Module):
    ''' Class for constructing the feature mixing component in TSmixer

        Authors(s): denns.liang@hilton.com

        init args
        ----------
        in_features(int): number of features (time series) going into the MLP
        hidden_layer_size (int): number of nodes in the hiddent layer
        out_features(int): number of feature (time serie) outputs
        activation_setting (str or dict): activation function setting.
                                          Available choice: ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
                                                                  'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
                                                                  'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention',
                                                                  'PReLU', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'LeakyReLU': {'negative_slope': 0.01}}.
        batchnorm2d_setting (dict): a dict for storing setting for batchnorm2d. e.g {'eps': 1e-05,
                                                                                     'momentum': 0.1,
                                                                                     'affine': True,
                                                                                     'track_running_stats': True}.
                                    For details, please see arguments for _BN2d
        bn_before_or_after (str): 'before' or 'after'. indicate whether to add batchnorm2d before or after the MLP layers
        weight_init (str or dict): nn weights initialization method. Available choice: ['uniform', 'normal', 'trunc_normal', 'constant',
                                                                                 'ones', 'zeros', 'eye', 'dirac', 'xavier_uniform',
                                                                                 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
                                                                                 'orthogonal', 'sparse']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'kaiming_normal': {'a': 0, 'nonlinearity': 'leaky_relu'}}.
        dropout_rate (float): between 0 and 1; 0 for not using dropout
    '''
    def __init__(self,
                 in_features: int,
                 hidden_layer_size: int = 'default',
                 out_features: int = 'default',
                 activation_setting: str | dict[str, dict] = "ReLU",
                 batchnorm2d_setting: Optional[dict] = {'eps': 1e-05,
                                                        'momentum': 0.1,
                                                        'affine': True,
                                                        'track_running_stats': True},
                 bn_before_or_after: str = 'before',
                 weight_init: str | dict[str, dict] = "kaiming_normal",
                 dropout_rate: float = 0.3):

        # store init argument values so we can use them to reset/re-initialize object
        self._init_args = {k:v for k,v in locals().items() if k in _get_call_args(self.__class__)}

        super().__init__()

        # construct weight initialization object
        self.weight_init = _weight_init(weight_init)

        self.in_features = in_features

        if hidden_layer_size == 'default':
            self.hidden_layer_size = in_features
        else:
            self.hidden_layer_size = hidden_layer_size

        if out_features == 'default':
            self.out_features = in_features
        else:
            self.out_features = out_features

        self.bn_before_or_after = bn_before_or_after

        #### initialize layers
        # initialize batchnorm 2d layer
        self.batchnorm2d = _BN2d(**batchnorm2d_setting) if batchnorm2d_setting else None

        if self.out_features == self.in_features:
            # no need to change dimension if the number of input features  == the number of outputs
            self.residual_conn = torch.nn.Identity()
        else:
            # use a linear projection if input and output are of different shape
            self.residual_conn = _initialize_linear_layer(in_features= self.in_features,
                                                          out_features = self.out_features,
                                                          weight_init = self.weight_init)

        # construct and register linear layers
        self.linear1 = _initialize_linear_layer(in_features = self.in_features,
                                                out_features = self.hidden_layer_size,
                                                weight_init = self.weight_init)
        self.linear2 = _initialize_linear_layer(in_features = self.hidden_layer_size,
                                                out_features = self.out_features,
                                                weight_init = self.weight_init)

        # construct activation
        self.activation = _construct_activation(activation_setting) if activation_setting else None

        # dropout setup
        self.dropout1 = torch.nn.Dropout(p = dropout_rate) if dropout_rate > 0 else None
        self.dropout2 = torch.nn.Dropout(p = dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        '''forward method for defining network structure (how data flows)'''

        # apply batchnorm2d before mlp if specified
        if self.bn_before_or_after == 'before':
            if self.batchnorm2d:
                y = self.batchnorm2d(x)
            else:
                y = x
        else:
            y = x

        # move data through linear layers
        y = self.linear1(y)

        y = self.activation(y) # apply activation

        if self.dropout1:
            y = self.dropout1(y) # apply dropout

        y = self.linear2(y)

        if self.dropout2:
            y = self.dropout2(y) # apply dropout

        if isinstance(self.residual_conn, torch.nn.Identity):
            y = y + x
        else:
            y = y + self.residual_conn(x)

        # apply batchnorm2d after mlp if specified
        if self.bn_before_or_after == 'after':
            if self.batchnorm2d:
                y = self.batchnorm2d(y)

        return y

    def reset(self):
        '''method for re-initializing the model'''
        self.__init__(**self._init_args)

class _FeatMixConditional(torch.nn.Module):
    ''' Class for constructing conditional feature mixing of TSmixer extended. Allow to mix in static features.

        Authors(s): denns.liang@hilton.com

        init args
        ----------
        in_features(int): number of features (time series) going into the MLP
        static_features (int): number of static features (things don't change with time) used.
        hidden_layer_size (int): number of nodes in the hiddent layer
        out_features(int): number of feature (time serie) outputs
        activation_setting (str or dict): activation function setting.
                                          Available choice: ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
                                                                  'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
                                                                  'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention',
                                                                  'PReLU', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'LeakyReLU': {'negative_slope': 0.01}}.
        batchnorm2d_setting (dict): a dict for storing setting for batchnorm2d. e.g {'eps': 1e-05,
                                                                                     'momentum': 0.1,
                                                                                     'affine': True,
                                                                                     'track_running_stats': True}.
                                    For details, please see arguments for _BN2d
        bn_before_or_after (str): 'before' or 'after'. indicate whether to add batchnorm2d before or after the MLP layers
        weight_init (str or dict): nn weights initialization method. Available choice: ['uniform', 'normal', 'trunc_normal', 'constant',
                                                                                 'ones', 'zeros', 'eye', 'dirac', 'xavier_uniform',
                                                                                 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
                                                                                 'orthogonal', 'sparse']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'kaiming_normal': {'a': 0, 'nonlinearity': 'leaky_relu'}}.
        dropout_rate (float): between 0 and 1; 0 for not using dropout
    '''
    def __init__(self,
                 in_features: int,
                 static_features: int,
                 hidden_layer_size: int = 'default',
                 out_features: int = 'default',
                 activation_setting: str | dict[str, dict] = "ReLU",
                 batchnorm2d_setting: Optional[dict] = {'eps': 1e-05,
                                                        'momentum': 0.1,
                                                        'affine': True,
                                                        'track_running_stats': True},
                 bn_before_or_after: str = 'before',
                 weight_init: str | dict[str, dict] = "kaiming_normal",
                 dropout_rate: float = 0.3):
        # store init argument values so we can use them to reset/re-initialize object
        self._init_args = {k:v for k,v in locals().items() if k in _get_call_args(self.__class__)}

        super().__init__()

        self.fmix_dynamic = _FeatureMixing(in_features = in_features + in_features,
                                           hidden_layer_size = hidden_layer_size,
                                           out_features = in_features, # force to have the same shape as the time series input
                                           activation_setting = activation_setting,
                                           batchnorm2d_setting = batchnorm2d_setting,
                                           bn_before_or_after = bn_before_or_after,
                                           weight_init = weight_init,
                                           dropout_rate = dropout_rate)

        # construct feature mixing for static features; used for shape alignment (staic feature shape -> time series feature shape)
        self.fmix_static = _FeatureMixing(in_features = static_features,
                                          hidden_layer_size = hidden_layer_size,
                                          out_features = in_features, # force to have the same shape as the time series input
                                          activation_setting = activation_setting,
                                          batchnorm2d_setting = batchnorm2d_setting,
                                          bn_before_or_after = bn_before_or_after,
                                          weight_init = weight_init,
                                          dropout_rate = dropout_rate)

        self.static_features = static_features

    def forward(self, x):
        '''forward method for defining network structure (how data flows)
           input = a tuple of (tensor for time series features, tensor for static features)
        '''

        # extract time series features and static features
        X, static = x[0], x[1]

        # apply feature mixing to force resulting static embeddings to have the same shape as the time series inputs
        static = self.fmix_static(static)

        # concatenate
        y = torch.cat([X, static], dim = -1)

        y = self.fmix_dynamic(y)

        return y # mix time series features and static features

    def reset(self):
        '''method for re-initialize parameters'''
        self.__init__(**self._init_args)


class _Mixer(nnModule):
    ''' Class for mixer layer (combining time mixing and feature mixing) of TSmixer
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        in_time_steps (int): number of time steps in time series used for training
        in_features (int): number of features (time series) going into the MLP
        fmixer_hidden_size (int): number of nodes in the hiddent layer for feature mixing component
        out_features (int): number of feature (time serie) outputs
        activation_setting (str or dict): activation function setting.
                                          Available choice: ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
                                                                  'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
                                                                  'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention',
                                                                  'PReLU', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'LeakyReLU': {'negative_slope': 0.01}}.
        batchnorm2d_setting (dict): a dict for storing setting for batchnorm2d. e.g {'eps': 1e-05,
                                                                                     'momentum': 0.1,
                                                                                     'affine': True,
                                                                                     'track_running_stats': True}.
                                    For details, please see arguments for _BN2d
        bn_before_or_after (str): 'before' or 'after'. indicate whether to add batchnorm2d before or after the MLP layers
        weight_init (str or dict): nn weights initialization method. Available choice: ['uniform', 'normal', 'trunc_normal', 'constant',
                                                                                 'ones', 'zeros', 'eye', 'dirac', 'xavier_uniform',
                                                                                 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
                                                                                 'orthogonal', 'sparse']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'kaiming_normal': {'a': 0, 'nonlinearity': 'leaky_relu'}}.
        dropout_rate (float): between 0 and 1; 0 for not using dropout
    '''
    def __init__(self,
                 in_time_steps: int,
                 in_features: int,
                 fmixer_hidden_size: int = 'default',
                 out_features: int = 'default',
                 activation_setting: str | dict[str, dict] = "ReLU",
                 batchnorm2d_setting: Optional[dict] = {'eps': 1e-05,
                                                        'momentum': 0.1,
                                                        'affine': True,
                                                        'track_running_stats': True},
                 bn_before_or_after: str = 'before',
                 weight_init: str | dict[str, dict] = "kaiming_normal",
                 dropout_rate: float = 0.3):

        # store init argument values so we can use them to reset/re-initialize object
        self._init_args = {k:v for k,v in locals().items() if k in _get_call_args(self.__class__)}

        super().__init__(weight_init = weight_init)

        # store input dimensions; useful for deteriming the input shape
        self.in_time_steps = in_time_steps
        self.in_features = in_features

        # set default
        if fmixer_hidden_size == 'default':
            fmixer_hidden_size = in_features

        if out_features == 'default':
            out_features = in_features

        # create the time mixing component
        self.time_mixer = _TimeMixing(in_time_steps = in_time_steps,
                                      activation_setting = activation_setting,
                                      batchnorm2d_setting = batchnorm2d_setting,
                                      bn_before_or_after = bn_before_or_after,
                                      weight_init = weight_init,
                                      dropout_rate = dropout_rate)

        # create the feature mixing component
        self.feature_mixer = _FeatureMixing(in_features = in_features,
                                            hidden_layer_size = fmixer_hidden_size,
                                            out_features = out_features,
                                            activation_setting = activation_setting,
                                            batchnorm2d_setting = batchnorm2d_setting,
                                            bn_before_or_after = bn_before_or_after,
                                            weight_init = weight_init,
                                            dropout_rate = dropout_rate)
    @property
    def input_shape(self):
        '''get input shape'''
        return torch.Size([-1, self.in_time_steps, self.in_features])

    @property
    def input_dtype(self):
        '''get input dtype'''
        return torch.float32

    def forward(self, x):
        '''Apply time mixing and then feature mixing
        '''
        x = torch.transpose(x, dim0 = -1, dim1 = -2)
        x = self.time_mixer(x)
        x = torch.transpose(x, dim0 = -1, dim1 = -2)
        x = self.feature_mixer(x)

        return x

    def reset(self):
        '''method for re-initialize model parameteres'''
        self.__init__(**self._init_args)


class _MixerConditional(nnModule):
    ''' Class for conditional mixer layer (combining time mixing and conditioanl feature mixing) of TSmixer_ext; allow the use of static features.
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        in_time_steps (int): number of time steps in time series used for training
        in_features (int): number of features (time series) going into the MLP
        fmixer_hidden_size (int): number of nodes in the hiddent layer for feature mixing component
        out_features (int): number of feature (time serie) outputs
        static_features (int): number of static features (things don't change with time) used.
        activation_setting (str or dict): activation function setting.
                                          Available choice: ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
                                                                  'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
                                                                  'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention',
                                                                  'PReLU', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'LeakyReLU': {'negative_slope': 0.01}}.
        batchnorm2d_setting (dict): a dict for storing setting for batchnorm2d. e.g {'eps': 1e-05,
                                                                                     'momentum': 0.1,
                                                                                     'affine': True,
                                                                                     'track_running_stats': True}.
                                    For details, please see arguments for _BN2d
        bn_before_or_after (str): 'before' or 'after'. indicate whether to add batchnorm2d before or after the MLP layers
        weight_init (str or dict): nn weights initialization method. Available choice: ['uniform', 'normal', 'trunc_normal', 'constant',
                                                                                 'ones', 'zeros', 'eye', 'dirac', 'xavier_uniform',
                                                                                 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
                                                                                 'orthogonal', 'sparse']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'kaiming_normal': {'a': 0, 'nonlinearity': 'leaky_relu'}}.
        dropout_rate (float): between 0 and 1; 0 for not using dropout
    '''
    def __init__(self,
                 in_time_steps: int,
                 in_features: int,
                 fmixer_hidden_size: int = 'default',
                 out_features: int = 'default',
                 static_features: int = 0,
                 activation_setting: str | dict[str, dict] = "ReLU",
                 batchnorm2d_setting: Optional[dict] = {'eps': 1e-05,
                                                        'momentum': 0.1,
                                                        'affine': True,
                                                        'track_running_stats': True},
                 bn_before_or_after: str = 'before',
                 weight_init: str | dict[str, dict] = "kaiming_normal",
                 dropout_rate: float = 0.3):
        # store init argument values so we can use them to reset/re-initialize object
        self._init_args = {k:v for k,v in locals().items() if k in _get_call_args(self.__class__)}

        super().__init__(weight_init = weight_init)

        # store input dimensions; useful for deteriming the input data shape
        self.in_time_steps = in_time_steps
        self.in_features = in_features
        self.static_features = static_features

        if fmixer_hidden_size == 'default':
            fmixer_hidden_size = in_features

        if out_features == 'default':
            out_features = in_features

        # time mixer component
        self.time_mixer = _TimeMixing(in_time_steps = in_time_steps,
                                      activation_setting = activation_setting,
                                      batchnorm2d_setting = batchnorm2d_setting,
                                      bn_before_or_after = bn_before_or_after,
                                      weight_init = weight_init,
                                      dropout_rate = dropout_rate)

        # feature mixer component
        self.feature_mixer = _FeatMixConditional(in_features = in_features,
                                                 static_features = static_features,
                                                 hidden_layer_size = fmixer_hidden_size,
                                                 out_features = out_features,
                                                 activation_setting = activation_setting,
                                                 batchnorm2d_setting = batchnorm2d_setting,
                                                 bn_before_or_after = bn_before_or_after,
                                                 weight_init = weight_init,
                                                 dropout_rate = dropout_rate)

    @property
    def input_shape(self):
        '''get input shape. Input to this module requires two tensors - one for time series features and one for static features'''
        return torch.Size([-1, self.in_time_steps, self.in_features]), torch.Size([-1, self.in_time_steps, self.static_features])

    @property
    def input_dtype(self):
        '''get input dtype'''
        return torch.float32, torch.float32

    def forward(self, x):
        # extract time series features and static features from the input tuple
        X, static = x[0], x[1]

        X = torch.transpose(X, dim0 = -1, dim1 = -2)
        X = self.time_mixer(X)
        X = torch.transpose(X, dim0 = -1, dim1 = -2)
        y = self.feature_mixer((X, static)) # conditional feature mixing requires both time series features and static features.

        return y

    def reset(self):
        '''re-initialize model parameters'''
        self.__init__(**self._init_args)

class TSmixer(nnModule):
    ''' Class for constructing TSmixer

        Authors(s): denns.liang@hilton.com

        init args
        ----------
        in_time_steps (int): number of time steps in time series used for training
        out_time_steps (int): number of forecasted time steps
        in_features (int): number of features (time series) going into the MLP
        out_features (int): number of feature (time serie) outputs
        fmixer_hidden_size (int): number of nodes in the hiddent layer for feature mixing component
        time_mix_only (bool): whether to use time mixing only. Default: False (apply both time mixing and feature mixing)
        n_mixer (int): number of mixing layers
        activation_setting (str or dict): activation function setting.
                                          Available choice: ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
                                                                  'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
                                                                  'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention',
                                                                  'PReLU', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'LeakyReLU': {'negative_slope': 0.01}}.
        batchnorm2d_setting (dict): a dict for storing setting for batchnorm2d. e.g {'eps': 1e-05,
                                                                                     'momentum': 0.1,
                                                                                     'affine': True,
                                                                                     'track_running_stats': True}.
                                    For details, please see arguments for _BN2d
        bn_before_or_after (str): 'before' or 'after'. indicate whether to add batchnorm2d before or after the MLP layers
        weight_init (str or dict): nn weights initialization method. Available choice: ['uniform', 'normal', 'trunc_normal', 'constant',
                                                                                 'ones', 'zeros', 'eye', 'dirac', 'xavier_uniform',
                                                                                 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
                                                                                 'orthogonal', 'sparse']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'kaiming_normal': {'a': 0, 'nonlinearity': 'leaky_relu'}}.
        dropout_rate (float): between 0 and 1; 0 for not using dropout
    '''
    def __init__(self,
                 in_time_steps: int,
                 out_time_steps: int,
                 in_features: int,
                 out_features: int = 'default',
                 fmixer_hidden_size: int = 'default',
                 time_mix_only: bool = False,
                 n_mixer: int = 2,
                 activation_setting: str | dict[str, dict] = "ReLU",
                 batchnorm2d_setting: Optional[dict] = {'eps': 1e-05,
                                                        'momentum': 0.1,
                                                        'affine': True,
                                                        'track_running_stats': True},
                 bn_before_or_after: str = 'before',
                 weight_init: str | dict[str, dict] = "kaiming_normal",
                 dropout_rate: float = 0.3):
        # store init argument values so we can use them to reset/re-initialize object
        self._init_args = {k:v for k,v in locals().items() if k in _get_call_args(self.__class__)}

        super().__init__(weight_init = weight_init)

        # store parameters
        self.in_time_steps = in_time_steps
        self.out_time_steps = out_time_steps
        self.in_features = in_features
        self.out_features = out_features
        self.time_mix_only = time_mix_only
        self.fmixer_hidden_size = fmixer_hidden_size

        #### initialize layers
        if self.time_mix_only:
            # if time mixing only, the mixer only consists of time mixing component
            self.mixers = torch.nn.ModuleList([_TimeMixing(in_time_steps = in_time_steps,
                                                           activation_setting = activation_setting,
                                                           batchnorm2d_setting = batchnorm2d_setting,
                                                           bn_before_or_after = bn_before_or_after,
                                                           weight_init = weight_init,
                                                           dropout_rate = dropout_rate) for i in range(n_mixer)])
        else:
            # apply both time and feature mixing
            self.mixers = torch.nn.ModuleList([_Mixer(in_time_steps = in_time_steps,
                                                      in_features = in_features,
                                                      fmixer_hidden_size = fmixer_hidden_size,
                                                      out_features = in_features,
                                                      activation_setting = activation_setting,
                                                      batchnorm2d_setting = batchnorm2d_setting,
                                                      bn_before_or_after = bn_before_or_after,
                                                      weight_init = weight_init,
                                                      dropout_rate = dropout_rate) for i in range(n_mixer)])

        # construct feature projection layer to generate required number of output time series
        if self.in_features == self.out_features:
            self.feature_projection_layer = None
        else:
            self.feature_projection_layer = _initialize_linear_layer(in_features = in_features,
                                                                     out_features = out_features,
                                                                     weight_init = self.weight_init)

        # construct temporal projection layer to generate desired number of forecasted time steps
        self.temporal_projection_layer = _initialize_linear_layer(in_features = in_time_steps,
                                                                  out_features = out_time_steps,
                                                                  weight_init = self.weight_init)

    @property
    def in_time_steps(self):
        return self._input_time_steps

    @in_time_steps.setter
    def in_time_steps(self, in_time_steps: int):
        '''setter with data check'''
        if not isinstance(in_time_steps, int):
            raise ValueError(f'<in_time_steps> has to be an integer >= 1. Got {in_time_steps} instead.')
        elif in_time_steps < 1:
            raise ValueError(f'<in_time_steps> has to be an integer >= 1. Got {in_time_steps} instead.')

        self._input_time_steps = in_time_steps

    @property
    def out_time_steps(self):
        return self._output_time_steps

    @out_time_steps.setter
    def out_time_steps(self, out_time_steps: int):
        '''setter with data check'''
        if not isinstance(out_time_steps, int):
            raise ValueError(f'<out_time_steps> has to be an integer >= 1. Got {out_time_steps} instead.')
        elif out_time_steps < 1:
            raise ValueError(f'<out_time_steps> has to be an integer >= 1. Got {out_time_steps} instead.')

        self._output_time_steps = out_time_steps

    @property
    def in_features(self):
        return self._in_features

    @in_features.setter
    def in_features(self, in_features: int):
        '''setter with data check'''
        if not isinstance(in_features, int):
            raise ValueError(f'<in_features> has to be an integer >= 1. Got {in_features} instead.')
        elif in_features < 1:
            raise ValueError(f'<in_features> has to be an integer >= 1. Got {in_features} instead.')

        self._in_features = in_features

    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: int):
        '''setter with data check'''
        if out_features == 'default':
            out_features = self.in_features

        if not isinstance(out_features, int):
            raise ValueError(f'<out_features> has to be an integer >= 1. Got {out_features} instead.')
        elif out_features < 1:
            raise ValueError(f'<out_features> has to be an integer >= 1. Got {out_features} instead.')

        self._out_features = out_features

    @property
    def fmixer_hidden_size(self):
        return self._fmixer_hidden_size

    @fmixer_hidden_size.setter
    def fmixer_hidden_size(self, fmixer_hidden_size: int):
        '''setter with data check'''
        if fmixer_hidden_size == 'default':
            fmixer_hidden_size = self.in_features

        if not isinstance(fmixer_hidden_size, int):
            raise ValueError(f'<fmixer_hidden_size> has to be an integer >= 1. Got {fmixer_hidden_size} instead.')
        elif fmixer_hidden_size < 1:
            raise ValueError(f'<fmixer_hidden_size> has to be an integer >= 1. Got {fmixer_hidden_size} instead.')

        self._fmixer_hidden_size = fmixer_hidden_size

    @property
    def time_mix_only(self):
        return self._time_mix_only

    @time_mix_only.setter
    def time_mix_only(self, time_mix_only: bool):
        '''setter with data check'''
        if not isinstance(time_mix_only, bool):
            raise ValueError(f'<time_mix_only> has to be either True or False. Got {time_mix_only} instead.')

        if self.in_features == 1:
            if time_mix_only is False:
                print('Switch to time-mix-only mode since there is only 1 feature.')
                time_mix_only = True

        self._time_mix_only = time_mix_only

    def forward(self, x: torch.Tensor):
        '''forward method for defining network structure (how data flows)'''

        if self.time_mix_only:
            x = torch.transpose(x, dim0 = -1, dim1 = -2)

        for mixer in self.mixers:
            x = mixer(x)

        if self.time_mix_only:
            x = torch.transpose(x, dim0 = -1, dim1 = -2)

        if self.feature_projection_layer is not None:
            x = self.feature_projection_layer(x)

        x = torch.transpose(x, dim0 = -1, dim1 = -2)
        x = self.temporal_projection_layer(x)
        x = torch.transpose(x, dim0 = -1, dim1 = -2)

        return x

    @property
    def input_shape(self):
        return torch.Size([-1, self.in_time_steps, self.in_features])

    @property
    def input_dtype(self):
        return torch.float32

    def reset(self):
        '''method for re-initializing the model parameters'''
        self.__init__(**self._init_args)


# class TSmixer_ext(TSmixer):
#     ''' Class for constructing TSmixer_ext, which allows the use of static features

#         Authors(s): denns.liang@hilton.com

#         init args
#         ----------
#         in_time_steps (int): number of time steps in time series used for training
#         out_time_steps (int): number of forecasted time steps
#         in_features (int): number of features (time series) going into the MLP
#         out_features (int): number of feature (time serie) outputs
#         fmixer_hidden_size (int): number of nodes in the hiddent layer for feature mixing component
#         static_features (bool): number of static features
#         n_mixer (int): number of mixing layers
#         activation_setting (str or dict): activation function setting.
#                                           Available choice: ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
#                                                                   'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
#                                                                   'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention',
#                                                                   'PReLU', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
#                                   Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'LeakyReLU': {'negative_slope': 0.01}}.
#         batchnorm2d_setting (dict): a dict for storing setting for batchnorm2d. e.g {'eps': 1e-05,
#                                                                                      'momentum': 0.1,
#                                                                                      'affine': True,
#                                                                                      'track_running_stats': True}.
#                                     For details, please see arguments for _BN2d
#         bn_before_or_after (str): 'before' or 'after'. indicate whether to add batchnorm2d before or after the MLP layers
#         weight_init (str or dict): nn weights initialization method. Available choice: ['uniform', 'normal', 'trunc_normal', 'constant',
#                                                                                  'ones', 'zeros', 'eye', 'dirac', 'xavier_uniform',
#                                                                                  'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
#                                                                                  'orthogonal', 'sparse']
#                                   Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'kaiming_normal': {'a': 0, 'nonlinearity': 'leaky_relu'}}.
#         dropout_rate (float): between 0 and 1; 0 for not using dropout
#     '''
#     def __init__(self,
#                  in_time_steps: int,
#                  out_time_steps: int,
#                  in_features: int,
#                  static_features: int,
#                  out_features: int = 'default',
#                  fmixer_hidden_size: int = 'default',
#                  n_mixer: int = 2,
#                  activation_setting: str | dict[str, dict] = "ReLU",
#                  batchnorm2d_setting: Optional[dict] = {'eps': 1e-05,
#                                                         'momentum': 0.1,
#                                                         'affine': True,
#                                                         'track_running_stats': True},
#                  bn_before_or_after: str = 'before',
#                  weight_init: str | dict[str, dict] = "kaiming_normal",
#                  dropout_rate: float = 0.3):
#         # store init argument values so we can use them to reset/re-initialize object
#         init_args = {k:v for k,v in locals().items() if k in _get_call_args(self.__class__)}

#         super().__init__(in_time_steps = in_time_steps,
#                          out_time_steps = out_time_steps,
#                          in_features = in_features,
#                          out_features = out_features,
#                          fmixer_hidden_size = fmixer_hidden_size,
#                          time_mix_only = True, # no use for this argument. But, set to True to avoid generating warning when <in_features> == 1
#                          n_mixer = n_mixer,
#                          activation_setting = activation_setting,
#                          batchnorm2d_setting = batchnorm2d_setting,
#                          bn_before_or_after = bn_before_or_after,
#                          weight_init = weight_init,
#                          dropout_rate = dropout_rate)

#         self._init_args = init_args # need to assign after super().__init__ otherwise the parent clas __ini__ will overwrite the value

#         # delete un-used attributes inherited from the TSmixer class
#         delattr(self, '_time_mix_only')

#         # store additional parameters
#         self.static_features = static_features

#         #### initialize layers
#         self.mixers = torch.nn.ModuleList([_MixerConditional(in_time_steps = in_time_steps,
#                                                              in_features = in_features,
#                                                              fmixer_hidden_size = fmixer_hidden_size,
#                                                              out_features = in_features,
#                                                              static_features = static_features,
#                                                              activation_setting = activation_setting,
#                                                              batchnorm2d_setting = batchnorm2d_setting,
#                                                              bn_before_or_after = bn_before_or_after,
#                                                              weight_init = weight_init,
#                                                              dropout_rate = dropout_rate) for i in range(n_mixer)])

#         # construct feature projection layer to generate required number of output time series
#         if self.in_features == self.out_features:
#             self.feature_projection_layer = None
#         else:
#             self.feature_projection_layer = _initialize_linear_layer(in_features = in_features,
#                                                                      out_features = out_features,
#                                                                      weight_init = self.weight_init)

#         # construct temporal projection layer
#         self.temporal_projection_layer = _initialize_linear_layer(in_features = in_time_steps,
#                                                                   out_features = out_time_steps,
#                                                                   weight_init = self.weight_init)

#     @property
#     def static_features(self):
#         return self._static_features

#     @static_features.setter
#     def static_features(self, static_features: int):
#         '''setter with data check'''
#         if not isinstance(static_features, int):
#             raise ValueError(f'<static_features> has to be an integer >= 1. Got {static_features} instead.')
#         elif static_features < 1:
#             raise ValueError(f'<static_features> has to be an integer >= 1. Got {static_features} instead.')

#         self._static_features = static_features

#     def forward(self, x: torch.Tensor):
#         '''forward method for defining network structure (how data flows)'''

#         # Since the input is a tuple of 2 tensors (time series input and static var input), seperate them first
#         X, static = x[0], x[1]

#         for mixer in self.mixers:
#             X = mixer([X, static]) # mixer requires both time seris and static var inputs

#         if self.feature_projection_layer is not None:
#             # if specified input and output feature shape are different, use a simple liner layer to project the result to the desired output shape
#             X = self.feature_projection_layer(X)

#         X = torch.transpose(X, dim0 = -1, dim1 = -2)
#         X = self.temporal_projection_layer(X) # project to the desired output time steps
#         X = torch.transpose(X, dim0 = -1, dim1 = -2)

#         return X

#     @property
#     def input_shape(self):
#         return torch.Size([-1, self.in_time_steps, self.in_features]), torch.Size([-1, self.in_time_steps, self.static_features])

#     @property
#     def input_dtype(self):
#         return torch.float32, torch.float32

#     def reset(self):
#         '''method for re-initializing the model parameters'''
#         self.__init__(**self._init_args)



class TSmixer_ext(TSmixer):
    ''' Class for constructing TSmixer_ext, which allows the use of static features

        Authors(s): denns.liang@hilton.com

        init args
        ----------
        in_time_steps (int): number of time steps in time series used for training
        out_time_steps (int): number of forecasted time steps
        in_features (int): number of features (time series) going into the MLP
        out_features (int): number of feature (time serie) outputs
        fmixer_hidden_size (int): number of nodes in the hiddent layer for feature mixing component
        static_features (bool): number of static features
        n_mixer (int): number of mixing layers
        activation_setting (str or dict): activation function setting.
                                          Available choice: ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
                                                                  'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
                                                                  'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention',
                                                                  'PReLU', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'LeakyReLU': {'negative_slope': 0.01}}.
        batchnorm2d_setting (dict): a dict for storing setting for batchnorm2d. e.g {'eps': 1e-05,
                                                                                     'momentum': 0.1,
                                                                                     'affine': True,
                                                                                     'track_running_stats': True}.
                                    For details, please see arguments for _BN2d
        bn_before_or_after (str): 'before' or 'after'. indicate whether to add batchnorm2d before or after the MLP layers
        weight_init (str or dict): nn weights initialization method. Available choice: ['uniform', 'normal', 'trunc_normal', 'constant',
                                                                                 'ones', 'zeros', 'eye', 'dirac', 'xavier_uniform',
                                                                                 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
                                                                                 'orthogonal', 'sparse']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'kaiming_normal': {'a': 0, 'nonlinearity': 'leaky_relu'}}.
        dropout_rate (float): between 0 and 1; 0 for not using dropout
    '''
    def __init__(self,
                 in_time_steps: int,
                 out_time_steps: int,
                 in_features: int,
                 other_dynamic_features: int, 
                 static_features: int,
                 out_features: int = 'default',
                 fmixer_hidden_size: int = 'default',
                 n_mixer: int = 2,
                 activation_setting: str | dict[str, dict] = "ReLU",
                 batchnorm2d_setting: Optional[dict] = {'eps': 1e-05,
                                                        'momentum': 0.1,
                                                        'affine': True,
                                                        'track_running_stats': True},
                 bn_before_or_after: str = 'before',
                 weight_init: str | dict[str, dict] = "kaiming_normal",
                 dropout_rate: float = 0.3):
        # store init argument values so we can use them to reset/re-initialize object
        init_args = {k:v for k,v in locals().items() if k in _get_call_args(self.__class__)}

        super().__init__(in_time_steps = in_time_steps,
                         out_time_steps = out_time_steps,
                         in_features = in_features,
                         out_features = out_features,
                         fmixer_hidden_size = fmixer_hidden_size,
                         time_mix_only = True, # no use for this argument. But, set to True to avoid generating warning when <in_features> == 1
                         n_mixer = n_mixer,
                         activation_setting = activation_setting,
                         batchnorm2d_setting = batchnorm2d_setting,
                         bn_before_or_after = bn_before_or_after,
                         weight_init = weight_init,
                         dropout_rate = dropout_rate)

        self._init_args = init_args # need to assign after super().__init__ otherwise the parent clas __ini__ will overwrite the value

        # delete un-used attributes inherited from the TSmixer class
        delattr(self, '_time_mix_only')

        # store additional parameters
        self.other_dynamic_features = other_dynamic_features
        self.static_features = static_features

        if self.other_dynamic_features > 0:
            self.align_other_dynamic = _FeatureMixing(in_features = other_dynamic_features,
                                                    hidden_layer_size = other_dynamic_features,
                                                    out_features = in_features, # force to have the same shape as the time series input
                                                    activation_setting = activation_setting,
                                                    batchnorm2d_setting = batchnorm2d_setting,
                                                    bn_before_or_after = bn_before_or_after,
                                                    weight_init = weight_init,
                                                    dropout_rate = dropout_rate)
        else:
            self.align_other_dynamic = None

        self.align_static = _FeatureMixing(in_features = static_features,
                                           hidden_layer_size = static_features,
                                           out_features = in_features, # force to have the same shape as the time series input
                                           activation_setting = activation_setting,
                                           batchnorm2d_setting = batchnorm2d_setting,
                                           bn_before_or_after = bn_before_or_after,
                                           weight_init = weight_init,
                                           dropout_rate = dropout_rate)


        #### initialize layers
        self.mixers = torch.nn.ModuleList([_Mixer(in_time_steps = in_time_steps,
                                                  in_features = in_features + in_features + in_features,
                                                  fmixer_hidden_size = fmixer_hidden_size,
                                                  out_features = in_features,
                                                  activation_setting = activation_setting,
                                                  batchnorm2d_setting = batchnorm2d_setting,
                                                  bn_before_or_after = bn_before_or_after,
                                                  weight_init = weight_init,
                                                  dropout_rate = dropout_rate) for i in range(n_mixer)])

        # construct feature projection layer to generate required number of output time series
        if self.in_features == self.out_features:
            self.feature_projection_layer = None
        else:
            self.feature_projection_layer = _initialize_linear_layer(in_features = in_features,
                                                                     out_features = out_features,
                                                                     weight_init = self.weight_init)

        # construct temporal projection layer
        self.temporal_projection_layer = _initialize_linear_layer(in_features = in_time_steps,
                                                                  out_features = out_time_steps,
                                                                  weight_init = self.weight_init)

    @property
    def other_dynamic_features(self):
        return self._other_dynamic_features

    @other_dynamic_features.setter
    def other_dynamic_features(self, other_dynamic_features: int):
        '''setter with data check'''
        if not isinstance(other_dynamic_features, int):
            raise ValueError(f'<static_features> has to be an integer >= 1. Got {other_dynamic_features} instead.')
        elif other_dynamic_features < 1:
            raise ValueError(f'<static_features> has to be an integer >= 1. Got {other_dynamic_features} instead.')

        self._other_dynamic_features = other_dynamic_features

    @property
    def static_features(self):
        return self._static_features

    @static_features.setter
    def static_features(self, static_features: int):
        '''setter with data check'''
        if not isinstance(static_features, int):
            raise ValueError(f'<static_features> has to be an integer >= 1. Got {static_features} instead.')
        elif static_features < 1:
            raise ValueError(f'<static_features> has to be an integer >= 1. Got {static_features} instead.')

        self._static_features = static_features

    def forward(self, x: torch.Tensor):
        '''forward method for defining network structure (how data flows)'''

        X, other_dynamic, static = x[0], x[1], x[2]

        # alignment phase; make sure the other inputs have the same shape as the time-series inputs
        dynamic_aligned = self.align_other_dynamic(other_dynamic)
        static_aligned = self.align_static(static)
        y = X

        for mixer in self.mixers:
            combined = torch.cat([y, dynamic_aligned, static_aligned], dim = -1)
            y = mixer(combined)

        if self.feature_projection_layer is not None:
            # if specified input and output feature shape are different, use a simple liner layer to project the result to the desired output shape
            y = self.feature_projection_layer(y)

        y = torch.transpose(y, dim0 = -1, dim1 = -2)
        y = self.temporal_projection_layer(y) # project to the desired output time steps
        y = torch.transpose(y, dim0 = -1, dim1 = -2)

        return y

    @property
    def input_shape(self):
        shape = [torch.Size([-1, self.in_time_steps, self.in_features])]

        if self.other_dynamic_features > 0:
            shape.append(torch.Size([-1, self.in_time_steps, self.other_dynamic_features]))

        if self.static_features:
            shape.append(torch.Size([-1, self.in_time_steps, self.static_features]))
        
        if len(shape) == 1:
            return shape[0]
        else:
            return shape

    @property
    def input_dtype(self):
        if isinstance(self.input_shape, torch.Size):
            return torch.float32
        else:
            return [torch.float32] * len(self.input_shape)

    def reset(self):
        '''method for re-initializing the model parameters'''
        self.__init__(**self._init_args)
