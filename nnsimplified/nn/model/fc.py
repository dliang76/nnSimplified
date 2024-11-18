from typing import Dict, List, Optional, Union
import inspect
from ..helper import _get_call_default_args
from ..module.base import nnModule, baseModule
from ..module.assemble import nnParallel
from .utils import _construct_activation
import torch

class mlpBase(nnModule):
    ''' Class for constructing a multilayer perceptron structure quickly

        Authors(s): denns.liang@hilton.com

        init args
        ----------
        mlp_layers (list of int): list of layer width/dimension (the first element is the input dimension and the last element is the ouput dimension)
                                       e.g. [20, 10, 4, 1] is a fully connected network with input dimension = 20 and output dimension = 1 and 2 hidden layers.
        activation (str or dict): activation function. Available choice: ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
                                                                  'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
                                                                  'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention',
                                                                  'PReLU', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'LeakyReLU': {'negative_slope': 0.01}}.
        weight_init (str or dict): nn weights initialization method. Available choice: ['uniform', 'normal', 'trunc_normal', 'constant',
                                                                                 'ones', 'zeros', 'eye', 'dirac', 'xavier_uniform',
                                                                                 'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
                                                                                 'orthogonal', 'sparse']
                                  Can also use a dictionary to specify torch settings (see PyTorch site). E.g. {'kaiming_normal': {'a': 0, 'nonlinearity': 'leaky_relu'}}.
        batchnorm_setting (dict): dictionary for specifying batch norm settings in Torch (see PyTorch site). E.g. {'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True}
                                  Use None to not include batchnorm layers.
        dropout_rate (float): between 0 and 1; 0 for not using dropout
    '''
    def __init__(self,
                 mlp_layers: List[int] = [],
                 activation_setting: str | Dict[str, dict] = "ReLU",
                 weight_init: str | Dict[str, dict] = "kaiming_normal",
                 batchnorm_setting: dict = {'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True},
                 dropout_rate: float = 0.5):

        super().__init__(weight_init = weight_init)

        # mlp layer structure
        self._mlp_layers = mlp_layers # [input_size, hidden_layer1_size, hidden_layer2_size, ..., output_size]

        # construct and register linear layers
        self.add_module(name = 'linear_layers', module = self.construct_linear_layers(mlp_layers = mlp_layers))

        # construct activation
        self.activation_setting = activation_setting

        # construct batch normalization layers
        self.batchnorm_setting = batchnorm_setting

        # dropout setup
        self.dropout_rate = dropout_rate

    @property
    def activation_setting(self):
        '''getter for activation name'''
        return self._activation_setting

    @activation_setting.setter
    def activation_setting(self, activation_setting: str | Dict[str, dict]):
        '''setter for activation'''

        if activation_setting is None:
            self._activation_setting = None
        else:
            activation = _construct_activation(activation_setting)
            self.add_module(name = 'activation', module = activation)

    @property
    def batchnorm_setting(self):
        return self._batchnorm_setting

    @batchnorm_setting.setter
    def batchnorm_setting(self, batchnorm_setting):
        '''setter for constructing batch normalization layers'''
        # create batch normalization layer if specified

        if batchnorm_setting:
            # batch normalization
            self.add_module(name = 'batchnorms', module = torch.nn.ModuleList([torch.nn.BatchNorm1d(num_features = i, **batchnorm_setting)
                                                                               for i in self._mlp_layers[1:-1]]))
        else:
            self.batchnorms = None

        self._batchnorm_setting = batchnorm_setting

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, dropout_rate):
        '''setter for dropout layer'''
        # input check
        if not isinstance(dropout_rate, (float, int)):
            raise ValueError('The dropout rate has to be a number between 0 and 1.')

        if (dropout_rate >= 1) or (dropout_rate < 0):
            raise ValueError('The dropout rate has to be a number between 0 and 1.')
        elif dropout_rate > 0:
            dropouts = [torch.nn.Dropout(p = dropout_rate) for i in self._mlp_layers[1:-1]]
            self.add_module(name = 'dropout', module = torch.nn.ModuleList(dropouts))
        elif dropout_rate == 0:
            # no dropout if dropout_rate is 0
            self.dropout = None

        self._dropout_rate = dropout_rate

    def construct_linear_layers(self, mlp_layers) -> torch.nn.ModuleList:
        '''method for constructing nn layers'''

        # construct layers using module list
        linear_layers = torch.nn.ModuleList([torch.nn.Linear(in_features = mlp_layers[i],
                                                             out_features = mlp_layers[i + 1],
                                                             bias = True)
                                            for i in range(len(mlp_layers) - 1)])

        # initialize weights and bias
        for layer in linear_layers:
            self._weight_init(layer.weight)
            torch.nn.init.zeros_(layer.bias)

        return linear_layers

    def forward(self, x):
        '''forward method for defining network structure (how data flows)'''
        # move data through linear layers
        for idx, layer in enumerate(self.linear_layers):
            x = layer(x)

            # add batch normaliztion, activation and dropout for non-output layer (idx = # of layers - 1)
            if idx < (len(self.linear_layers) - 1):
                if self.batchnorms:
                    x = self.batchnorms[idx](x) # perform batch normalization

                x = self.activation(x) # apply activation

                if self.dropout:
                    x = self.dropout[idx](x) # apply dropout
        return x

    @property
    def input_shape(self):
        return torch.Size([-1, self._mlp_layers[0]])

    @property
    def input_dtype(self):
        '''method for model input dtype.'''
        return torch.float32 # torch.nn.Linear accept torch.float32

    def reset(self):
        '''method for re-initializing the model'''
        original_device = self.device # keep track of original computing device

        # re-construct linear layers (reset weight)
        self.linear_layers = self.construct_linear_layers()

        # re-construct batch normalization layers
        self.batchnorm_setting = self._batchnorm_setting

        self.dropout_rate = self._dropout_rate

        self.to(original_device) # move to original computing device


class mlp(mlpBase):
    def __init__(self,
                 nn_structure: List[int] = [],
                 activation_setting: str | Dict[str, dict] = "ReLU",
                 weight_init: str | Dict[str, dict] = "kaiming_normal",
                 batchnorm_setting: dict = {'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True},
                 dropout_rate: float = 0.3):

        self.nn_structure = nn_structure

        # initialize mlp structure
        super().__init__(mlp_layers = self._mlp_layers,
                         activation_setting = activation_setting,
                         weight_init = weight_init,
                         batchnorm_setting = batchnorm_setting,
                         dropout_rate = dropout_rate)

        if nn_structure:
            # register input layer as a nnParallel object if it is of format list, tuple, or dict
            if isinstance(nn_structure[0], torch.Size):
                self.input_layer = nn_structure[0][-1]
            elif isinstance(nn_structure[0], (list, tuple, dict)):
                self.add_module(name = 'input_layer', module = nnParallel(module_list = nn_structure[0], output_combine_method = 'concat'))
            else:
                self.input_layer = nn_structure[0]
        else:
            self.input_layer = None

    @property
    def nn_structure(self):
        return self._nn_structure

    @nn_structure.setter
    def nn_structure(self, nn_structure):
        # store nn_structure
        self._nn_structure = nn_structure

        if self._nn_structure:
            input_layer = nn_structure[0]

            # get input layer shape
            if isinstance(input_layer, int):
                n_nodes = input_layer
            elif isinstance(input_layer, torch.Size):
                n_nodes = input_layer[-1]
            elif isinstance(input_layer, (list, tuple, dict)):
                n_nodes = 0
                if isinstance(input_layer, dict):
                    input_layer = list(input_layer.values())

                for m in input_layer:
                    if isinstance(m, int):
                        n_nodes += m
                    elif isinstance(m, torch.Size):
                        n_nodes += m[-1]
                    else:
                        n_nodes += m.output_shape[-1]
            else:
                raise ValueError('Unknown structure found in the first layer input; must be an int or a list of int, a list of nn modules or a dict of nn modules with names.')

            self._mlp_layers = [n_nodes] + nn_structure[1:]
        else:
            self._mlp_layers = []

    def forward(self, x):
        '''forward method for defining network structure (how data flows)'''

        if isinstance(self.input_layer, nnModule):
            # if input layer structure is not a simple linear layer, run through input layer first
            x = self.input_layer(x)

            if isinstance(x, (list,tuple)):
                x = torch.cat(x, dim = -1)

        x = super().forward(x)

        return x

    @property
    def input_shape(self):
        if isinstance(self.input_layer, int):
            return torch.Size([-1, self.input_layer])
        elif isinstance(self.input_layer, baseModule):
            return self.input_layer.input_shape

    @property
    def input_dtype(self):
        '''method for model input dtype; must be implemented.'''

        if isinstance(self.input_layer, int):
            return torch.float32
        elif isinstance(self.input_layer, baseModule):
            return self.input_layer.input_dtype

    def reset(self):
        '''method for re-initializing the model'''
        original_device = self.device # keep track of original computing device

        # reset input structure if it contains nn modules (e.g embedding layers)
        if isinstance(self.input_layer, nnModule):
            self.input_layer.reset()

        # re-construct linear layers (reset weight)
        self.linear_layers = self.construct_linear_layers(mlp_layers = self._mlp_layers)

        # re-construct batch normalization layers
        self.batchnorm_setting = self._batchnorm_setting

        self.dropout_rate = self._dropout_rate

        self.to(original_device) # move to original computing device