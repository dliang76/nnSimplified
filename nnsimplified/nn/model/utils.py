from typing import Optional, Union, Any
import inspect
import torch
from ..helper import _get_call_default_args, _weight_init

def _construct_activation(activation_setting: str | dict[str, dict]):
    '''function for constructing activation layer from a dict with name = activation type and value = settings'''

    # obtain available pytorch activation
    available = [k for k, v in torch.nn.modules.activation.__dict__.items() if
                callable(v) and v.__module__ == 'torch.nn.modules.activation']

    # extract info from activation_setting
    if isinstance(activation_setting, str):
        name = activation_setting # type of activation
        given_config = {} # given configuration

    elif isinstance(activation_setting, dict):
        if len(activation_setting) > 1:
            raise ValueError('Cannot specify more than 1 activation function!')

        name = list(activation_setting.keys())[0] # type of activation
        given_config = activation_setting[name] # given configuration

        if not isinstance(given_config, dict):
            raise ValueError('Activation function arguments must be specified in python dict format!')
    else:
        raise ValueError('<activation_setting> argument accepts only string (name of init method) or dict ({"name of activation": dictionary of activation args})')

    if name not in available:
        raise ValueError(f'{name} is not a valid activation function. Available options: {available}')

    call = eval(f'torch.nn.modules.activation.{name}') # get activation function call

    # construct activation function
    return call(**given_config)

def _initialize_linear_layer(in_features: int,
                             out_features: int,
                             weight_init: _weight_init,
                             device: str | torch.device = None) -> torch.nn.Linear:
    '''method for constructing a nn layer'''

    linear_layer = torch.nn.Linear(in_features = in_features,
                                   out_features = out_features,
                                   bias = True,
                                   device = device)

    # initialize weights and bias
    weight_init(linear_layer.weight)
    torch.nn.init.zeros_(linear_layer.bias)

    return linear_layer