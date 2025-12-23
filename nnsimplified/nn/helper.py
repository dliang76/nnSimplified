import os
import torch
from typing import Union, Callable
import inspect
import functools


def _get_call_args(call: Callable) -> dict:
    return list(inspect.signature(call).parameters.keys())


def _get_call_default_args(call: Callable) -> dict:
    return {
        k: v.default
        for k, v in inspect.signature(call).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def _extract_object_config_info(
    object_name_args: Union[str, dict[str, dict]],
) -> tuple[str, dict]:
    """Simple helper func for parsing configurations for loss, optmizer or lr scheduler.
    The function takes object_config (a dictionary of the form {name: arg_dict})
    e.g
        {'CrossEntropyLoss': {'weight': torch.FloatTensor([1,2,3]), 'reduction': 'none'}}
    and returns name and argument dictionary
       'CrossEntropyLoss' ,  {'weight': torch.FloatTensor([1,2,3]), 'reduction': 'none'}}

    If object_config only contains str name, then name and an empty {} are returned; this will force default argument values

    Author(s): dliang1122@gmail.com

    Arg
    -----
    object_name_args (str, dict): configuration input. Can be simply name (e.g loss function name, optimizer name etc)
                              or {name: {arg: value}} dictionary for specifying arguments loss, optmizer or lr scheduler.
                              If only name is specified, the object will just use default values.

    Return
    -----
    name (str) and config (dict)
    """
    # extract info
    if isinstance(object_name_args, dict):
        name = list(object_name_args)[0]
        arg_dict = object_name_args[name]
    elif isinstance(object_name_args, str):
        name = object_name_args
        arg_dict = {}
    else:
        raise ValueError(
            "Format not recognized. The object_config argument must be either a str name or a dictionary with the str name as the key and its arguments (dict) as the value"
        )

    return name, arg_dict


class _weight_init:
    # available pytorch weight initialization methods
    available = [
        k[:-1]
        for k, v in torch.nn.init.__dict__.items()
        if k.endswith("_") and callable(v) and not k.startswith("_")
    ]

    def __init__(self, weight_init: str | dict[str, dict]):
        self.weight_init = weight_init

    @property
    def weight_init(self):
        """getter for weight_init method name"""
        return self._weight_init

    @weight_init.setter
    def weight_init(self, weight_init: str | dict[str, dict]):
        """setter for weight_init method"""

        # check input format
        if isinstance(weight_init, str):
            name = weight_init
            give_config = {}

        elif isinstance(weight_init, dict):
            if len(weight_init) > 1:
                raise ValueError(
                    "Cannot specify more than 1 weight initialization method!"
                )

            name = list(weight_init.keys())[0]
            give_config = weight_init[name]

            if not isinstance(give_config, dict):
                raise ValueError(
                    "Weight initialization arguments must be specified in python dict format!"
                )
        else:
            raise ValueError(
                'weight_init accepts only string (name of init method) or dict ({"name of init method": dictionary of init method args})'
            )

        # check input validity
        if name not in self.available:
            raise ValueError(
                f"{name} is not a valid initialization method. Available: {self.available}"
            )

        # get function
        call = eval(f"torch.nn.init.{name}_")
        # get function default arguments
        default_config = _get_call_default_args(call)

        ### Check config arguments
        # check whether the function accepts keyword arguments
        accept_kwargs = (
            "**" in list(inspect.signature(call).parameters.values())[-1].__str__()
        )

        if not accept_kwargs:
            unexpected_args = set(give_config) - set(default_config)

            if unexpected_args:
                raise TypeError(
                    f"'{name}' method has no argument {', '.join(unexpected_args)}!"
                )

        # store config
        self._config = default_config | give_config

        # create weight init function
        self._weight_init = functools.partial(call, **self._config)
        functools.update_wrapper(self._weight_init, call)

    def __call__(self, tensor: torch.Tensor):
        return self._weight_init(tensor)

    def __repr__(self):
        config_str = ", ".join(
            [
                f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}"
                for k, v in self._config.items()
            ]
        )

        result = f"{self._weight_init}\nInit Method: {self._weight_init.__name__}\nConfig: ({config_str})"

        return result


def load_torch_object(path: str, device: Union[str, torch.device] = "cpu"):
    """method for loading object that was saved using torch; load into cpu device only for safety

    Author(s): dliang1122@gmail.com

    Args
    ----------
    path (str): object path
    device (str): 'cpu' or gpu device

    Returns
    -------
    torch objects
    """

    return torch.load(f=path, map_location=device)


def save_torch_object(torch_obj, path: str):
    """method for saving torch object

    Author(s): dliang1122@gmail.com

    Args
    ----------
    torch_obj (anything picklable): torch objects. E.g. model or model/optimizer parameter (state dict)
    path (str): save path

    Returns
    -------
    None
    """

    # create directory if not found
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # save to local
    torch.save(torch_obj, path)


def load_torch_model(path: str, device: Union[str, torch.device] = "cpu"):
    """wrapper for loading torch model

    Author(s): dliang1122@gmail.com

    Args
    ----------
    path (str): object path

    Returns
    -------
    torch.nn.module
    """

    # load model object
    model = load_torch_object(path=path, device=device)

    # set to eval mode for safety
    model.eval()

    return model
