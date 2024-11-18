from ..helper import save_torch_object, load_torch_object, load_torch_model, _get_call_default_args, _get_call_args
from ..metrics.helper import calc_metrics, metrics_higher_better_regression, metrics_higher_better_classification
from ..metrics.helper import metrics_lower_better_regression, metrics_lower_better_classification, metrics_non_directional
from ..metrics.helper import all_supported_metrics
from typing import Optional, Union, Dict, List, Tuple, Optional, Any
import string
import random
from awstools.s3 import s3_object_exists, list_s3_dir, delete_s3_objects
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from ...dataloading.dataloader import DataLoader, _DataLoader
from ..model import mlp
from ..metrics.reporting import trainingReport
import torch
from torch.optim.optimizer import Optimizer
from .. import additional_loss
import inspect
import shutil
from datetime import datetime
# from tempfile import TemporaryDirectory
# import mlflow

# for jupyter notebooks
from IPython.display import clear_output, display
from IPython import get_ipython

class _optimizer_init:

    def __init__(self, model, optimizer_config: Union[Optimizer| str, Dict[str | Optimizer, dict]], **kwargs):

        # self.optimizer = {optimizer_name: kwargs}
        self._construct(model, optimizer_config = optimizer_config, **kwargs)

    @property
    def supported_optimizer(cls):
        '''return supported optimizers'''
        return {k:v for k, v in torch.optim.__dict__.items() if type(v) == type and v.__base__ == Optimizer}

    def _construct(self, model, optimizer_config, **kwargs):

        if isinstance(optimizer_config, str):
            name, config = optimizer_config, {} | kwargs
        elif inspect.isclass(optimizer_config) and issubclass(optimizer_config, Optimizer):
            name, config = optimizer_config.__name__, {} | kwargs
        elif isinstance(optimizer_config, dict):
            name = list(optimizer_config)[0]
            config = optimizer_config[name] | kwargs

            if inspect.isclass(name) and issubclass(name, Optimizer):
                name = name.__name__
        else:
            raise ValueError("Invalid optimizer format; Valid formats are a torch Optimizer, a str(name of optimizer) or dict ({name_of_optimizer: {parameter_name: value}})")

        if name in self.supported_optimizer:
            # construct optimizer
            self._optimizer = self.supported_optimizer[name](model.parameters(), **config)

        else:
            raise ValueError(f"Optimizer not found. Valid options: {list(self.supported_optimizer)}")

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def name(self):
        return self.optimizer.__class__.__name__

    @property
    def settings(self):
        return self._optimizer.defaults

    def _to_device(self, optimizer_state, device):
        '''recursive method for moving optimizer state parameters to the specified device'''
        for k,v in optimizer_state.items():
            if isinstance(v, torch.Tensor):
                optimizer_state[k] = v.to(device)
            elif isinstance(v, dict):
                self._to_device(v, device)

    def to(self, device):
        '''wrapper method for moving optimizer to a specific device'''
        self._to_device(optimizer_state = self._optimizer.state_dict(), device = device)

    def __repr__(self):

        return f"{self._optimizer.__class__.__name__}(params, {', '.join([f'{k} = {v}' for k,v in self.settings.items()])})"

class _loss_init:

    def __init__(self,
                 loss_config: Union[torch.nn.modules.loss._Loss,
                                    additional_loss._CustomLossBase,
                                    str,
                                    Dict[str, dict]]
                , **kwargs):
        self._construct(loss_config = loss_config, **kwargs)

    @property
    def supported_loss(cls):
        '''return supported loss functions'''
        # metric learning loss currently not supported; they do not fit in current framework
        not_supported = 'TripletMarginLoss', 'TripletMarginWithDistanceLoss', 'MarginRankingLoss', 'CosineEmbeddingLoss'

        torch_loss = {k:v for k, v in torch.nn.modules.loss.__dict__.items() if type(v) == type and
                                                                                not k.startswith('_') and
                                                                                torch.nn.modules.loss._Loss in getattr(v, '__mro__', ()) and
                                                                                k not in  not_supported}

        custom_loss = {k:v for k, v in additional_loss.__dict__.items() if callable(v) and
                                                                           not k.startswith('_') and
                                                                           additional_loss._CustomLossBase in getattr(v, '__mro__', ())}

        supported = {**torch_loss, **custom_loss} # if loss funcs in custom_loss will overload loss funcs in torch_loss if they share the same name

        return supported

    @property
    def loss(self):
        return self._loss

    @property
    def name(self):
        return self.loss.__class__.__name__

    @property
    def settings(self):
        return self._settings

    def _construct(self,
                   loss_config: Union[torch.nn.modules.loss._Loss,
                                      additional_loss._CustomLossBase,
                                      str,
                                      Dict[str, dict]],
                   **kwargs):

        # validate loss_conif argument
        if isinstance(loss_config, torch.nn.modules.module.Module):
            # if loss_config is already a torch.nn.modules.module.Module (all loss classes inherit Module class), pass it to self._loss_ directly

            self._loss = loss_config

            if self._loss.reduction != 'sum':
                # only use sum for reduction; will calculae average loss on our own for accurate results (needed for class-weighted average)
                print("For loss function, swithing reduction to 'sum'. Will average using our own method for accurate results.")
                self._loss.reduction = 'sum'

            default_config = _get_call_args(loss_config.__class__) # get default config
            self._settings = {k:v for k,v in loss_config.__dict__.items() if k in default_config} # keep only non-deprecated arguments
        else:
            if isinstance(loss_config, str):
                name, config = loss_config, {'reduction': 'sum'} | kwargs

            elif isinstance(loss_config, dict):
                name = list(loss_config)[0]
                config = loss_config[name] | kwargs

                if 'reduction' in config:
                    if config['reduction'] != 'sum':
                        # only use sum for reduction; will calculae average loss on our own for accurate results (needed for class-weighted average)
                        print("For loss function, swithing reduction to 'sum'. Will average using our own method for accurate results.")

                config['reduction'] = 'sum'

            else:
                raise ValueError("Unsupported loss function format. Please provide your own loss function object or a str(name of loss function) or dict ({name_of_loss: {parameter_name: value}})")

            if name in self.supported_loss:
                init_config = _get_call_default_args(self.supported_loss[name]) | config # update with config provided
                #init_config = {k:v  for k,v in init_config.items() if k in self.supported_loss[name]().__dict__} # filter out deprecated arguments

                self._loss = self.supported_loss[name](**init_config)
                self._settings = init_config
            else:
                raise ValueError(f"Unsupported loss function. Valid options: {list(self.supported_loss)}")

    def __repr__(self):

        return f"{self.loss.__class__.__name__}({', '.join([f'{k} = {v}' for k,v in self.settings.items()])})"


class _lr_scheduler_init():

    def __init__(self,
                 optimizer: Optimizer,
                 scheduler_config: Union[str, Dict[str, dict], List[Dict[str, dict]]],
                 **kwargs):

        self._construct(optimizer = optimizer, scheduler_config = scheduler_config, **kwargs)

    @property
    def supported_lr_scheduler(self):
        '''return supported learning rate schedulers'''
        return {k:v for k, v in torch.optim.lr_scheduler.__dict__.items() if type(v) == type
                                                                             and not k.startswith('_')
                                                                             and (v.__base__ == torch.optim.lr_scheduler.LRScheduler
                                                                                  or
                                                                                  k == 'ReduceLROnPlateau')} # ReduceLROnPlateau scheduler has a different base class
    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def name(self):
        return self.lr_scheduler.__class__.__name__

    @property
    def settings(self):
        return self._settings

    def _construct(self, optimizer: Optimizer, scheduler_config: Union[str, Dict[str, dict], List[Dict[str, dict]]], **kwargs):
        '''Construct lr schedulers; take scheduler name or a dict of scheduler names and settings or a list of dict of
        scheduler names and settings (scheduler chaining) to create lr scheduler object(s)'''

        if scheduler_config is None:
            self._lr_scheduler = None
            self._settings = {}
        else:
            # validate lr_scheduler argument
            if isinstance(scheduler_config, list):
                # scheduler chaining
                schedulers = []
                self._settings = []

                # loop through scheduler info provided and construct each scheduler
                for s in scheduler_config:
                    if isinstance(s, str):
                        name = s
                        config = {}
                    elif isinstance(s, dict):
                        name = list(s)[0]
                        config = s[name]

                    scheduler_obj = self.supported_lr_scheduler[name]
                    default_args = _get_call_default_args(scheduler_obj)
                    config = default_args | config | kwargs
                    config = {k:v for k, v in config.items() if k in _get_call_args(scheduler_obj)}

                    if name in self.supported_lr_scheduler:
                        scheduler = scheduler_obj(optimizer, **config)
                    else:
                        # raise error if the specified scheduler is not in Torch library.
                        raise ValueError(f"The learning rate scheduler ({scheduler}) is not supported in Torch. Valid options: {self.supported_lr_scheduler}")

                    schedulers.append(scheduler)
                    self._settings.append({name: config} if config else name)


                # construct chained scheduler using the list of schedulers
                self._lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers)

            else:
                if isinstance(scheduler_config, str):
                    name = scheduler_config
                    config = {}
                elif isinstance(scheduler_config, dict):
                    name = list(scheduler_config)[0]
                    config = scheduler_config[name]

                scheduler_obj = self.supported_lr_scheduler[name]
                default_args = _get_call_default_args(scheduler_obj)
                config = default_args | config | kwargs
                config = {k:v for k, v in config.items() if k in _get_call_args(scheduler_obj)}

                if name in self.supported_lr_scheduler:
                    self._lr_scheduler = scheduler_obj(optimizer, **config)
                    self._settings = config
                else:
                    # raise error if the specified scheduler is not in Torch library.
                    raise ValueError(f"The learning rate scheduler ({scheduler}) is not supported in Torch. Valid options: {self.supported_lr_scheduler}")

    def __repr__(self):
        if self.lr_scheduler is not None:
            return f"{self.lr_scheduler.__class__.__name__}(optimizer, {', '.join([f'{k} = {v}' for k,v in self.settings.items()])})"
        else:
            return 'None'

class nnTrainer():
    ''' Base trainer class for neural net

        Authors(s): dliang1122@gmail.com

        init args
        ----------
        model (torch model): torch model; typically a subclass of torch.nn.Module
        loss (str or dict): loss fucntion name or a dict with name as the key and settings (dict) as the values
                                    e.g.
                                        'CrossEntropyLoss'
                                    or
                                        {'CrossEntropyLoss': {'weight': torch.tensor([0.2, 0.4, 0.4])}}
        optimizer (str or dict): optimizer name or a dict with name as the key and settings (dict) as the values
                                    e.g.
                                        'AdamW'
                                    or
                                        {'AdamW': {lr: 0.01,  weight_decay: 0}}
        device (str or torch.device): computation device; e.g 'cpu' or 'cuda' for gpu
        init_weights (bool): whether to initialize the weights in the model. False if resume training
        tracked_metrics (list): list of metrics to track
        target_metric (str): target metric that we want to optimize
        target_optimization_mode (str): 'min' or 'max'. Decides whether we want to maximize or minimize target metric
        save_dir (str): director path for saving trainer items (model, trainer setting, metrics tracker)
        save_every_epoch (bool): whether to save trainer for every epoch
    '''

    def __init__(self,
                 model = mlp(),
                 loss: Union[torch.nn.modules.loss._Loss,
                             additional_loss._CustomLossBase,
                             str,
                             Dict[str, dict]] = 'MSELoss',
                 optimizer: Union[Optimizer, str, Dict[str | Optimizer, dict]] = "AdamW",
                 device: Union[str, torch.device] = 'cpu',
                 init_weights: bool = True,
                 tracked_metrics: Optional[Union[str, List[str]]] = None,
                 target_metric: Optional[str] = None,
                 target_optimization_mode: str = 'min',
                 save_dir: Optional[str] = None,
                 save_every_epoch: bool = True):

        self.device = device
        self.model = model.to(device)

        # initialize configuration dictionary (for optimizer, loss, learning rate scheduler etc.)
        self.optimizer = optimizer
        self.loss = loss
        self.lr_scheduler = None # initialize this to none. Provided during fitting.

        if init_weights:
            self.reset_components(components = ['model', 'optimizer', 'loss', 'lr_scheduler'])

        # initialize metrics tracking
        self.tracked_metrics = tracked_metrics
        self.target_metric = target_metric
        self.target_optimization_mode = target_optimization_mode
        self._epoch_metrics = dict()
        self._metrics_tracker = trainingReport()

        self._best_epoch = None
        self._best_target_metric = None

        # save related
        self.save_dir = save_dir # store log uri
        self.save_every_epoch = save_every_epoch # whether to log every epoch

        # initialize epoch trackers
        self.epoch = -1

    def __getattr__(self, name: str):
        '''get attribute method to get config settings'''
        if '_config' in self.__dict__:
            _configs = self.__dict__['_config']
            if name in _configs:
                return _configs[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __repr__(self):
        repr_text = str(self.__class__) + '\n'
        repr_text += '\n'.join([f'{k} = {v}' for k,v in self.setting.items()])

        return repr_text

    ########## Property and Setters ############
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if not isinstance(model, torch.nn.Module):
            raise ValueError('model has to be a torch nn module.')

        self._model = model

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        # check device
        if device == 'mps':
            if not torch.backends.mps.is_available():
                print('No mps device found!')
                device = 'cpu'
        elif device == 'cuda':
            if not torch.cuda.is_available():
                print('No cuda device found!')
                device = 'cpu'
        else:
            device = 'cpu'

        self._device = device

    @property
    def optimizer(self):
        '''return optimizer object'''
        return self._optimizer_init_obj._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Union[Optimizer | str, Dict[str | Optimizer, dict]]):

        self._optimizer_init_obj = _optimizer_init(model = self.model, optimizer_config = optimizer)

    @property
    def loss(self):
        return self._loss_init_obj._loss

    @loss.setter
    def loss(self, loss:Union[torch.nn.modules.loss._Loss,
                              additional_loss._CustomLossBase,
                              str,
                              Dict[str, dict]]):
        self._loss_init_obj = _loss_init(loss_config = loss)

        # move to specified device
        self._loss_init_obj._loss.to(self.device)

    @property
    def lr_scheduler(self):
        return self._lr_scheduler_init_obj._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, scheduler_config: Union[str, Dict[str, dict], List[Dict[str, dict]]]):

        self._lr_scheduler_init_obj = _lr_scheduler_init(optimizer = self.optimizer, scheduler_config = scheduler_config)

    @property
    def setting(self):
        '''Show current trainer settings'''

        trainer_setting = {"model": self.model,
                           "loss": self._loss_init_obj,
                           "optimizer": self._optimizer_init_obj,
                           "lr_scheduler": self._lr_scheduler_init_obj if self._lr_scheduler_init_obj.lr_scheduler is not None else None,
                           "epoch": self.epoch,
                           "tracked_metrics": self.tracked_metrics,
                           "target_metric": self.target_metric,
                           "target_optimization_mode": self.target_optimization_mode,
                           "save_dir": self.save_dir,
                           "save_every_epoch": self.save_every_epoch}

        return trainer_setting

    @property
    def tracked_metrics(self):
        '''getter for tracked metrics'''
        return self._tracked_metrics

    @tracked_metrics.setter
    def tracked_metrics(self, tracked_metrics):
        '''setter for tracked metrics with input check'''
        loss_type = self.loss.__class__.__name__

        if tracked_metrics is not None:
            # input check
            if isinstance(tracked_metrics, str):
                tracked_metrics = [tracked_metrics]

            if isinstance(tracked_metrics, (tuple, list)):
                for m in tracked_metrics:
                    if m not in (all_supported_metrics + [loss_type]):
                        raise ValueError(f'Metrics not supported. Supported metrics: {all_supported_metrics}')
            else:
                raise ValueError("<tracked_metrics> argument only accepts a string or a list of string or a dict with key = metric name and value = setting.")

            self._tracked_metrics = [loss_type] + [m for m in tracked_metrics if m != loss_type]
        else:
            print(f'No <tracked_metrics> specified. Will only track loss/objective function ({self.loss.__class__.__name__}).')
            self._tracked_metrics = [loss_type]

    @property
    def target_metric(self):
        '''getter for target metric'''
        return self._target_metric

    @target_metric.setter
    def target_metric(self, target_metric):
        '''setter for target_metric with input check'''
        if target_metric is not None:
            # input check
            if target_metric not in all_supported_metrics + [self.loss.__class__.__name__]:
                raise ValueError(f'Metrics not supported. Supported metrics: {all_supported_metrics}')

            self._target_metric = target_metric
        else:
            print(f'No <target_metric> specified. Will only use loss/objective function ({self.loss.__class__.__name__}) as target.')
            self._target_metric = self.loss.__class__.__name__

    @property
    def target_optimization_mode(self):
        '''getter for metric optimization mode'''
        return self._target_optimization_mode

    @target_optimization_mode.setter
    def target_optimization_mode(self, mode: str):
        '''setter for metric optimization mode with input check'''
        # input check
        if mode not in ('min', 'max'):
            raise ValueError("<target_optimization_mode> argument has to be either 'min' or 'max'.")

        if self.target_metric == self.loss.__class__.__name__:
            if mode == 'max':
                print("<target_optimization_mode> can only be 'min' when <target_metric> is the loss. Switch to 'min'...")

            self._target_optimization_mode = 'min'

        else:
            self._target_optimization_mode = mode

    @property
    def metrics_tracker(self):
        return self._metrics_tracker

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, save_dir):
        '''setter for save_dir'''

        if save_dir:
            if not (save_dir.startswith('s3:') or save_dir.startswith('hdfs:')):
                # get absolute path for local directory (not one of the following: s3, hdfs, mlflow uri address)
                self._save_dir = os.path.abspath(save_dir)
        else:
            self._save_dir = None

    @property
    def save_every_epoch(self):
        return self._save_every_epoch

    @save_every_epoch.setter
    def save_every_epoch(self, save_every_epoch):
        if not isinstance(save_every_epoch, bool):
            raise ValueError("<save_every_epoch> has to either be True or False.")

        self._save_every_epoch = save_every_epoch

    @property
    def current_lr(self):
        '''method to extract current learning rate'''
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_target_metric(self):
        return self._best_target_metric

    ########## End: Property and Setters ############

    def to(self, device: Union[str, torch.device]):
        '''move torch tensors to the specified device'''
        self.model.to(device)
        self._optimizer_init_obj.to(device)
        self.loss.to(device)
        self.device = device

    def reset_components(self, components: List[str] | str):
        '''method for resetting each components (e.g. model, optimizer, metrics tracking etc) in the trainer'''
        if isinstance(components, str):
            components = [components]

        # Do not re-arrange the order of components. Optimizer requires model as input and lr scheduler requires optimizer as input
        if 'model' in components:
            self.model.reset()

        if 'optimizer' in components:
            self.optimizer = {self._optimizer_init_obj.name: self._optimizer_init_obj.settings}

        if 'loss' in components:
            self.loss = {self._loss_init_obj.name: self._loss_init_obj.settings}

        if 'lr_scheduler' in components:
            self.lr_scheduler = {self._lr_scheduler_init_obj.name: self._lr_scheduler_init_obj.settings} if self.lr_scheduler is not None else None

        if 'metrics_tracker' in components:
            self._epoch_metrics = dict()
            self._metrics_tracker = trainingReport()
            self._best_epoch = None
            self._best_target_metric = None

    def reset(self):
        '''method for resetting the entire trainer'''
        # reset epoch counter
        self.epoch = -1

        # reset components (model, optimizer, loss, lr_scheduler and metrics tracker).
        # Order matters as one might require previous component(s) as inputs.
        self.reset_components(components = ['model', 'optimizer', 'loss', 'lr_scheduler', 'metrics_tracker'])

    ##### General saving and Loading methods ######
    def _save_trainer_setting(self, path: str):
        '''method for saving trainer settings'''
        # Get trainer setting. Exlude model; we'll save the model seperately
        trainer_setting = {k:v for k, v in self.setting.items() if k != 'model'}

        save_torch_object(torch_obj = trainer_setting, path = path)

    def _restore_trainer_setting(self, trainer_setting_dict):
        '''method for restoring trainer components'''
        # restore loss function
        self._loss_init_obj = trainer_setting_dict['loss']

        # restore optimizer
        self.optimizer = {trainer_setting_dict['optimizer'].name: trainer_setting_dict['optimizer'].settings}

        # restore lr scheduler
        if trainer_setting_dict.get('lr_scheduler'):
            self.lr_scheduler = {trainer_setting_dict['lr_scheduler'].name: trainer_setting_dict['lr_scheduler'].settings}
        else:
            self.lr_scheduler = None

        # restore metrics setting
        self.epoch = trainer_setting_dict['epoch']
        self.tracked_metrics = trainer_setting_dict["tracked_metrics"]
        self.target_metric = trainer_setting_dict["target_metric"]
        self.target_optimization_mode = trainer_setting_dict["target_optimization_mode"]

        # restore other setting
        self.save_dir = trainer_setting_dict["save_dir"]
        self.save_every_epoch = trainer_setting_dict["save_every_epoch"]

    def save(self, dir_path: str):
        '''method for saving a trainer'''
        # save model
        self.model.save(os.path.join(dir_path, 'model.pkl'))

        # save trainer setting
        self._save_trainer_setting(path = os.path.join(dir_path, 'trainer_setting.pkl'))

        # save metric tracker
        self.metrics_tracker.save(file_path = os.path.join(dir_path, 'metrics_tracker.pkl'))

    def _check_file(self, path):
        '''check whether a file exists'''
        if path.startswith('s3://'):
            return s3_object_exists(path)
        elif path.startswith('hdfs'):
            raise RuntimeError('To be implemented...')
        else:
            return os.path.exists(path)

    def _load_metrics_tracker(self, path):
        '''method for loading saved metrics tracker'''
        metrics_tracker = trainingReport() # initialize metrics tracker
        metrics_tracker.load(file_path = path) # load metrics tracker from local

        return metrics_tracker

    def _load(self,
              dir_path: str,
              device: Union[str, torch.device] ='cpu'):
        '''method for loading a saved trainer'''
        # load model
        self.model = load_torch_object(os.path.join(dir_path, 'model.pkl'), device = device)
        self.model.eval()

        # load saved trainer settings
        trainer_setting = load_torch_object(os.path.join(dir_path, 'trainer_setting.pkl'))
        self._restore_trainer_setting(trainer_setting_dict = trainer_setting)

        # load metrics tracker; if no metrics tracker file found, ignore
        try:
            self._metrics_tracker = self._load_metrics_tracker(os.path.join(dir_path, 'metrics_tracker.pkl')) # load metrics tracker
            self._epoch_metrics = self.metrics_tracker[-1] # get latest epoch metrics

            # retrieve best performance
            metrics_df = self.metrics_tracker.to_df() # convert to dataframe for easier processing
            eval_set = 'test' if metrics_df.get('test') is not None else 'train' # eval metrics = 'test' if test metrics found else use 'train'
            self._best_target_metric = metrics_df.get(eval_set)[self.target_metric].agg(self.target_optimization_mode)
            self._best_epoch = metrics_df.get(eval_set)[self.target_metric].agg(f'idx{self.target_optimization_mode}')[0]

        except OSError: # capture file not found error
            pass

        # put the tensors to the specified device
        self.to(device)

    @classmethod
    def load(cls, dir_path: str, device:Union[str, torch.device] ='cpu'):
        '''class method for loading saved trainer'''
        # create an empty trainer class
        trainer = cls.__new__(cls)

        # restore settings
        trainer._load(dir_path = dir_path, device = device)

        return trainer

    def load_epoch(self, epoch: Optional[Union[int, str]] = None, device:Union[str, torch.device] ='cpu'):
        '''method for loading saved epoch'''
        if not self.save_dir:
            raise RuntimeError('No save files found!!')

        if epoch is None:
            print('No epoch number provided. Loading the best epoch...')
            epoch = 'best'
        else:
            if isinstance(epoch, int):
                epoch = str(epoch)
            elif not epoch in ('best', 'latest'):
                raise ValueError("'epoch' argument has to be one of the following: an int, 'best' or 'latest'.")

        epoch_dir = os.path.join(self.save_dir, f'epoch={epoch}')

        # restore trainer for the epoch specified
        self._load(dir_path = epoch_dir, device = device)
    ##### End: General saving and Loading methods ######


    ######## Methods for loss and metric calculations #####
    def _calc_batch_loss(self, y: torch.Tensor, X: torch.Tensor | List[torch.Tensor], loss_runtime_arg: dict = {}, train_mode = False) -> Tuple[torch.Tensor, float, float]:
        '''method for loss calculation
            Arg
            ------
            y (torch.Tensor): target (truth)
            X (torcj.Tensor): model input
            loss_runtime_arg (dict): runtime arguments for loss function; required for some loss function
            train_mode (bool): toggle train or validation mode; Train mode will trigger back propagation

            Return
            ------
            score, aggregated loss for the batch (float), sum of point weights (float)
        '''
        # get runtime argument for certain loss functions
        add_runtime_arg = self._get_loss_runtime_arg(loss_runtime_arg = loss_runtime_arg)

        if train_mode:
            ### training mode
            # zero the parameter gradients for each batch as backward propagation accumulates gradients
            self.optimizer.zero_grad()

            # forward operation to get predictions/outputs
            score = self.model(X)

            # loss calculation
            loss = self.loss(score, y, *add_runtime_arg) # tensor with gradient
            loss_value = loss.item() # extract loss value

            # back propagation
            loss.backward()
            self.optimizer.step()  # optimization (updates parameters)

        else:
            ### scoring mode
            with torch.no_grad(): # turn off gradient
                self.model.eval() # enter eval mode

                # forward operation to get predictions/outputs
                score = self.model(X)

                # loss calculation
                loss = self.loss(score, y, *add_runtime_arg) # tensor with gradient
                loss_value = loss.item() # extract loss value

        # track sum of data point weights for proper weighted loss calculation.
        # For classification with class weights only; for equal label weights (most cases), data_weight_sum = batch_size.
        data_weight_sum = self._determin_data_weight_sum(y)

        return score, loss_value, data_weight_sum

    def _calc_epoch_loss(self,
                         dataloader: DataLoader,
                         train_mode: bool = False,
                         loss_runtime_arg: dict = {},
                         lr_scheduler_runtime_arg: dict = {}):
        '''method for calculating loss of the epoch'''
        total_loss = 0
        total_data_weight = 0

        # loop through batches
        for batch_idx, (y, X) in enumerate(dataloader):

            if isinstance(X, (list, tuple)):
                X = tuple(x.to(self.device, non_blocking=True) for x in X)
            else:
                X.to(self.device, non_blocking=True)

            y = y.to(self.device)

            # calculate loss
            _, batch_loss, data_weight_sum = self._calc_batch_loss(y = y,
                                                                   X = X,
                                                                   train_mode = train_mode,
                                                                   loss_runtime_arg = loss_runtime_arg)

            # accumulate loss and weight sum
            total_data_weight += data_weight_sum
            total_loss += batch_loss

            if train_mode:
                if self.lr_scheduler is not None:
                    # update learning rate if using batch-level schedulers
                    self._update_lr_batch(batch_idx = batch_idx, **lr_scheduler_runtime_arg)

        return total_loss / total_data_weight

    def calc_epoch_metrics(self,
                      dataloader: DataLoader,
                      metrics: Dict[str, dict]| List[str],
                      loss_runtime_arg: dict = {}):
        '''method for calculating epoch metrics'''
        loss = self._calc_epoch_loss(dataloader = dataloader,
                                     train_mode = False,
                                     loss_runtime_arg = loss_runtime_arg)

        if metrics:
            metrics = calc_metrics(model = self.model,
                                   dataloader = dataloader,
                                   metrics = metrics,
                                   device = self.device)
        else:
            metrics = {}

        return {self.loss.__class__.__name__: loss} | metrics
    ######## End: Methods for loss and metric calculations #####


    ######## Methods for Model training #########
    def _determin_data_weight_sum(self, y: torch.Tensor) -> float:
        '''method for determining sum of all data point weights for final weighted loss calculation;
           for classification with class weights only'''

        # check if loss function use class weighting
        if isinstance(self.loss, (torch.nn.modules.loss._WeightedLoss, additional_loss._CustomWeightedLoss)) and (self.loss.weight is not None):
            # loss with class weight
            if y.dim() == 1:
                y = y.unsqueeze(1)

            weight_sum = self.loss.weight.broadcast_to((y.size(0),self.loss.weight.size(0))).gather(1, y.long()).sum().item()

        elif self.loss.__class__.__name__ in ('BCEWithLogitsLoss') and (self.loss.pos_weight is not None):
            # loss with positive class weight
            class_weights = torch.FloatTensor([1, self.loss.pos_weight]).to(self.device) # get class weights for both pos and neg
            weight_sum = class_weights.broadcast_to([y.size(0),2]).gather(1, y.long()).sum().item()

        else:
            weight_sum = len(y) # weight_sum = length for unweighted cases

        return weight_sum

    def _get_loss_runtime_arg(self, loss_runtime_arg: dict = {}) -> tuple:
        '''method for processing additional runtime argument for certain loss functions'''
        # configure special loss functions that requires extra arguments
        if self.loss.__class__.__name__ == 'CTCLoss':
            ### CTCLoss requires extra inputs 'Input_lengths' and 'Target_lengths'

            # construct extra arguments
            if ('Input_lengths' not in loss_runtime_arg) or ('Target_lengths' not in loss_runtime_arg):
                raise RuntimeError("CTCLoss requires extra arguments 'Input_lengths' and 'Target_lengths'!")

            add_runtime_arg = (loss_runtime_arg['Input_lengths'].to(self.device), loss_runtime_arg['Target_lengths'].to(self.device)) # arg order important

        elif self.loss.__class__.__name__ == 'GaussianNLLLoss':
            ### GaussianNLLLoss requires an extra input - Var (variance)

            if 'Var' not in loss_runtime_arg:
                raise RuntimeError("GaussianNLLLoss requires an extra argument 'Var'!")

            # construct extra arguments
            add_runtime_arg = (loss_runtime_arg['Var'].requires_grad_().to(self.device),)

        else:
            add_runtime_arg = ()

        return add_runtime_arg

    def _fit_epoch(self,
                   dataloader: DataLoader,
                   loss_runtime_arg: dict = {},
                   lr_scheduler_runtime_arg: dict = {}) -> None:
        '''method for performing model training for a single epoch'''

        #### training epoch
        # make sure train mode is on
        self.model.train()

        lr = self.current_lr # need to record initial lr before the epoch run as some learning rate schedulers update lr every batch

        # calculate train loss for the epoch
        train_loss = self._calc_epoch_loss(dataloader = dataloader,
                                           train_mode = True,
                                           loss_runtime_arg = loss_runtime_arg,
                                           lr_scheduler_runtime_arg = lr_scheduler_runtime_arg)

        # record epoch metrics for training
        self._epoch_metrics = {'epoch': self.epoch} # record epoch
        self._epoch_metrics['initial_lr'] = lr # record initial learning rate
        self._epoch_metrics['train'] = {self.loss.__class__.__name__: train_loss}

    def fit(self,
            dataloader_train: _DataLoader,
            dataloader_test: _DataLoader,
            lr_scheduler: Optional[Union[str, Dict[str, dict], List[Dict[str, dict]]]] = 'existing',
            resume: bool = True,
            n_epoch: int = 50,
            display_progress: str | Dict[str, Dict[str, str]] | None = 'text',
            loss_runtime_arg_train: dict = {},
            loss_runtime_arg_test: dict = {},
            lr_scheduler_runtime_arg: dict = {}) -> None:

        '''method for perform model training

            Args
            ----------
            dataloader_train (DataLoader, DistributedDataLoader): dataloader for train set
            dataloader_test (DataLoader, DistributedDataLoader): dataloader for test set
            lr_scheduler (str or dict or list of dict or None): optimizer name or a dict with name as the key and settings (dict) as the values
                                                                 or a list of dict if doing scheduler chaining
                                                                 e.g.
                                                                     'OneCycleLR'
                                                                 or
                                                                     {'OneCycleLR': {'max_lr': 0.01, 'epochs': 10, 'steps_per_epoch': 10}}
                                                                 or
                                                                     [{'scheduler1': {setting_dict}}, {'scheduler2': {setting_dict}}]
                                                                 or
                                                                     None
            resume (bool): whether to resume model training from previous state
            n_epoch (int): number of epochs to train
            display_progress (str, dict): mode of progress display ('text', 'plot' or 'table'). Can also use a dictionary to specify type of display mode and its configuration
            loss_runtime_arg_train (dict): extra arguments required for loss function at runtime for train; required for some loss functions
            loss_runtime_arg_test (dict): extra arguments required for loss function at runtime for test; required for some loss functions
            lr_scheduler_runtime_arg (dict): extra arguments required for loss function at runtime; required for some loss functions
        '''
        # if using OneCycleLR scheduler, overwrite scheduler's epochs parameter with n_epoch arg
        if lr_scheduler != 'existing':
            # if using OneCycleLR scheduler, overwrite scheduler's epochs parameter with n_epoch arg
            if isinstance(lr_scheduler, dict):
                scheduler_name = list(lr_scheduler)[0]

                if scheduler_name == 'OneCycleLR':
                    lr_scheduler[scheduler_name]['epochs'] = n_epoch

            if lr_scheduler is not None:
                print('Initialize lr scheduler...')
                # set scheduler

            self.lr_scheduler = lr_scheduler
        else:
            if self.lr_scheduler is not None:
                print('Use existing lr_scheduler...')

        # check display type
        if not (isinstance(display_progress, dict) or display_progress in ('text', 'table', 'plot', None)):
            raise ValueError("'display_progress' can only takes in the following: 'text', 'table', 'plot', dictionary of display_type and config (e.g. {'plot': {'y_scale': 'log'}}) or None")

        if resume:
            if self.epoch < 0:
                # resume training using the current model weights; no initialize model
                self.reset_components(components = ['optimizer', 'loss', 'lr_scheduler'])
            self._clean_up_save_dir()
        else:
            self.reset()
            if self.save_dir:
                if self.save_dir.startswith('s3://'):
                    delete_s3_objects(self.save_dir)
                elif self.save_dir.startswith('hdfs'):
                    raise RuntimeError('To be implemented...')
                else:
                    if os.path.isdir(self.save_dir):
                        shutil.rmtree(self.save_dir)

        #### model training
        self.epoch += 1

        training_start_time = datetime.now()

        # start training loop
        for epoch in range(self.epoch, self.epoch + n_epoch):

            epoch_start_time = datetime.now() # record epoch training start time; used for keeping track of epoch training time

            # update epoch info
            self.epoch = epoch

            # shuffle data before training
            dataloader_train.shuffle()

            # get train loss. Caution: this loss is calculated with things such as dropout and batchnorm in effect hence higher loss
            self._fit_epoch(dataloader = dataloader_train,
                            loss_runtime_arg = loss_runtime_arg_train,
                            lr_scheduler_runtime_arg = lr_scheduler_runtime_arg)

            # evaluate test metrics if a test dataloader is provided
            self._record_test_metrics(dataloader_test = dataloader_test,
                                        loss_runtime_arg_test = loss_runtime_arg_test)

            # update best epoch and best epoch metric; if no test data, use best loss
            self._update_best_epoch_metric(dataloader_test = dataloader_test)

            # add (copy of the) current report to metrics tracker
            self.metrics_tracker.add_entry(self._epoch_metrics)

            if display_progress is not None:
                print(f'\nEpoch {self._epoch_metrics["epoch"]} (training time = {str(datetime.now() - epoch_start_time).split(".")[0]}):')
                self._print_epoch_metrics(display_type = display_progress) # print metrics
                print(f'Total time elasped: {datetime.now() - training_start_time}')

            # update learning rate if using epoch-level schedulers
            if self.lr_scheduler is not None:
                self._update_lr_epoch(**lr_scheduler_runtime_arg)

            # save trainer and metrics
            if self.save_dir:
                # log last epoch
                self.save(dir_path = os.path.join(self.save_dir, 'epoch=latest'))

                # log best epoch
                if self._best_epoch == self.epoch:
                    self.save(dir_path = os.path.join(self.save_dir, 'epoch=best'))

                if self.save_every_epoch:
                    # log all epochs in addition to the best (by specified target metric) and the last one.
                    self.save(dir_path = os.path.join(self.save_dir, f'epoch={self.epoch}'))

    def _record_test_metrics(self,
                             dataloader_test: DataLoader,
                             loss_runtime_arg_test: dict):
        '''method for recording test metrics'''
        if dataloader_test is not None:
            test_metrics = self.calc_epoch_metrics(dataloader = dataloader_test,
                                                metrics = self.tracked_metrics[1:], # remove loss from tracked metrics
                                                loss_runtime_arg = loss_runtime_arg_test)

            # Track test metrics
            assert isinstance(test_metrics, dict)  # assert we have gotten a dictionary

            for k, v in test_metrics.items():
                if 'test' in self._epoch_metrics:
                    self._epoch_metrics['test'][k] = v
                else:
                    self._epoch_metrics['test'] = {k: v}

    def _update_best_epoch_metric(self, dataloader_test):
        '''method for updating best_target_metric and best_epoch attribute'''
        if dataloader_test is not None:
            epoch_target_metric = self._epoch_metrics['test'][self.target_metric]
        else:
            # if not test data, use train loss as target metric
            epoch_target_metric = self._epoch_metrics['train'][self.target_metric]

        # update best target metric and best epoc
        if self._best_target_metric is None:
            self._best_target_metric = epoch_target_metric
            self._best_epoch = self.epoch
        else:
            if ((self.target_optimization_mode == 'min' and epoch_target_metric < self._best_target_metric)
                or
                (self.target_optimization_mode == 'max' and epoch_target_metric > self._best_target_metric)):
                self._best_target_metric = epoch_target_metric
                self._best_epoch = self.epoch

    def _clean_up_save_dir(self):
        '''method for cleaning up unwanted epochs in the save directory; noramlly used when loading a save epoch and resuming'''
        if self.save_dir:
            #### clean up previous saves
            if self.save_dir.startswith('s3://'):
                # check existing epochs saved in s3
                existing_epochs = [int(re.search('[0-9]+', e).group(0)) for e in list_s3_dir(s3_path = self.save_dir, return_full_path = False)
                                      if e.startswith('epoch=') and not ('best' in e or 'latest' in e)]

                # remove all later saved epochs
                existing_later_epochs_dir = [os.path.join(self.save_dir, f'epoch={e}') for e in existing_epochs if e > self.epoch]
                for p in existing_later_epochs_dir:
                    delete_s3_objects(p)

            elif self.save_dir.startswith('hdfs://'):
                # hadoop file system
                raise RuntimeError('To be implemented...')
            else:
                # local file system
                if os.path.isdir(self.save_dir):
                    # check existing epochs
                    existing_epochs = [int(re.search('[0-9]+', e).group(0)) for e in os.listdir(self.save_dir)
                                       if e.startswith('epoch=') and not ('best' in e or 'latest' in e)]
                    # remove all later saved epochs
                    existing_later_epochs_dir = [os.path.join(self.save_dir, f'epoch={e}') for e in existing_epochs if e > self.epoch]
                    for d in existing_later_epochs_dir:
                        shutil.rmtree(d)

            #### overwrite the last epoch with the loaded epoch before training
            self.save(dir_path = os.path.join(self.save_dir, 'epoch=latest'))

            #### Save the new best epoch
            epoch_path = os.path.join(self.save_dir, f'epoch={self.best_epoch}') # get best epoch path

            # check if the path for best epoch exists
            epoch_saved = self._check_file(path = epoch_path)

            if epoch_saved:
                # save the best epoch if epoch saved
                best_epoch_trainer = self.load(epoch_path)
                best_epoch_trainer.save(dir_path=os.path.join(self.save_dir, 'epoch=best'))
            else:
                # save the current as the new best epoch if no such epoch saved
                self.save(dir_path=os.path.join(self.save_dir, 'epoch=best'))

    def _print_epoch_metrics(self, display_type: str | Dict[str, dict] = 'text'):
        '''method for displaying epoch metrics'''
        if isinstance(display_type, dict):
            display_type, display_config = tuple(display_type.items())[0]
        else:
            display_config = {}

        # check if in ipython environment
        if (get_ipython() is None) and (display_type != 'text'):
            display_type = 'text'
            print("'plot' and 'table' display options only work in iPython interactive environment (e.g Jupyter Notebook)")

        # display progress based on type selected
        loss_type = self.loss.__class__.__name__
        if display_type == 'text':
            print(f'\tLearning Rate: {self._epoch_metrics["initial_lr"]}')

            print(f"\tTrain Loss ({loss_type}): {self._epoch_metrics['train'][loss_type]}")

            test_metrics = self._epoch_metrics.get('test')
            if test_metrics is not None:
                print(f'\tTest Metrics: {test_metrics}')

        elif display_type == 'table':
            clear_output(wait=True)
            display(self.metrics_tracker.to_df())

        elif display_type == 'plot':
            plt.close() # close to free up memory
            clear_output(wait=True) # clear previous plot

            # determine whether to annotate min or max
            annotate_min_metrics = [self.loss.__class__.__name__] # for loss function, min is
            annotate_max_metrics = []

            if self.target_metric != self.loss.__class__.__name__:
                if self.target_optimization_mode == 'min':
                    annotate_min_metrics += [self.target_metric]
                elif self.target_optimization_mode == 'max':
                    annotate_max_metrics += [self.target_metric]

            self.metrics_tracker.plot(metrics = self.tracked_metrics,
                                      annotate_min_metrics = annotate_min_metrics,
                                      annotate_max_metrics = annotate_max_metrics,
                                      **display_config)
            plt.draw() # refresh plot
            plt.pause(0.01)  # need this to refresh the plot correctly

    def _update_lr_batch(self, batch_idx: int, **kwargs):
        '''method for updating learning rate at batch level using scheduler'''

        if self.lr_scheduler.__class__.__name__ in ['CyclicLR', 'OneCycleLR']:
            self.lr_scheduler.step()

        elif self.lr_scheduler.__class__.__name__ in ['CosineAnnealingWarmRestarts']:
            # CosineAnnealingWarmRestarts scheduler requires batch_idx and n_batches_per_epoch arguments
            n_batches_per_epoch = kwargs.get('n_batches_per_epoch')

            if not n_batches_per_epoch:
                raise RuntimeError("Cannot use 'CosineAnnealingWarmRestarts' scheduler unless the number of batches in a epoch ('n_batches_per_epoch') is specified.")

            self.lr_scheduler.step(self.epoch + batch_idx / n_batches_per_epoch)

    def _update_lr_epoch(self, **kwargs):
        '''method for updating learning rate at epoch level using scheduler'''

        if self.lr_scheduler.__class__.__name__ in ['LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR',
                                                     'ConstantLR', 'LinearLR', 'ExponentialLR', 'CosineAnnealingLR',
                                                     'ChainedScheduler', 'SequentialLR']:

            self.lr_scheduler.step()

        elif self.lr_scheduler.__class__.__name__ in ['ReduceLROnPlateau']:
            # ReduceLROnPlateau scheduler requires metric evaluted on validation set

            metric = self._epoch_metrics['test'].get(self.target_metric) # test metrics

            self.lr_scheduler.step(metric)

    ########End: Methods for Model training #########

    ######## Methods for scoring and prediction #########
    def score(self, X: torch.Tensor | List[torch.Tensor]):
        '''method for getting raw model predictions'''
        with torch.no_grad():
            # set model mode to evaluation
            self.model.eval()

            # forward operation to get predictions/outputs

            if isinstance(X, (list, tuple)):
                X = tuple(x.to(self.device) for x in X)

                raw_score = self.model(X)

            else:
                raw_score = self.model(X.to(self.device))

        return raw_score

    def score_dl(self, dataloader: DataLoader):
        '''method for getting raw model predictions'''
        with torch.no_grad():
            # set model mode to evaluation
            self.model.eval()

            # forward operation to get predictions/outputs
            batch_score_list = []
            for y, X in dataloader:
                if isinstance(X, (list, tuple)):
                    X = tuple(x.to(self.device) for x in X)

                    score = self.model(X)
                else:
                    score = self.model(X.to(self.device))

                batch_score_list.append(score)

        return torch.cat(batch_score_list, dim = 0)
