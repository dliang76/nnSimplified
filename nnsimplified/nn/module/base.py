import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import copy
import torchview
from ..helper import save_torch_object, load_torch_object, _weight_init

def _get_output_shape(outputs):
    ### recurive function for getting model output dimension
    if outputs is None:
        return None

    elif isinstance(outputs, torch.Tensor):
      return torch.Size([-1]) + outputs.shape[1:]

    else:
      output_shape = []
      for x in outputs:
          if isinstance(x, torch.Tensor):
              shape = torch.Size([-1]) + x.shape[1:]
              output_shape.append(shape)
          elif isinstance(x, (list, tuple)):
              output_shape.extend(_get_output_shape(x))

      return tuple(output_shape)

class baseModule(torch.nn.Module, ABC):
    '''Base class (template) for simple module.Implemented default save, load and copy methods.
       Forces child classes to have input_shape, output_shape, input_dtype, output_dtype properties and reset() method
    '''

    def __init__(self):
        ''' help set up weight initialization method and activation function'''
        super().__init__()

    @property
    @abstractmethod
    def input_shape(self):
        '''Abstract property for getting input dimension; must be implemented.'''
        pass

    @property
    @abstractmethod
    def output_shape(self):
        '''Abstract property for getting output dimension; must be implemented.'''
        pass

    @property
    @abstractmethod
    def input_dtype(self):
        '''Abstract property for module input dtype; must be implemented.'''
        pass

    @property
    @abstractmethod
    def output_dtype(self):
        '''Abstract property for module output dtype; must be implemented.'''
        pass

    @abstractmethod
    def reset(self):
        '''Abstract method for resetting a module. e.g. re-initialize weight and erase module memory (e.g. batchnorm); must be implemented.'''
        pass

    def save(self, path: str):
        '''default method for saving model'''
        save_torch_object(torch_obj=self, path=path)

    def load(self, path: str):
        '''default method for loading model'''
        # get model
        module = load_torch_object(path)

        # copy settings
        self.__dict__ = module.__dict__

        # set to evaluation mode for safety
        self.eval()

    def copy(self, device: str = 'cpu'):
        '''default method for copying model'''
        new_model = copy.deepcopy(self)
        new_model.to(device)
        new_model.eval()

        return new_model


class nnModule(baseModule, ABC):
    '''Base class for nn module; take care of setting up weight initialization method

       Authors(s): dliang1122@gmail.com

       Args
       ----------
       weight_init (str): weight initialization methods.
                          Available options: ['uniform', 'normal', 'trunc_normal', 'constant',
                                              'ones', 'zeros', 'eye', 'dirac', 'xavier_uniform',
                                              'xavier_normal', 'kaiming_uniform', 'kaiming_normal',
                                              'orthogonal', 'sparse']
       activation (str): activation function.
                         Available options: ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid',
                                             'Tanh', 'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
                                             'Hardshrink', 'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention',
                                             'PReLU', 'Softsign', 'Tanhshrink', 'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']

       Returns
       -------
       None
    '''
    # class variables for storing torch dtypes
    torch_float_type = (torch.bfloat16, torch.float16, torch.float32, torch.float64)
    torch_int_type = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)

    def __init__(self, weight_init: str = 'kaiming_normal'):
        ''' help set up weight initialization method and activation function'''
        super().__init__()
        self.weight_init = weight_init

    @property
    def weight_init(self):
        '''getter for weight_init method name'''
        return self._weight_init

    @property
    def n_parameters(self):
        '''property for total number of parameters in the model'''
        return sum(param.numel() for param in self.parameters())

    @weight_init.setter
    def weight_init(self, weight_init: str | Dict[str, dict]):
        self._weight_init = _weight_init(weight_init = weight_init)

    @abstractmethod
    def forward(self):
        '''Abstract for foward method; must be implemented.'''
        pass

    @abstractmethod
    def reset(self):
        '''must be implemented.'''
        pass

    @property
    @abstractmethod
    def input_shape(self):
        '''must be implemented.'''
        pass

    @property
    @abstractmethod
    def input_dtype(self):
        '''Abstract method for model input dtype; must be implemented.'''
        pass

    @property
    def output_shape(self):
        if not hasattr(self, '_output_shape'):
            # property for getting model output shape
            dummy_output = self._create_dummy_output() # generate dummy output

            if dummy_output is None:
                self._output_shape = None

            elif isinstance(dummy_output, torch.Tensor):
                self._output_shape =  torch.Size([-1]) + dummy_output.shape[1:]

            elif isinstance(dummy_output, (list, tuple)):
                self._output_shape = tuple(torch.Size([-1]) + i.shape[1:] for i in dummy_output)
        
        return self._output_shape

    @property
    def output_dtype(self):
        if not hasattr(self, '_output_dtype'):
            # property for getting model output dtype
            dummy_output = self._create_dummy_output() # generate dummy output

            if dummy_output is None:
                self._output_dtype = None

            elif isinstance(dummy_output, torch.Tensor):
                self._output_dtype = dummy_output.dtype

            elif isinstance(dummy_output, (list, tuple)):
                self._output_dtype = tuple(t.dtype for t in dummy_output)

        return self._output_dtype

    @property
    def device(self):
        '''getter for which device the model parameters are on (e.g. cpu, cuda)'''
        parameters = list(self.parameters())

        return parameters[0].device if parameters else 'cpu'

    def _create_dummy_input(self, batch_size = 1):
        # method for generate a dummy input tensor from input_shape and input_dtype

        input_shape = self.input_shape
        input_dtype = self.input_dtype

        if input_shape is None:
            return None

        elif isinstance(input_shape, torch.Size):
            # single-tensor input
            input_shape = [input_shape] # force list for so we can use the same code to treat both single or multiple inputs


            if input_dtype is None:
                # use float32 if the input_dtype is None (e.g cannot be inferred)
                input_dtype = [torch.float32]
            else:
                input_dtype = [input_dtype] # force list for so we can use the same code to treat both single or multiple inputs

        elif isinstance(input_shape, tuple):
            # multi-tensor input
            if None in input_shape:
                # if any of the input shape cannot be inferred
                return None
            else:
                if input_dtype is None:
                    input_dtype = [torch.float32 for i in len(self.input_shape)]
                else:
                    # use float32 if any dtype is None
                    input_dtype = [torch.float32 if i is None else i for i in input_dtype]

        ### generate dummy input from input_shape and input_dtype
        # initialize input list
        dummy_input = []

        for idx, d in enumerate(input_shape):
            # replace first dim (# of data batches) with the batch size specified
            shape = torch.Size([batch_size]) + d[1:]

            # obtain dtype for the input
            dtype = input_dtype[idx]

            if dtype in self.torch_float_type:
                # generate random float tensor if dtype is of float type
                input = torch.rand(shape,
                                   dtype = dtype,
                                   device = self.device)

            elif dtype in self.torch_int_type:
                # generate random int tensor if dtype is of int type
                input = torch.randint(low = 0,
                                      high=1,
                                      size = shape,
                                      dtype = dtype,
                                      device = self.device)

            dummy_input.append(input)

        if len(dummy_input) == 1:
            # return a tensor if input only contains 1 tensor
            return dummy_input[0]
        else:
            # return a tuple if input only contains multiple tensors
            return tuple(dummy_input)

    def _create_dummy_output(self, batch_size = 1):
        # method for generate a dummy output tensor； useful for checking the data type and shape of the output tensors

        dummy_input = self._create_dummy_input(batch_size = batch_size) # create mock input data

        if dummy_input is not None:
            training_mode = self.training # record current training mode (train or eval) so we can restore it later

            # diable gradient calculation and set the model to eval mode so we don't accidentally modify the model parameters
            with torch.no_grad():
                self.eval()
                output = self(dummy_input) # run the mock input data through the model

            if training_mode: # restore previous training mode
                self.train()

            return output
        else:
            return None

    def freeze_weights(self):
        '''method for freezing parameter weights so they don't get updated during training; useful when one wants to fix weights of certain sub-module of a larger model '''
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_weights(self):
        '''method for unfreezing parameter weights so they can get updated during training'''
        for p in self.parameters():
            p.requires_grad = True

    def visualize(self,
                  input_data = None,
                  graph_dir = 'TB',
                  roll = True,
                  expand_nested = True,
                  **kwargs):

        if input_data is None:
            # if no input_data provided, genereate a dummy input data.
            input_data = self._create_dummy_input()

            if input_data is None:
                raise RuntimeError('Unable to generate mock input data for visualizing the network structure. Please provide <input_data>.')

        # This fixes the error arising when <input_data> is a tuple or a list of tensors in torchview.draw_graph()
        if isinstance(input_data, (list, tuple)):
            input_data = (input_data,)

        return torchview.draw_graph(self,
                                    input_data = input_data,
                                    graph_dir = graph_dir,
                                    roll = roll,
                                    expand_nested = expand_nested,
                                    **kwargs).visual_graph
