from typing import Optional
from collections import deque
import copy
import torch
import torch._dynamo
import warnings
import inspect
from .base import nnModule, baseModule
from .custom import Passthrough
from ...utils import flatten_list

def _retrieve_module_name(module):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    """
    for i in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in i.frame.f_locals.items() if var_val is module]
        if len(names) > 0:
            return names[0]


def _rename_duplicates(lst: list[str]):
    postfix_dict = dict() # keep track of postfixes

    for s in lst:

        # add postfix if the item is a duplicate
        if s not in postfix_dict:
            # first/original string
            postfix_dict[s] = deque([f'']) # postfix is an empty string

        elif postfix_dict[s][-1] == '':
            # first duplicates
            postfix_dict[s].append('_1')
        else:
            prev_duplicated_str = postfix_dict[s][-1]
            # add postfixes to duplicates
            if prev_duplicated_str == '':
                postfix = '_1'
            else:
                # add one to the postfix
                postfix = int(postfix_dict[s][-1].replace('_','')) + 1
            postfix_dict[s].append(f'_{postfix}')

    result = []
    for i, e in enumerate(lst):
        postfix_num = postfix_dict[e].popleft()
        result.append(lst[i] + postfix_num)

    return result


def _remove_none_from_nested_list(lst: list | tuple) -> list:
    '''method for removing None from nested list'''
    result = []
    for i in lst:
        if isinstance(i, list):
            result.append(_remove_none_from_nested_list(i))
        elif i is not None:
            result.append(i)

    return result


class nnParallel(nnModule):
    ''' Class for grouping a list of nn modules parallely into one large module

        Authors(s): denns.liang@hilton.com

        init args
        ----------
        module_list (list or dict of nnModule, torch.nn.Module, int or torch.Size): a list of modules or dict of modules (named modules) to be grouped together
        copy_module (bool): whether to make a copy or to reference the modules. Default: False
        use_common_input (bool): whether for all modules to take in the same input. Default: False
        output_combine_method (str): method for cominbing outputs from different modules. Support 'sum' and 'concat' currently.
    '''

    def __init__(self,
                 module_list: list[torch.nn.Module | baseModule | int | torch.Size] | dict[str, torch.nn.Module | baseModule | int | torch.Size] = [],
                 copy_module: bool = False,
                 use_common_input: bool = False,
                 output_combine_method: Optional[str] = None):

        super().__init__()

        # go through the list of modules and register them in PyTorch.
        self.register_modules(module_list = module_list, copy_module = copy_module)

        self.use_common_input = use_common_input

        self.output_combine_method = output_combine_method

    @property
    def module_list(self) -> list:
        return self._module_list

    def register_modules(self,
                         module_list: list[nnModule | baseModule | int] | dict[str, nnModule | baseModule | int],
                         copy_module: bool):
        '''Method for registering modules in the module list'''

        if module_list:
            if copy_module:
                # make a copy of the modules
                module_list = copy.deepcopy(module_list)

            if isinstance(module_list, dict):
                module_list = list(module_list.items())  # Some risk: rely on python dictionary ordering behavior (ordered by insertion)

            elif isinstance(module_list, (list, tuple)):
                # get module names.
                # If module is assigned to a variable, use variable name as module name.
                # Otherwise, use class name as module name
                names = []
                for module in module_list:
                    if isinstance(module, int):
                        module_name = 'Passthrough'
                    else:
                        module_name = _retrieve_module_name(module) # get variable name if the module is assigned to a variable
                        if module_name == inspect.getfullargspec(_retrieve_module_name).args[0]:
                            # module name is not assigned to a variable (get the same name as the argument of _retrieve_module_name())
                            module_name = module.__class__.__name__

                    names.append(module_name)

                names = _rename_duplicates(names) # add numbered postfix to duplicated names

                module_list = list(zip(names,module_list))

            else:
                raise ValueError("The <module_list> argument has to be either a list of modules or dict of named modules")

            # extract all module_list
            self._module_list = []
            unknown_input_shape_module = [] # keep track of the modules that we have no info about their input data shape.

            for name, module in module_list:
                if isinstance(module, torch._dynamo.eval_frame.OptimizedModule):
                    raise ValueError('Modules included cannont be of the type OptimizedModule (compiled). Please compile the resulting model only.')

                if isinstance(module, int):
                    # an integer value indicates it can passthrough a tensor with width = the value specified without any modification;
                    module = Passthrough(shape = torch.Size([-1, module]))

                elif isinstance(module, torch.Size):
                    # a torch.Size input indicates it can passthrough a tensor with shape of the specified torch.Size without any modification

                    # change the first dim (data batch size) to -1 to indicate arbitrary length
                    shape = torch.Size([-1]) + module[1:]
                    module = Passthrough(shape = shape)

                elif isinstance(module, baseModule):
                    if module.input_shape is None:
                        # the module's input shape is unknown
                        unknown_input_shape_module.append(name)

                elif isinstance(module, torch.nn.Module):
                    if isinstance(module, torch.nn.modules.linear.Linear):
                        module.input_shape = torch.Size([-1, module.in_features])
                        module.input_dtype = torch.int32
                    else:

                        # No easy way to obtain input shapes for general torch.nn.Module objects
                        unknown_input_shape_module.append(name)

                else:
                    raise RuntimeError('Every element in <module_list> has to be one of following: a torch.nn.Module, an int or a torch.Size.')

                self.add_module(name = str(name), module = module)
                self._module_list.append((name, module))

            if unknown_input_shape_module:
                warning_text = f'Unable to determine the required input data shape for the following modules: {unknown_input_shape_module}. Since we cannot check the integrity of the structure, please proceed with caution.'
                warnings.warn(warning_text, RuntimeWarning)

    @property
    def structure_input_shape(self) -> tuple[torch.Size]:
        '''network structure input shape. Could be different from user input shape due to the flag use_common_input'''

        if not hasattr(self, '_structure_input_shape'):
            # get input shape for each module
            module_input_shape = flatten_list([module.input_shape if isinstance(module, baseModule) else None for name, module in self._module_list])

            if module_input_shape:
                if (len(module_input_shape) == 1):
                    self._structure_input_shape = module_input_shape[0] # return torch.Size if the list only has 1 element
                else:
                    self._structure_input_shape = tuple(module_input_shape) # return tuple of torch.Size if multiple
            else:
                self._structure_input_shape = None # return None if all module input_shape are None

        return self._structure_input_shape

    @property
    def input_shape(self) -> tuple[torch.Size]:
        '''required input shape. Might be different from network input shape due to the flag use_common_input'''
        shape = self.structure_input_shape # get structure input shape

        if self.use_common_input:
            if shape:
                if type(shape) == tuple:
                    shape = [s for s in shape if s is not None] # remove any None from the list

                    if shape:
                        shape = shape[0] # retrieve the first input shape as the common input shape

        return shape

    @property
    def structure_input_dtype(self) -> tuple[torch.dtype]:
        '''network structure input dtype. Could be different from user input dtype due to the flag use_common_input'''

        if not hasattr(self, '_structure_input_dtype'):
            module_input_dtype = []

            for name, module in self._module_list:
                if isinstance(module, baseModule):
                    # extract info using input_shape and input_dtype attributes (required for baseModule objects) if the module is an instance of baseModule
                    input_shape = module.input_shape
                    input_dtype = module.input_dtype

                    if isinstance(input_shape, torch.Size):
                        if input_dtype is None:
                            input_dtype = torch.float32 # use torch.float32 as default
                        elif isinstance(input_dtype, (list, tuple)):
                            input_dtype = [torch.float32 if dtype is None else dtype for dtype in input_dtype] # use torch.float32 as default

                    module_input_dtype.append(input_dtype)
                else:
                    # not a baseModule object; unknown input dtype
                    module_input_dtype.append(None)

            module_input_dtype = flatten_list(module_input_dtype) # flatten nested list

            if module_input_dtype:
                if (len(module_input_dtype) == 1):
                    self._structure_input_dtype = module_input_dtype[0] # return torch.dtype if the list only has 1 element
                else:
                    self._structure_input_dtype = tuple(module_input_dtype) # return tuple of torch.dtyp if multiple
            else:
                self._structure_input_dtype = None # return None if all module input_dtype are None

        return self._structure_input_dtype

    @property
    def input_dtype(self) -> tuple[torch.dtype]:
        '''required input dtype. Might be different from structure input dtype due to the flag use_common_input'''
        dtype = self.structure_input_dtype # get structure input dtype

        if self.use_common_input:
            if dtype:
                if type(dtype) == tuple:
                    dtype = [s for s in dtype if s is not None] # remove any None from the list

                    if dtype:
                        dtype = dtype[0] # retrieve the first input dtype as the common input dtype

        return dtype


    @property
    def use_common_input(self) -> bool:
        return self._use_common_input

    @use_common_input.setter
    def use_common_input(self, use_common_input: bool):

        if use_common_input:
            input_shapes = self.structure_input_shape
            input_dtypes = self.structure_input_dtype

            if type(input_shapes) != tuple:
                input_shapes = [input_shapes]

            if type(input_dtypes) != tuple:
                input_dtypes = [input_dtypes]

            # check module_list to make sure inputs to all the modules are of the same shape
            contain_unknown_shape = None in input_shapes

            if contain_unknown_shape:
                input_shapes = [s for s in input_shapes if s is not None] # remove None
                input_dtypes = [s for s in input_dtypes if s is not None] # remove None

            if len(set(input_shapes)) > 1:
                raise RuntimeError('The inputs to the sub-modules are of different dimensions! Cannot use a common input.')

            if len(set(input_dtypes)) > 1:
                raise RuntimeError('The inputs to the sub-modules are of different data types! Cannot use a common input.')

            if contain_unknown_shape:
                warnings.warn(f'Unable to determine the required input data shape for some of the modules. Since we cannot check the integrity of the structure, please proceed with caution.', RuntimeWarning)

        self._use_common_input = use_common_input

    @property
    def output_combine_method(self) -> str:
        return self._output_combine_method

    @output_combine_method.setter
    def output_combine_method(self, output_combine_method):
        '''setter with data check'''
        if output_combine_method:
            # check output shape
            output_shapes = flatten_list([m.output_shape if hasattr(m, 'output_shape') else None for name, m in self.module_list])

            if None in output_shapes:
                pass # if one of the output_shapes is unknown; let it pass (no ouput shape check)
            elif len(output_shapes) == 1:
                # only 1 output tensor; no need to combine
                output_combine_method = None
            else:
                if output_combine_method == 'sum':
                    # make sure the outputs of the modules are of same shape
                    if len(set(output_shapes)) > 1:
                        # mismatch in output shape ; cannot be summed
                        raise RuntimeError('Outputs of the modules are of different shapes! Cannot be summed.')
                elif output_combine_method == 'concat':
                    if len(set([d[0:-1] for d in output_shapes])) > 1:
                        # mismatch in -1 dimension; cannot be concated
                        raise RuntimeError('Outputs of the modules are of different length! Cannot be concated.')

        self._output_combine_method = output_combine_method

    def forward(self, inputs):
        '''forward method for defining network structure (how data flows)'''
        outputs = []

        # Use different inputs for different modules
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        if self.use_common_input:
            # if using common input, reference (no copy) the same input multiple times to fit the structure requirement
            inputs = inputs * len(self.structure_input_shape)

        # run elements of input list through the matching modules.
        # e.g. modules = [module1, module2, ...] and inputs = [input1, input2...]
        # input1 -> module1, input2 -> module2 etc.
        pos = 0
        for name, module in self._module_list:
            if isinstance(module, Passthrough):
                # numeric module indicates passthrough; output tensor = input tensor
                module_output = inputs[pos]

                module_input_len = 1 # only use 1 tensor if passthrough

            elif isinstance(module, torch.nn.Module):
                if hasattr(module, 'input_shape') and module.input_shape:
                    module_input_shape = module.input_shape

                    if isinstance(module_input_shape, torch.Size):
                        module_input_len = 1
                        module_output = module(inputs[pos])

                    elif isinstance(module_input_shape, (list, tuple)):
                        module_input_len = len(module_input_shape)
                        module_output = module(inputs[pos:(pos + module_input_len)])

                else:
                    module_input_len = 1 # assume the module only accepts 1 input tensor if no input_shape property found
                    module_output = module(inputs[pos])

            if isinstance(module_output, torch.Tensor):
                outputs.append(module_output)
            else:
                outputs.extend(module_output)

            pos += module_input_len

        if len(outputs) == 1:
            # only keep the element of the output only has 1 element
            outputs = outputs[0]
        else:
            outputs = tuple(outputs)

            if self.output_combine_method == 'sum':
                # sum outputs if specified
                for idx, t in enumerate(outputs):
                    if idx == 0:
                        result = t
                    else:
                        result += t

                outputs = result

            elif self.output_combine_method == 'concat':
                # concat outputs if specified
                outputs = torch.concat(outputs, dim = -1)

        return outputs

    def __getitem__(self, idx):
        '''make the objectg indexible. E.g can use object[idx] to retrieve modules'''
        return [v for v in self._modules.values()][idx]

    def reset(self):
        '''method for resetting parameters'''
        for name, module in self._module_list:
            if isinstance(module, baseModule):
                module.reset()
            elif isinstance(module, torch.nn.Module):
                # for general torch.nn.Module
                if hasattr(module, 'weight'): # some have attribute weight
                    self.weight_init(module.weight)
            else:
                # will add more after identifying attributes that can be initialized
                pass


class nnModular(nnModule):
    ''' Class for creating complex neural structure. Used for connecting modules in a series into a single model.
        The order goes from left to right; one module's output will feed into the next module (to the right in the list).
        As a result, the output tensor shape of one module has to be compatible with the input of the next module.
        Similar to torch.nn.Sequential but can have more complicated input modules.

        The <network_structure> is a list of modules that we want to connect.
        Here are accepted elements in the list.
        1. An integer element means simply passing a tensor of width 2 through without doing anything.
           Good for specifying model input shape.
            e.g 2 means a 2D tensor of shape torch.Size([-1, 2]) (-1 indicates arbitrary batch size and 2 is the width of the 2D tensor)

        2. A torch.Size element means simply passing a tensor of shape specified by the torch.Size through without processing.
            e.g torch.Size([-1,10,3]) means a 3D tensor of shape torch.Size([-1,10,3]) (arbitrary batch size, height of 10 and width of 3)

        3. A torch.nn.Module. Run the the output tensor(s) of the previous module through the forward() method to generate the next output.

        4. A list of (torch.nn.Module or int or torch.Size). The class will group the elements in the list to form 1 single nnParallel object and run the forward() method of the nnParallel object

        Example:
        1. model = nnModular(network_structure([torch.nn.Linear(10,20),
                                                mlp([20, 10, 1])]))
           to connect a linear layer to an MLP
        2. module = nnModular([[3,4,5],  # level 0:requires 3 tensor inputs of width 3, 4 and 5
                               [mlp([3,3,3]), mlp([4,5,6]), mlp([5,3,4])], # level 1: contains 3 MLP modules of input widths 3,4 and 5.
                               Concat(-1), # level 2: concatenate 3 MLP output tensors from the previous level into 1 single tensor
                               mlp([13,3,4]) # level 3: another mlp of input width = 13
                              ]
                              )

        Authors(s): denns.liang@hilton.com

        init args
        ----------
        network_structure (list or dict of nnModule, torch.nn.Module, int or torch.Size): a list of modules or dict of modules (named modules) to be connected 1 after another (from left to right in the list)
        copy_module (bool): whether to make a copy or to reference the modules
    '''
    def __init__(self,
                 network_structure: list[list[baseModule | torch.nn.Module] | baseModule | torch.nn.Module],
                 copy_module: bool = False,
                 **kwargs):

        super(nnModular, self).__init__()

        # register given modules
        self.register_modules(network_structure, copy_module = copy_module)

        # make sure modules can be connected
        self._check_structure_integrity()

        self.additional_args = kwargs

    def register_modules(self,
                         network_structure: list[list[baseModule | torch.nn.Module] | baseModule | torch.nn.Module] | dict[str, list[baseModule | torch.nn.Module] | baseModule | torch.nn.Module],
                         copy_module: bool):
        '''method for register network components'''

        if copy_module:
            network_structure = copy.deepcopy(network_structure)

        # initailize variable for storing modules in each level
        self._level_modules = []

        for lvl, structure in enumerate(network_structure):

            if isinstance(structure, int):
                # store module as a Passthrough module if an integer is passed.
                lvl_module = Passthrough(shape = torch.Size([-1, structure]))

            elif isinstance(structure, torch.Size):
                # # store module as a Passthrough module if a torch.Size is passed.
                lvl_module = Passthrough(shape = torch.Size([-1]) + structure[1:])

            elif isinstance(structure, torch.nn.Module):
                # store module as a general torch.nn.Module
                lvl_module = structure

            elif isinstance(structure, (list, tuple, dict)):
                if len(structure) > 1:
                    lvl_module = nnParallel(structure) # form nnParallel if the level structure contains more than 1 module
                else:
                    # extract the only element
                    if isinstance(structure, dict):
                        lvl_module = list(structure.values())[0]
                    else:
                        lvl_module = structure[0]

                    if isinstance(lvl_module, int):
                        # module is Passthrough if an integer is passed.
                        lvl_module = Passthrough(shape = torch.Size([-1, lvl_module])) # set first dim to -1 to indicate arbitrary batch size
                    elif isinstance(lvl_module, torch.Size):
                        # module is Passthrough if a torch.Size is passed.
                        lvl_module = Passthrough(shape = torch.Size([-1]) + lvl_module[1:]) # change first dim to -1 to indicate arbitrary batch size

            else:
                raise ValueError("Elements of 'network_structure' has to be either a list of modules) or dict of named modules")

            self._level_modules.append(lvl_module)
            self.add_module(f'level{lvl}', lvl_module)

    @property
    def input_shape(self) -> torch.Size:
        ''' get required input shape'''
        input_module = self._level_modules[0]

        if isinstance(input_module, baseModule):
            return input_module.input_shape
        else:
            return None

    @property
    def input_dtype(self) -> torch.dtype:
        ''' get required input dtype'''
        input_module = self._level_modules[0]

        if isinstance(input_module, baseModule):
            return input_module.input_dtype
        else:
            return None

    def _check_structure_integrity(self):
        '''make sure modules from different level can be connected;
           combined output shape of the previous level must matche the input shape of each module in the next level)'''

        # create a mock input data and run through the model to see if there is any error
        x = self._create_dummy_input()

        if x is None:
            # rais a warning if we cannot genereate mock input data automatically
            warnings.warn('Unable to determine structure integrity because required input data shape cannot be determined! Proceed with caution.')
        else:
            # Run input through the module in each level
            for lvl, lvl_module in enumerate(self._level_modules):
                training_mode = lvl_module.training # record current training mode for the module

                try:
                    # use no grad and eval mode to avoid modifying the module
                    with torch.no_grad():
                        lvl_module.eval()
                        x = lvl_module(x)

                except Exception as e:

                    if lvl == 0:
                        raise RuntimeError(f'Error in connecting to Level {lvl} module! ' + str(e))
                    else:
                        raise RuntimeError(f'Error in connecting to Level {lvl - 1} to Level {lvl} module! ' + str(e))

                finally:
                    self.train(training_mode) # re-store previous mode

    def forward(self, input):
        '''forward method for defining network structure (how data flows)'''

        for lvl, m in enumerate(self._level_modules):

            if lvl == 0:
                x = input if isinstance(m, Passthrough) else m(input)
            else:
                if not isinstance(m, Passthrough):
                    x =  m(x)

        return x

    def __getitem__(self, idx):
        return [v for v in self._modules.values()][idx]

    def reset(self):
        for module in self._level_modules:
            if isinstance(module, baseModule):
                module.reset()
            elif isinstance(module, torch.nn.Module):
                # for general torch.nn.Module
                if hasattr(module, 'weight'): # some have attribute weight
                    self.weight_init(module.weight)
            else:
                # will add more after identifying attributes that can be initialized
                pass