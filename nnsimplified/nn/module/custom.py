import torch
from typing import Optional
from .base import baseModule

class embeddingModule(torch.nn.Embedding, baseModule):
    '''Simple module for generating embedding layer
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        n_categories (int): number of categories
        embedding_dim (int): the size of each embedding vector
        input_data_shape (torch.Size): shape of input data
        padding_idx (int): If specified, the entries at padding_idx do not contribute to the gradient; therefore, the embedding vector at padding_idx is not updated during training. Default to all zero. Commonly used for missing categories.
        kwargs: additional keyword arguments that can be passed to torch.nn.Embedding
    '''
    def __init__(self,
                 n_categories: int,
                 embedding_dim: int,
                 input_data_shape: torch.Size = torch.Size([-1]),
                 padding_idx: int | None = None,
                 **kwargs):

        # store initial arguments
        self._kwargs = kwargs | {'num_embeddings': n_categories, 'embedding_dim': embedding_dim, 'padding_idx': padding_idx}

        super().__init__(**self._kwargs)

        self.input_data_shape = input_data_shape

    @property
    def input_shape(self):
        # Return None due to various input shape. embeddingModule's inputs can be of different dimensions and shapes
        return self.input_data_shape

    @property
    def input_dtype(self):
        return torch.int64 if self.device == 'cpu' else torch.int32

    @property
    def output_shape(self):
        # unable to confirm output shape due to various input shape
        if self.input_shape:
            output_shape = self.input_shape + torch.Size([self.embedding_dim])

        return output_shape

    @property
    def output_dtype(self):
        return torch.float32

    @property
    def device(self):
        return self.weight.device.type

    def reset(self):
        super().__init__(**self._kwargs)


class Passthrough(baseModule):
    '''Simple module for allowing a tensor of specified shape and dtype to passthrough; similar to torch.nn.Identity() but with
       specified shape and dtype attributes useful for figuring out input and ouput of more advanced network structures'''
    def __init__(self,
                 shape: torch.Size | list[torch.Size],
                 dtype: torch.dtype | list[torch.dtype] = torch.float32):

        super().__init__()

        self.shape = shape
        self.dtype = dtype

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape: torch.Size | list[torch.Size]):
        # assignment with data check
        if isinstance(shape, torch.Size):
            self._shape = torch.Size([-1]) + shape[1:] # replace the first dim with -1 to indicate arbitrary batch size

        elif isinstance(shape, (list, tuple)):
            for idx, i in enumerate(shape):
                if not isinstance(i, torch.Size):
                    raise ValueError('<shape> has to be either a list of torch.Size or torch.Size')

            self._shape = tuple(torch.Size([-1]) + i[1:] for i in shape)
        else:
            raise ValueError('<shape> has to be either a list of torch.Size or torch.Size')

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype | list[torch.dtype]):
        # assignment with data check
        if isinstance(self.shape, torch.Size):
            if isinstance(dtype, torch.dtype):
                self._dtype = dtype
            else:
                raise ValueError('<dtype> has to be a torch.dtype')

        elif isinstance(self.shape, (list, tuple)):
            if isinstance(dtype, torch.dtype):
                self._dtype = (dtype,) * len(self.shape)

            elif isinstance(dtype, (list, tuple)):
                if len(dtype) == 1:
                    dtype = dtype * len(self.shape)
                elif len(dtype) != len(self.shape):
                    raise ValueError(f'The number of dtypes ({len(dtype)}) in <dtype> does not match the number of allowed input tensors ({len(self.shape)}) in <shape>')

                self._dtype = tuple(dtype)
            else:
                raise ValueError('<dtype> has to be either a list of torch.dtype or torch.dtype')

    @property
    def input_shape(self):
        return self.shape

    @property
    def output_shape(self):
        return self.shape

    @property
    def input_dtype(self):
        return self.dtype

    @property
    def output_dtype(self):
        return self.dtype

    def reset(self):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}(shape = {self._shape}, dtype = {self._dtype})'

    def _input_check(self, x):

        if isinstance(x, torch.Tensor):
            shape = torch.Size([-1]) + x.shape[1:]
        elif isinstance(x, (list, tuple)):
            shape = tuple(torch.Size([-1]) + t.shape[1:] for t in x)

        if shape != self.shape:
            raise RuntimeError('Incorrect shape for the Passthrough module! The input requires shape {self.shape} but got {shape} instead.')

    def forward(self, x):
        self._input_check(x)

        return x


class Concat(torch.nn.Module):
    '''Simple module for concatenating outputs from previous modules
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        dim(int): dimension to concat
    '''
    def __init__(self,
                 dim: int):
        super().__init__()
        self.dim = dim

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, dim: int):
        if not isinstance(dim, int):
            raise ValueError('<dim> has to either be an integer.')
        self._dim = dim

    def forward(self, x):
        return torch.cat(x, dim = self.dim)

    def __repr__(self):
        txt = str(self.__class__) + '\n'
        txt += f'''{type(self).__name__}(dim = {self.dim})'''

        return txt


class TensorSum(torch.nn.Module):
    '''Simple module for summing the outputs of previous modules
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        weights(float): weights added to the outputs of the previous module(s)
    '''
    def __init__(self,
                 weights: list[float] = None):
        super().__init__()
        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: float):
        if not weights:
            self._weights = None
        else:
            if not isinstance(weights, (list, tuple)):
                raise ValueError('<weights> has to either be None or a list of numbers.')
            self._weights = torch.tensor(weights)

    def forward(self, x):
        x = torch.stack(x)
        if self.weights is not None:
            return torch.sum(self.weights.view(len(x), *([1]*(x.dim() - 1))) * x, dim = 0)
        else:
            return x.sum(dim = 0)

    def __repr__(self):
        txt = str(self.__class__) + '\n'
        txt += f'''{type(self).__name__}(weights = {self.weights})'''

        return txt


class TensorMean(TensorSum):
    '''Simple module for averagikng the outputs of the previous modules
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        weights(float): weights added to the outputs of the previous module(s)
    '''
    def __init__(self,
                 weights: list[float] = None):
        super().__init__(weights = weights)
        if self.weights is not None:
            self._normalized_weights = self.weights/self.weights.sum()

    def forward(self, x):
        x = torch.stack(x)
        if self.weights is not None:
            return torch.sum(self._normalized_weights.view(len(x), *([1]*(x.dim() - 1))) * x, dim = 0)
        else:
            return x.mean(dim = 0)


class Split(torch.nn.Module):
    '''Simple module for splitting the tensors
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        split_size_or_sections (int, list[int]): integer value = size of all resulting tensor; list or integers - specify size of each resulting tensor
            e.g. input tensor = [2,3,1,4]
                    -  split_size_or_sections = 2 -> two resulting tensors [2,3], [1,4]
                    -  split_size_or_sections = [2,1,1] -> 3 resulting tensors [2,3], [1], [4]
       dim(int): along which dimension to split
    '''
    def __init__(self,
                 split_size_or_sections: int | list[int],
                 dim: int):
        super().__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    @property
    def split_size_or_sections(self):
        return self._split_size_or_sections

    @split_size_or_sections.setter
    def split_size_or_sections(self, split_size_or_sections: int | list[int]):
        if isinstance(split_size_or_sections, int):
            self._split_size_or_sections = split_size_or_sections
        elif isinstance(split_size_or_sections, (list, tuple)):
            for i in split_size_or_sections:
                if not isinstance(i, int):
                    raise ValueError('<split_size_or_sections> has to either be an int or a list of integers.')

            self._split_size_or_sections = split_size_or_sections
        else:
            raise ValueError('<split_size_or_sections> has to either be an int or a list of integers.')

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, dim: int):
        if not isinstance(dim, int):
            raise ValueError('<dim> has to either be an integer.')
        self._dim = dim

    def forward(self, x):
        return torch.split(tensor = x, split_size_or_sections = self.split_size_or_sections, dim = self._dim)

    def __repr__(self):
        txt = str(self.__class__) + '\n'
        txt += f'''{type(self).__name__}(split_size_or_sections = {self.split_size_or_sections}, dim = {self.dim})'''

        return txt


class ConcatNSplit(torch.nn.Module):
    '''Simple module for concating and re-splitting the tensors
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        split_size_or_sections (int, list[int]): integer value = size of all resulting tensor; list or integers - specify size of each resulting tensor
            e.g. input tensor = [2,3,1,4]
                    -  split_size_or_sections = 2 -> two resulting tensors [2,3], [1,4]
                    -  split_size_or_sections = [2,1,1] -> 3 resulting tensors [2,3], [1], [4]
        concat_dim (int): dimension to concat
        dim(int): dimension to split after concatenation
    '''
    def __init__(self,
                 split_size_or_sections: int | list[int],
                 concat_dim: int = -1,
                 split_dim: int = -1):
        super().__init__()
        self.concat_dim = concat_dim
        self.split_size_or_sections = split_size_or_sections
        self.split_dim = split_dim

    @property
    def concat_dim(self):
        return self._concat_dim

    @concat_dim.setter
    def concat_dim(self, concat_dim: int):
        if not isinstance(concat_dim, int):
            raise ValueError('<concat_dim> has to either be an integer.')
        self._concat_dim = concat_dim

    @property
    def split_size_or_sections(self):
        return self._split_size_or_sections

    @split_size_or_sections.setter
    def split_size_or_sections(self, split_size_or_sections: int | list[int]):
        if isinstance(split_size_or_sections, int):
            self._split_size_or_sections = split_size_or_sections
        elif isinstance(split_size_or_sections, (list, tuple)):
            for i in split_size_or_sections:
                if not isinstance(i, int):
                    raise ValueError('<split_size_or_sections> has to either be an int or a list of integers.')

            self._split_size_or_sections = split_size_or_sections
        else:
            raise ValueError('<split_size_or_sections> has to either be an int or a list of integers.')

    @property
    def split_dim(self):
        return self._split_dim

    @split_dim.setter
    def split_dim(self, split_dim: int):
        if not isinstance(split_dim, int):
            raise ValueError('<split_dim> has to either be an integer.')
        self._split_dim = split_dim

    def forward(self, x):
        x = torch.cat(tensors = x, dim = self.concat_dim)
        x = torch.split(tensor = x, split_size_or_sections = self.split_size_or_sections, dim = self.split_dim)
        return x

    def __repr__(self):
        txt = str(self.__class__) + '\n'
        txt += f'''{type(self).__name__}(split_size_or_sections = {self.split_size_or_sections}, concat_dim = {self.concat_dim}, split_dim = {self.split_dim})'''

        return txt


class Transpose(torch.nn.Module):
    '''Simple module for transposing tensors
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        dim0(int): the first dimension to be transposed
        dim1(int): the second dimension to be transposed
    '''
    def __init__(self,
                 dim0,
                 dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    @property
    def dim0(self):
        return self._dim0

    @dim0.setter
    def dim0(self, dim0: int):
        if not isinstance(dim0, int):
            raise ValueError('<dim0> has to either be an integer.')
        self._dim0 = dim0

    @property
    def dim1(self):
        return self._dim1

    @dim1.setter
    def dim1(self, dim1: int):
        if not isinstance(dim1, int):
            raise ValueError('<dim1> has to either be an integer.')
        self._dim1 = dim1

    def forward(self, x):
        return torch.transpose(input = x, dim0 = self.dim0, dim1 = self.dim1)

    def __repr__(self):
        txt = str(self.__class__) + '\n'
        txt += f'''{type(self).__name__}(dim0 = {self.dim0}, dim1 = {self.dim1})'''

        return txt


class BatchedDot(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        x, y = input
        return (x * y).sum(dim = -1).unsqueeze(dim = -1)

    def __repr__(self):
        txt = str(self.__class__) + '\n'
        txt += f'''{type(self).__name__}()'''

        return txt


class MatryoshkaLayer(torch.nn.Module):
    '''module for generating matryoshka output (nested)
        Authors(s): denns.liang@hilton.com

        init args
        ----------
        rep_sizes(list[int]): represenation sizes. E.g. for a 128-dim vector, [4, 16, 128] means using first 4, first 16, and the whole 128
        min_size(int): minimum size for auto splitting
    '''
    def __init__(self, rep_sizes: list[int] = None, min_size = 4):
        super().__init__()

        self.rep_sizes = rep_sizes
        self.min_size = min_size # for auto split only

    @property
    def rep_sizes(self):
        return self._rep_sizes

    @rep_sizes.setter
    def rep_sizes(self, rep_sizes: Optional[list]):

        if rep_sizes:
            if isinstance(rep_sizes, (list, tuple)):
                self._rep_sizes = sorted(rep_sizes)

                if self._rep_sizes[0] < 1:
                    raise ValueError("<rep_sizes> provided must be a list of integers > 0!")
            else:
                raise ValueError("<rep_sizes> provided must be either a list or a tuple!")
        else:
            # rep_sizes not provide. Will split automatically based on input tensor
            self._rep_sizes = None

    @property
    def min_size(self):
        return self._min_size

    @min_size.setter
    def min_size(self, min_size: int):
        if self.rep_sizes:
            self._min_size = self.rep_sizes[0]
        else:
            if isinstance(min_size, int) and min_size > 0:
                self._min_size = min_size
            else:
                raise ValueError("<min_size> must be a positive integer!")

    def _auto_sizes(self, input: torch.Tensor):
        size = input.shape[-1]
        rep_sizes = []

        while size >= self.min_size:
            rep_sizes.append(size)
            size = size // 2

        return sorted(rep_sizes)

    def _nested_rep(self, input: torch.Tensor, rep_sizes: list[int]):
        return tuple([input.narrow(dim = -1, start = 0, length = s) for s in rep_sizes])

    def forward(self, input):
        if not self.rep_sizes:
            rep_sizes = self._auto_sizes(input)
        else:
            rep_sizes = self.rep_sizes

        return self._nested_rep(input, rep_sizes)

    def __repr__(self):
        txt = str(self.__class__) + '\n'
        if self.rep_sizes:
            txt += f'''{type(self).__name__}(rep_sizes = {self.rep_sizes})'''
        else:
            txt += f'''{type(self).__name__}(rep_sizes = None, min_size = {self.min_size})'''

        return txt