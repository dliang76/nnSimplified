from typing import Union
from itertools import compress
import warnings
from abc import ABCMeta, abstractmethod
from ..utils import save_pickle, load_pickle


class _encoder(metaclass=ABCMeta):
    """Base class for converting categorical labels to int.
    Author(s): dliang1122@gmail.com

    """

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass

    def fit_transform(self, y):
        """wrapper method for performing fit and then transform

        args
        ----------
        y (list of str): values to be mapped

        return
        ----------
        list of int: encoded values
        """

        self.fit(y)

        return self.transform(y)

    def save(self, path: str, **kwargs):
        """save method for writing label encoder object to disk

        args
        ----------
        path: save path
        """
        save_pickle(obj=self, path=path, **kwargs)

    def load(self, path: str, **kwargs):
        """load saved object from storage

        args
        ----------
        path: path to save the encoder object
        """
        self.__dict__ = load_pickle(path=path, **kwargs).__dict__


class labelEncoder(_encoder):
    """Class for converting string labels to int. This is more flexible than other label encoding implementations (e.g sklearn's).
    This has the ability to
        1) specify label ordering through 'classes' argument
        3) convert unseen values to a specified value

    Author(s): dliang1122@gmail.com

    init args
    ----------
    classes (list of str): list of unique label values in the desired order.
                                    E.g [x, y, z] -> [0, 1, 2]
    unseen_value_label (str or int): a value for mapping unseen labels to

    """

    def __init__(
        self,
        classes: list[Union[str, int]] = None,
        unseen_value_label: Union[str, int] = "__unseen__",
    ):
        self.classes = classes
        self._unseen_value_label = unseen_value_label

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, classes):
        if classes:
            # input check
            if len(classes) > len(set(classes)):
                raise ValueError("classes cannot have duplicated values!")

            # generate encoder and decoder mapping dictionary
            self._encoder_dict = {k: v for v, k in enumerate(classes)}
            self._decoder_dict = {k: v for k, v in enumerate(classes)}
        else:
            self._encoder_dict = None
            self._decoder_dict = None

        self._classes = classes

    @property
    def unseen_value_label(self):
        return self._unseen_value_label

    def fit(self, y: list[str] = None, verbose: bool = False):
        """fit method for mapping values to integer indices

        args
        ----------
        y (list of str): values to be mapped. Can be None if ordered_labels is provided

        return
        ----------
        None
        """
        if not self.classes:
            # only run if 'classes' has not been specified.

            classes = sorted(set(y))  # extract all unique labels from data

            # generate encoder and decoder mapping dictionary
            self._encoder_dict = {k: v for v, k in enumerate(classes)}
            self._decoder_dict = {k: v for k, v in enumerate(classes)}

            # store the value for unseen data
            if verbose:
                print(f"Mapping all future unseen values to {len(classes)}")
        else:
            print("Class values have been pre-specified. No need to run fit() method.")

    def transform(self, y: list[Union[str, int]]) -> list[int]:
        """transform method for encoding values to integer indices (starting at 0)
        args
        ----------
        y (list of str): values to be converted

        return
        ----------
        list of int: integer indices
        """
        if isinstance(y, (str, int)):
            y = [y]

        if not self._encoder_dict:
            raise RuntimeError("No encoder found. Please run fit() method first")

        # multi-class ecoding; set all unseen labels to the last integer value (length of the encoder dict size)
        result = [self._encoder_dict.get(v, len(self._encoder_dict)) for v in y]

        return result[0] if len(result) == 1 else result

    def inverse_transform(self, y: list[Union[str, int]]) -> list[str]:
        """method for converting encoded values back to their original values
        args
        ----------
        y (list of int): encoded values/indices

        return
        ----------
        list of str: original values
        """
        if isinstance(y, int):
            y = [y]

        if not self._encoder_dict:
            raise RuntimeError("No encoded values found. Please run fit() method first")

        result = [self._decoder_dict.get(v, self._unseen_value_label) for v in y]

        return result[0] if len(result) == 1 else result


class binaryEncoder(_encoder):
    """Class for converting string labels to binary integers. This is more flexible than other label encoding implementations (e.g sklearn's).

    Author(s): dliang1122@gmail.com

    init args
    ----------
    binary_values (list of 2 integers): determine how we map postive and negative classes. Default: [0,1]
                                         E.g For binary_values = [0,1], map positive class to 1 and negative class to 0.
                                             For binary_values = [-1,1], map positive class to 1 and negative class to -1.

    pos_label (str): value for positive class. E.g if pos_label = 'positive', it will map the value 'positive' to int 1 (if 1 is the designated positive class) and the rest to 0 (if 0 is the designated negative class)

    """

    def __init__(self, binary_values: list[int] = [0, 1], pos_label: str | int = None):
        self.binary_values = binary_values
        self.pos_label = pos_label

    @property
    def binary_values(self):
        return self._binary_values

    @binary_values.setter
    def binary_values(self, binary_values):
        if binary_values:
            # input check
            if not isinstance(binary_values, (list, tuple)) and len(binary_values) != 2:
                raise ValueError(
                    "The 'binary_values' arguement has to be a list of 2 integers (e.g. [0,1], [-1,1]) etc)"
                )

            # check elements
            for c in binary_values:
                if not isinstance(c, int):
                    raise ValueError(
                        "The 'binary_values' arguement has to be a list of 2 integers (e.g. [0,1], [-1,1]) etc)"
                    )

            self._binary_values = binary_values
        else:
            self._binary_values = [0, 1]
            print(
                "Use default binary_values. Map label values to 0 (negative class) and 1 (positive class)."
            )

    @property
    def pos_label(self):
        return self._pos_label

    @pos_label.setter
    def pos_label(self, pos_label):
        if pos_label:
            self._encoder_dict = {pos_label: self.binary_values[1]}
            self._decoder_dict = {self.binary_values[1]: pos_label}
        else:
            self._encoder_dict = None
            self._decoder_dict = None

        self._pos_label = pos_label

    def fit(self, y: list[str] = None, verbose: bool = False):
        """fit method for mapping values to integer indices

        args
        ----------
        y (list of str): values to be mapped. Can be None if ordered_labels is provided

        return
        ----------
        None
        """
        if not self.pos_label:
            # only run if 'pos_label' has not been specified.
            classes = sorted(set(y))  # extract all unique labels from data

            if len(classes) > 2:
                warnings.warn(
                    f"Warning! More than 2 unique values found. No pos_label is specified. Use '{classes[-1]}' as the positive class."
                )

            pos_class = classes[-1]

            # generate encoder and decoder mapping dictionary
            self._encoder_dict = {pos_class: self.binary_values[1]}
            self._decoder_dict = {self.binary_values[1]: pos_class}
        else:
            print("pos_label has been pre-specified. No need to run fit() method.")

    def transform(self, y: list[Union[str, int]]) -> list[int]:
        """transform method for encoding values to integer indices (starting at 0)
        args
        ----------
        y (list of str): values to be converted

        return
        ----------
        list of int: integer indices
        """
        if isinstance(y, (str, int)):
            y = [y]

        if not self._encoder_dict:
            raise RuntimeError("No encoder found. Please run fit() method first")

        # binarys ecoding
        result = [self._encoder_dict.get(v, self.binary_values[0]) for v in y]

        return result[0] if len(result) == 1 else result

    def inverse_transform(self, y: list[Union[str, int]]) -> list[str]:
        """method for converting encoded values back to their original values
        args
        ----------
        y (list of int): encoded values/indices

        return
        ----------
        list of str: original values
        """
        if isinstance(y, int):
            y = [y]

        if not self._encoder_dict:
            raise RuntimeError("No encoded values found. Please run fit() method first")

        pos_class = list(self._encoder_dict.keys())[0]
        result = [self._decoder_dict.get(v, f"__not_{pos_class}") for v in y]

        return result[0] if len(result) == 1 else result


class multiLabelEncoder(_encoder):
    """Class for converting multi-labels to int. This is more flexible than sklearn's multi-label encoding implementations.
    This has the ability to
        1) use fit_binary() to convert multiclass labels to binary by specifying a target class (positive class)
        2) specify label ordering
        3) convert unseen values to a specified values

    Author(s): dliang1122@gmail.com

    """

    def __init__(self):
        self.classes_ = None  # for storing ordered labels
        self._unrecognized_value_mapping = (
            None  # map unseen labels to the specified values
        )
        self._encoder_dict = None  # encoding map
        self._decoder_dict = None  # decoding map

    def fit(
        self,
        y: list[list[Union[str, int]]],
        ordered_labels: list[str] = None,
        unrecognized_value_mapping: Union[str, int] = None,
    ):
        """fit method for mapping values to binary value (0, 1)

        args
        ----------
        y (list of list): values to be mapped
        ordered_labels (list of str): list of unique label values in the desired order.
                                      E.g [x, y, z] -> [0, 1, 2]
                                      Will throw away any label not specified here.
        """
        self.__init__()  # clear attributes

        if isinstance(y[0], (list, set)):
            label_set = set([v for l in y for v in l])
        else:
            label_set = set(y)  # get all unique labels

        if ordered_labels:
            self.classes_ = ordered_labels
        else:
            # extract all unique labels from data if not provided
            self.classes_ = sorted(label_set)

        if unrecognized_value_mapping:
            self._unrecognized_value_mapping = unrecognized_value_mapping
            self.classes_.append(unrecognized_value_mapping)

        # generate encoder and decoder mapping dictionary
        self._encoder_dict = {k: v for v, k in enumerate(self.classes_)}
        self._decoder_dict = {k: v for k, v in enumerate(self.classes_)}

    def transform(self, y: list[list[Union[str, int]]]) -> list[int]:
        """transform method for encoding values to integer indices (starting at 0)

        args
        ----------
        y (list of str): values to be converted

        return
        ----------
        list of int: integer indices
        """
        if self.classes_ is None:
            raise RuntimeError("No encoder found. Please run fit() method first")

        if not isinstance(y[0], (list, set)):
            y = [y]

        try:
            result = [
                sorted(
                    set(
                        self._encoder_dict.get(
                            v, self.classes_.index(self._unrecognized_value_mapping)
                        )
                        for v in l
                    )
                )
                for l in y
            ]
        except ValueError:
            raise ValueError(
                'Unrecognized label! Please specify "unrecognized_value_mapping" argument to map unseen label to a specified value.'
            )

        return result[0] if len(result) == 1 else result

    def inverse_transform(self, y: list[int]) -> list[str]:
        """method for converting encoded values back to their original values

        args
        ----------
        y (list of int): encoded values/indices

        return
        ----------
        list of str: original values
        """
        if isinstance(y[0], int):
            y = [y]

        if self.classes_ is None:
            raise RuntimeError("No encoder found. Please run fit() method first")

        result = [
            [self._decoder_dict.get(v, self._unrecognized_value_mapping) for v in l]
            for l in y
        ]

        return result[0] if len(result) == 1 else result

    def fit_transform(
        self,
        y: list[str],
        ordered_labels: list[str] = None,
        unrecognized_value_mapping: Union[str, int] = None,
    ):
        """fit method for mapping values to integer indices

        args
        ----------
        y (list of str): values to be mapped
        ordered_labels (list of str): list of unique label values in the desired order.
                                      E.g [x, y, z] -> [0, 1, 2]
        unrecognized_value_mapping (str or int): a value for mapping unseen labels to

        return
        ----------
        list of int: encoded values

        """
        self.fit(
            y,
            ordered_labels=ordered_labels,
            unrecognized_value_mapping=unrecognized_value_mapping,
        )

        return self.transform(y)

    def transform_binary(self, y: list[list[Union[str, int]]]) -> list[int]:
        """transform method for encoding multi-labels to binary (0, 1 for each label)

        args
        ----------
        y (list of str): values to be converted

        return
        ----------
        list of int: integer indices
        """

        # convert labels to integers first
        encoded_labels = self.transform(y)
        n_labels = len(self.classes_)

        # binarize
        if not isinstance(encoded_labels[0], list):
            encoded_labels = [encoded_labels]

        # initialize resulting list of list
        result = [[0 for i in range(n_labels)] for j in range(len(encoded_labels))]

        for idx, l in enumerate(encoded_labels):
            for i in l:
                if i < n_labels:
                    result[idx][i] = 1

        return result[0] if len(result) == 1 else result

    def inverse_transform_binary(self, y: list[int]) -> list[str]:
        """method for converting encoded binary values back to their original labels

        args
        ----------
        y (list of int): encoded values/indices

        return
        ----------
        list of str: original values
        """
        if isinstance(y[0], int):
            y = [y]

        if self.classes_ is None:
            raise RuntimeError("No encoder found. Please run fit() method first")

        result = [list(compress(self.classes_, l)) for l in y]

        return result[0] if len(result) == 1 else result

    def fit_transform_binary(
        self,
        y: list[str],
        ordered_labels: list[str] = None,
        unrecognized_value_mapping: Union[str, int] = None,
    ):
        """fit method for mapping values to integer indices

        args
        ----------
        y (list of str): values to be mapped
        ordered_labels (list of str): list of unique label values in the desired order.
                                      E.g [x, y, z] -> [0, 1, 2]
        unrecognized_value_mapping (str or int): a value for mapping unseen labels to

        return
        ----------
        list of int: encoded values
        """
        self.fit(
            y,
            ordered_labels=ordered_labels,
            unrecognized_value_mapping=unrecognized_value_mapping,
        )

        return self.transform_binary(y)
