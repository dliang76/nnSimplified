import torch
from .base import nnModule
from .assemble import nnModular, nnParallel
from .custom import Concat


class skipConnection1D(nnModule):
    """module for create skip connection (1D)
    Authors(s): denns.liang@hilton.com

    init args
    ----------
    in_features(int):
    out_features(int):
    batchnorm_setting(dict): batch normalization (1d) setting
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        batchnorm_setting: dict = {
            "eps": 1e-05,
            "momentum": 0.1,
            "affine": True,
            "track_running_stats": True,
        },
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.skip_layers = torch.nn.ModuleList()

        # construct skip connection component
        if in_features == out_features:
            # output size == input size
            self.skip_layers.append(torch.nn.Identity())
        else:
            # output size != input size; use a simple linear transformation to change dimension
            self.skip_layers.append(
                torch.nn.Linear(
                    in_features=in_features, out_features=out_features, bias=True
                )
            )

        if batchnorm_setting:
            self.skip_layers.append(
                torch.nn.BatchNorm1d(num_features=out_features, **batchnorm_setting)
            )

    @property
    def in_features(self):
        return self._in_features

    @in_features.setter
    def in_features(self, in_features: int):
        if not isinstance(in_features, int):
            raise ValueError("The <in_features> argument has to be an int.")

        self._in_features = in_features

    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: int):
        if not isinstance(out_features, int):
            raise ValueError("The <out_features> argument has to be an int.")

        self._out_features = out_features

    def forward(self, x):
        for l in self.skip_layers:
            x = l(x)

        return x

    @property
    def input_shape(self):
        return torch.Size([-1, self.in_features])

    @property
    def input_dtype(self):
        return torch.float32

    @property
    def output_shape(self):
        return torch.Size([-1, self.out_features])

    @property
    def output_dtype(self):
        return torch.float32

    def reset(self):
        for m in self.skip_layers:
            if isinstance(m, torch.nn.Linear):
                self.weight_init(m.weight)
                torch.nn.init.ones_(m.bias)
            elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats()


def addResidualConn1D(
    model: nnModule,
    skip_batchnorm_setting: dict = {
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
    },
):
    """function for adding residual connection to a model/module
    Authors(s): denns.liang@hilton.com

    init args
    ----------
    model: neural net model
    """
    model_input_shape = model.input_shape
    model_output_shape = model.output_shape

    if model_input_shape:
        # check model input
        if isinstance(model_input_shape, torch.Size):
            # single tensor input
            skip_connection = skipConnection1D(
                in_features=model_input_shape[-1],
                out_features=model_output_shape[-1],
                batchnorm_setting=skip_batchnorm_setting,
            )

        elif isinstance(model_input_shape, (list, tuple)):
            skip_connection = nnModular(
                [
                    Concat(dim=-1),
                    skipConnection1D(
                        in_features=sum([d[-1] for d in model_input_shape]),
                        out_features=model_output_shape[-1],
                        batchnorm_setting=skip_batchnorm_setting,
                    ),
                ]
            )

        residualBlock = nnParallel(
            {"orig_model": model, "skip": skip_connection},
            use_common_input=True,
            output_combine_method="sum",
        )
    else:
        raise RuntimeError(
            "Cannot determine the required input data shape. Unable to add a residual connection."
        )

    return residualBlock
