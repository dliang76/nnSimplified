import torch
from torch.nn import functional as F
from abc import ABCMeta, abstractmethod
from typing import Optional
from copy import copy

"""define custom loss function for pytorch"""


class _CustomLossBase(torch.nn.Module, metaclass=ABCMeta):
    """Base class template for custom loss fucntions

    Author(s): dliang1122@gmail.com
    """

    def __init__(self, reduction: str = "mean") -> None:
        super(_CustomLossBase, self).__init__()

        self.reduction = reduction

    @abstractmethod
    def calc_loss_tensor(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """abstract method for calculating loss contribution from each prediction; must be defined"""
        pass

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        """common forward method for carrying out loss calculation"""

        loss_tensor = self.calc_loss_tensor(input_tensor, target_tensor)

        return self.reduce(loss_tensor)

    def reduce(self, loss_tensor: torch.Tensor):
        """method for reducing the loss tensor to a single number"""
        # reduced to single number if specified
        if self.reduction == "none":
            return loss_tensor
        elif self.reduction == "sum":
            return loss_tensor.sum()
        elif self.reduction == "mean":
            return loss_tensor.mean()


class _CustomWeightedLoss(_CustomLossBase):
    """Base class template for custom loss functions with class weighting

    Author(s): dliang1122@gmail.com
    """

    def __init__(
        self, weight: Optional[torch.FloatTensor] = None, reduction: str = "mean"
    ) -> None:
        super(_CustomWeightedLoss, self).__init__(reduction=reduction)

        if weight is not None:
            if isinstance(weight, torch.Tensor):
                weight = weight.float()
            elif isinstance(weight, (tuple, list)):
                weight = torch.FloatTensor(weight)
            else:
                raise ValueError(
                    "Unrecognized weight value. Must be a torch tensor, list of numerical values or None"
                )

        self.register_buffer(
            "weight", weight
        )  # register as a torch module buffer instead of parameter to take advantage of built-in move-to-device methods (to(), cuda(), cpu())

    def reduce(self, loss_tensor: torch.Tensor):
        """method for reducing the loss tensor to a single number"""
        # reduced to single number if specified
        if self.reduction == "none":
            return loss_tensor
        elif self.reduction == "sum":
            return loss_tensor.sum()
        elif self.reduction == "mean":
            total_weight = (
                len(loss_tensor) if self.weight is None else self.weight.sum()
            )

            return loss_tensor.sum() / total_weight


class CosineDissimilarityLoss(_CustomLossBase):
    """
    Author(s): kumarsajal49@gmail.com

    A class for cosine dissimlarity loss
    """

    def __init__(self, reduction: str = "mean") -> None:
        # call super().__init__ to inherit backward() method for enabling gradient calculation
        super(CosineDissimilarityLoss, self).__init__(reduction=reduction)

    def calc_loss_tensor(
        self, input1: torch.Tensor, input2: torch.Tensor
    ) -> torch.Tensor:
        """method for calculating loss contribution from each prediction"""

        # calculate loss tensor
        loss_tensor = 1 - F.cosine_similarity(input1, input2, dim=-1)

        return loss_tensor


class MapeLoss(_CustomLossBase):
    """
    Author(s): dliang1122@gmail.com

    A class for MAPE loss
    """

    def __init__(self, reduction: str = "mean") -> None:
        # call super().__init__ to inherit backward() method for enabling gradient calculation
        super().__init__(reduction=reduction)

    def calc_loss_tensor(
        self, preds: torch.Tensor, target: torch.Tensor, epsilon: float = 1.17e-06
    ) -> torch.Tensor:
        """method for calculating loss contribution from each prediction"""

        # calculate loss tensor
        abs_diff = torch.abs(preds - target)
        abs_per_error = abs_diff / torch.clamp(torch.abs(target), min=epsilon)

        loss_tensor = torch.sum(abs_per_error)

        return loss_tensor


class FocalLossBinary(_CustomWeightedLoss):
    """Class for calculating focal loss for binary classifier

    Author(s): dliang1122@gmail.com

    reference
    ----------
    https://arxiv.org/pdf/1708.02002v2.pdf

    init args
    ----------
    gamma (float): focusing (modulation) parameter; used to increase influence(weight) of hard-to-classify points. Larger values -> more weight
    weight (torch.FloatTensor): class weights
    reduction (str): 'mean', 'sum' or 'none'. Point loss aggregation method
    """

    def __init__(
        self,
        gamma: float = 2,
        weight: Optional[torch.FloatTensor] = None,
        reduction: str = "mean",
    ):
        # call super().__init__ to inherit backward() method for enabling gradient calculation
        super(FocalLossBinary, self).__init__(weight=weight, reduction=reduction)
        self.gamma = gamma

    def calc_loss_tensor(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """method for calculating loss contribution from each prediction"""

        target_tensor = (
            target_tensor.float()
        )  # for binary, target tensor has to be of type float

        # get target probability
        pt = target_tensor * torch.sigmoid(input_tensor) + (1 - target_tensor) * (
            1 - torch.sigmoid(input_tensor)
        )

        # get target log probability; use logsigmoid() to improve numerical stability (avoid nan when magnitude of an input is extremely large in either + or - direction)
        log_pt = target_tensor * torch.nn.functional.logsigmoid(input_tensor) + (
            1 - target_tensor
        ) * torch.nn.functional.logsigmoid(-input_tensor)

        # get target weight
        pos_weight = 1 if self.weight is None else self.weight[1] / self.weight[0]
        alphat = target_tensor * pos_weight + (1 - target_tensor) * 1

        # calculate loss for each point
        loss_tensor = -alphat * ((1 - pt) ** self.gamma) * log_pt

        return loss_tensor


class FocalLossMultiClass(_CustomWeightedLoss):
    """Class for calculating focal loss for multiclass classifier

    Author(s): dliang1122@gmail.com

    reference
    ----------
    https://arxiv.org/pdf/1708.02002v2.pdf

    init args
    ----------
    gamma (float): focusing (modulation) parameter; used to increase influence(weight) of hard-to-classify points. Larger values -> more weight
    weight (torch.FloatTensor): class weights
    reduction (str): 'mean', 'sum' or 'none'. Point loss aggregation method
    """

    def __init__(
        self,
        gamma: float = 2,
        weight: Optional[torch.FloatTensor] = None,
        reduction: str = "mean",
    ):
        super(FocalLossMultiClass, self).__init__(weight=weight, reduction=reduction)
        self.gamma = gamma

    def calc_loss_tensor(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """method for calculating loss contribution from each prediction"""

        # get target class probability
        pt = (
            torch.nn.functional.softmax(input_tensor, dim=-1)
            .gather(1, target_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # get target class log probability
        log_pt = (
            torch.nn.functional.log_softmax(input_tensor, dim=-1)
            .gather(1, target_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # get target class weight tensor
        weights = (
            torch.ones(1, device=input_tensor.device)
            if self.weight is None
            else self.weight
        )  # if no weights provided, all classes share the same weight (1)
        class_weights_t = (
            weights.broadcast_to(input_tensor.shape)
            .gather(1, target_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # get loss tensor
        loss_tensor = -class_weights_t * (1 - pt).pow(self.gamma) * log_pt

        return loss_tensor


class FocalLossOrdinal(_CustomWeightedLoss):
    """Class for calculating focal loss for ordinal classifier (experiment)

    Author(s): dliang1122@gmail.com

    reference
    ----------
    https://arxiv.org/pdf/1708.02002v2.pdf

    init args
    ----------
    gamma (float): focusing (modulation) parameter; used to increase influence(weight) of hard-to-classify points. Larger values -> more weight
    beta (float): class distance weighting parameter. Determine how large of a weight to apply if moving away from actual class
    weight (torch.FloatTensor): class weights
    reduction (str): 'mean', 'sum' or 'none'. Point loss aggregation method
    """

    def __init__(
        self,
        gamma: float = 2,
        beta: float = 2,
        weight: torch.FloatTensor = None,
        reduction: str = "mean",
    ):
        super(FocalLossOrdinal, self).__init__(weight=weight, reduction=reduction)
        self.gamma = gamma
        self.beta = beta

    def calc_loss_tensor(
        self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """method for calculating loss contribution from each prediction"""

        # convert input score to class probability using softmax
        class_probs = torch.nn.functional.softmax(input_tensor, dim=-1)

        # get target class probability
        pt = class_probs.gather(1, target_tensor.unsqueeze(1)).squeeze(1)

        # get target class log probability
        log_pt = (
            torch.nn.functional.log_softmax(input_tensor, dim=-1)
            .gather(1, target_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # get target class weight tensor
        weights = (
            torch.ones(1, device=input_tensor.device)
            if self.weight is None
            else self.weight
        )  # if no weights provided, all classes share the same weight (1)
        class_weights_t = (
            weights.broadcast_to(input_tensor.shape)
            .gather(1, target_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # calculate the class distance weight. The farther away from the actual class, the heavier the weight is
        class_distance_weight = (
            (
                torch.arange(0, input_tensor.size(1))
                .float()
                .broadcast_to(input_tensor.shape)
                - target_tensor.unsqueeze(1)
            ).abs()
            + 1
        ).pow(self.beta)  # squared distance from actual class
        class_distance_weight = torch.sum(class_distance_weight * class_probs, dim=1)

        # get loss tensor
        loss_tensor = (
            -class_weights_t * class_distance_weight * (1 - pt).pow(self.gamma) * log_pt
        )

        return loss_tensor


class MultipleLossSingleTarget(_CustomLossBase):
    """Class for calculating loss for multiple output tensors using the same loss function and target
    Requirement: Each output tensor must have the same shape as the target tensor.

    Author(s): dliang1122@gmail.com

    init args
    ----------
    loss (torch.nn.Module): torch loss used
    component_weights (list of numbers): weights applied to the loss value of each output
    component_aggregation (str): loss aggregation methods. 'sum' or 'mean'
    aggregation (str): method for aggregating multiple loss values to a single value. Default: None (no aggregation)
    """

    def __init__(
        self,
        loss: torch.nn.Module = None,
        component_weights: Optional[list[int]] = None,
        component_aggregation: Optional[str] = None,
        reduction: str = "mean",
    ):
        # call super().__init__ to inherit backward() method for enabling gradient calculation
        super().__init__(reduction=reduction)

        # normalize component weights so they sum up to 1
        self.component_weights = (
            torch.tensor(component_weights) / sum(component_weights)
            if component_weights
            else None
        )
        self.component_aggregation = component_aggregation

        self.loss = copy(
            loss
        )  # make a copy so we don't modified the original loss object
        self.__dict__["loss"] = (
            self.loss
        )  # a workaround to make loss the first level attributes. The class inherits from torch.nn.Module, which overwrites the __setattr__ method and always puts Module input in _module attribute.
        self.loss.reduction = (
            self.reduction
        )  # sync the <reduction> attributes in the base loss object

    def calc_loss_tensor(
        self, inputs: list[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        """abstract method for calculating loss contribution from each prediction; must be defined"""

        if not isinstance(inputs, (list, tuple)):
            raise ValueError(
                "Input to the loss aggregation function have to be a list!"
            )

        # use vmap to vectorize the base loss function(object)
        loss_map = torch.vmap(self.loss)

        # calculate loss for every input-target pair. E.g. loss(input 1, target), loss(input 2, target)....
        loss = loss_map(
            torch.stack(inputs, dim=0),
            target.expand(torch.Size([len(inputs)]) + target.shape),
        )

        # weight the loss components if <component_weights> is provided
        if self.component_weights is not None:
            loss = loss * self.component_weights

        # Aggregate
        if self.component_aggregation == "sum":
            loss = loss.sum()
        elif self.component_aggregation == "mean":
            if self.component_weights is not None:
                # weighted average
                loss = loss.sum() / self.component_weights.sum()
            else:
                loss = loss.mean()

        return loss
