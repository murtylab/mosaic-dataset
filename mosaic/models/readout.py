import warnings

import numpy as np
import torch
from torch import nn as nn

import warnings
from typing import Any, Literal, Mapping, Optional

import torch
from torch import nn as nn
from torch.nn.modules import Module
from torch.nn.parameter import Parameter

Reduction = Literal["sum", "mean", None]


class ConfigurationError(Exception):
    pass


# ------------------ Base Classes -------------------------


class Readout(Module):
    """
    Base readout class for all individual readouts.
    The MultiReadout will expect its readouts to inherit from this base class.
    """

    features: Parameter
    bias: Parameter

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "initialize is not implemented for ", self.__class__.__name__
        )

    def regularizer(
        self, reduction: Reduction = "sum", average: Optional[bool] = None
    ) -> torch.Tensor:
        raise NotImplementedError(
            "regularizer is not implemented for ", self.__class__.__name__
        )

    def apply_reduction(
        self,
        x: torch.Tensor,
        reduction: Reduction = "mean",
        average: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Applies a reduction on the output of the regularizer.
        Args:
            x: output of the regularizer
            reduction: method of reduction for the regularizer. Currently possible are ['mean', 'sum', None].
            average: Deprecated. Whether to average the output of the regularizer.
                            If not None, it is transformed into the corresponding value of 'reduction' (see method 'resolve_reduction_method').

        Returns: reduced value of the regularizer
        """
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)

        if reduction == "mean":
            return x.mean()
        elif reduction == "sum":
            return x.sum()
        elif reduction is None:
            return x
        else:
            raise ValueError(
                f"Reduction method '{reduction}' is not recognized. Valid values are ['mean', 'sum', None]"
            )

    def resolve_reduction_method(
        self, reduction: Reduction = "mean", average: Optional[bool] = None
    ) -> Reduction:
        """
        Helper method which transforms the old and deprecated argument 'average' in the regularizer into
        the new argument 'reduction' (if average is not None). This is done in order to agree with the terminology in pytorch).
        """
        if average is not None:
            warnings.warn(
                "Use of 'average' is deprecated. Please consider using `reduction` instead"
            )
            reduction = "mean" if average else "sum"
        return reduction

    def resolve_deprecated_gamma_readout(
        self,
        feature_reg_weight: Optional[float],
        gamma_readout: Optional[float],
        default: float = 1.0,
    ) -> float:
        if gamma_readout is not None:
            warnings.warn(
                "Use of 'gamma_readout' is deprecated. Use 'feature_reg_weight' instead. If 'feature_reg_weight' is defined, 'gamma_readout' is ignored"
            )

        if feature_reg_weight is None:
            if gamma_readout is not None:
                feature_reg_weight = gamma_readout
            else:
                feature_reg_weight = default
        return feature_reg_weight

    def initialize_bias(self, mean_activity: Optional[torch.Tensor] = None) -> None:
        """
        Initialize the biases in readout.
        Args:
            mean_activity: Tensor containing the mean activity of neurons.

        Returns:

        """
        if mean_activity is None:
            warnings.warn("Readout is NOT initialized with mean activity but with 0!")
            self.bias.data.fill_(0)
        else:
            self.bias.data = mean_activity

    def __repr__(self) -> str:
        return super().__repr__() + " [{}]\n".format(self.__class__.__name__)  # type: ignore[no-untyped-call,no-any-return]


class FullFactorized2d(Readout):
    """
    Factorized fully connected layer. Weights are a sum of outer products between a spatial filter and a feature vector.
    """

    def __init__(
        self,
        in_shape,  # channels, height, width
        outdims,
        bias,
        normalize=True,
        init_noise=1e-3,
        constrain_pos=False,
        positive_weights=False,
        positive_spatial=False,
        shared_features=None,
        mean_activity=None,
        spatial_and_feature_reg_weight=None,
        gamma_readout=None,
        **kwargs,
    ):
        """

        Args:
            in_shape: batch, channels, height, width (batch could be arbitrary)
            outdims: number of neurons to predict
            bias: if True, bias is used
            normalize: if True, normalizes the spatial mask using l2 norm
            init_noise: the std for readout  initialisation
            constrain_pos: if True, negative values in the spatial mask and feature readout are clamped to 0
            positive_weights: if True, negative values in the feature readout are turned into 0
            positive_spatial: if True, spatial readout mask values are restricted to be positive by taking the absolute values
            shared_features: if True, uses a copy of the features from somewhere else
            mean_activity: the mean for readout  initialisation
            spatial_and_feature_reg_weight: lagrange multiplier (constant) for L1 penalty,
                the bigger the number, the stronger the penalty
            gamma_readout: depricated, use spatial_and_feature_reg_weight instead
            **kwargs:
        """

        super().__init__()

        h, w = in_shape[1:]  # channels, height, width
        self.in_shape = in_shape
        self.outdims = outdims
        self.positive_weights = positive_weights
        self.constrain_pos = constrain_pos
        self.positive_spatial = positive_spatial
        if positive_spatial and constrain_pos:
            warnings.warn(
                f"If both positive_spatial and constrain_pos are True, "
                f"only constrain_pos will effectively take place"
            )
        self.init_noise = init_noise
        self.normalize = normalize
        self.mean_activity = mean_activity
        self.spatial_and_feature_reg_weight = self.resolve_deprecated_gamma_readout(
            spatial_and_feature_reg_weight, gamma_readout, default=1.0
        )

        self._original_features = True
        self.initialize_features(**(shared_features or {}))
        self.spatial = nn.Parameter(torch.Tensor(self.outdims, h, w))

        if bias:
            bias = nn.Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.initialize()

    @property
    def shared_features(self):
        return self._features

    @property
    def features(self):
        if self._shared_features:
            return self.scales * self._features[self.feature_sharing_index, ...]
        else:
            return self._features

    @property
    def weight(self):
        if self.positive_weights:
            self.features.data.clamp_min_(0)
        n = self.outdims
        c, h, w = self.in_shape
        return self.normalized_spatial.view(n, 1, w, h) * self.features.view(n, c, 1, 1)

    @property
    def normalized_spatial(self):
        """
        Normalize the spatial mask
        """
        if self.normalize:
            norm = self.spatial.pow(2).sum(dim=1, keepdim=True)
            norm = norm.sum(dim=2, keepdim=True).sqrt().expand_as(self.spatial) + 1e-6
            weight = self.spatial / norm
        else:
            weight = self.spatial
        if self.constrain_pos:
            weight.data.clamp_min_(0)
        elif self.positive_spatial:
            weight = torch.abs(weight)
        return weight

    def regularizer(self, reduction="sum", average=None):
        return (
            self.l1(reduction=reduction, average=average)
            * self.spatial_and_feature_reg_weight
        )

    def l1(self, reduction="sum", average=None):
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)
        if reduction is None:
            raise ValueError("Reduction of None is not supported in this regularizer")

        n = self.outdims
        c, h, w = self.in_shape
        ret = (
            self.normalized_spatial.view(self.outdims, -1)
            .abs()
            .sum(dim=1, keepdim=True)
            * self.features.view(self.outdims, -1).abs().sum(dim=1)
        ).sum()
        if reduction == "mean":
            ret = ret / (n * c * w * h)
        return ret

    def initialize(self, mean_activity=None):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """
        if mean_activity is None:
            mean_activity = self.mean_activity
        self.spatial.data.normal_(0, self.init_noise)
        self._features.data.normal_(0, self.init_noise)
        if self._shared_features:
            self.scales.data.fill_(1.0)
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c = self.in_shape[0]
        if match_ids is not None:
            assert self.outdims == len(match_ids)

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (
                    n_match_ids,
                    c,
                ), f"shared features need to have shape ({n_match_ids}, {c})"
                self._features = shared_features
                self._original_features = False
            else:
                self._features = nn.Parameter(
                    torch.Tensor(n_match_ids, c)
                )  # feature weights for each channel of the core
            self.scales = nn.Parameter(
                torch.Tensor(self.outdims, 1)
            )  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer("feature_sharing_index", torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = nn.Parameter(
                torch.Tensor(self.outdims, c)
            )  # feature weights for each channel of the core
            self._shared_features = False

    def forward(self, x, shift=None, **kwargs):
        if shift is not None:
            raise NotImplementedError("shift is not implemented for this readout")
        if self.constrain_pos:
            self.features.data.clamp_min_(0)

        c, h, w = x.size()[1:]
        c_in, h_in, w_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError(
                "the specified feature map dimension is not the readout's expected input dimension"
            )

        y = torch.einsum("ncwh,owh->nco", x, self.normalized_spatial)
        y = torch.einsum("nco,oc->no", y, self.features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        c, h, w = self.in_shape
        r = (
            self.__class__.__name__
            + " ("
            + "{} x {} x {}".format(c, w, h)
            + " -> "
            + str(self.outdims)
            + ")"
        )
        if self.bias is not None:
            r += " with bias"
        if self._shared_features:
            r += ", with {} features".format(
                "original" if self._original_features else "shared"
            )
        if self.normalize:
            r += ", normalized"
        else:
            r += ", unnormalized"
        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r


# Classes for backwards compatibility
class SpatialXFeatureLinear(FullFactorized2d):
    pass
