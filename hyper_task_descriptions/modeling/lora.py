from typing import Any, Callable, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import lax
from typing_extensions import TypeAlias

from hyper_task_descriptions.modeling.hyper_network import HyperT5Config
from hyper_task_descriptions.modeling.layers import Initializer, SimpleLinear

param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint
from t5x.examples.t5.layers import _canonicalize_tuple, _normalize_axes

NumArray: TypeAlias = jnp.ndarray


def lora_linear(
    inputs: NumArray,
    kernel: NumArray,
    lora_a: NumArray,
    lora_b: NumArray,
    alpha: int,
    rank: int,
    axis: Union[Iterable[int], int] = -1,
) -> NumArray:

    axis = _canonicalize_tuple(axis)
    axis = _normalize_axes(axis, inputs.ndim)
    contract_ind = tuple(range(0, len(axis)))
    dimension_numbers = ((axis, contract_ind), ((), ()))

    # Linear computation: output = W0x
    output = lax.dot_general(inputs, kernel, dimension_numbers)
    # output = output + bias

    # Lora addition: output += BAx

    # TODO: confirm for cases when axis != -1
    x = lax.dot_general(inputs, lora_a, dimension_numbers=dimension_numbers)
    x = lax.dot_general(x, lora_b, dimension_numbers=dimension_numbers)

    output = output + x * (alpha / rank)

    return output


# Mostly copied from t5x/examples/t5/layers.py
# ------------------------------------------------------------------------------
# DenseGeneral for attention layers.
# ------------------------------------------------------------------------------
class LoraDenseGeneral(nn.Module):
    """A linear transformation (without bias) with flexible axes.
    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
    """

    features: Union[Iterable[int], int]
    rank: int
    alpha: int = 1
    axis: Union[Iterable[int], int] = -1
    dtype: jnp.dtype = jnp.float32
    kernel_init: Initializer = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
    kernel_axes: Tuple[str, ...] = ()
    lora_a_init: Initializer = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")

    @nn.compact
    def __call__(self, inputs: NumArray) -> NumArray:
        """Applies a linear transformation to the inputs along multiple dimensions.
        Args:
          inputs: The nd-array to be transformed.
        Returns:
          The transformed input.
        """
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
        kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]), np.prod(features))
        kernel = param_with_axes(
            "kernel", self.kernel_init, kernel_param_shape, jnp.float32, axes=self.kernel_axes
        )
        kernel = jnp.asarray(kernel, self.dtype)
        kernel = jnp.reshape(kernel, kernel_shape)

        # CHANGE from t5x
        assert self.rank > 0
        lora_a = param_with_axes("lora_a", self.lora_a_init, (inputs.shape[-1], self.rank))
        lora_b = param_with_axes("lora_b", nn.initializers.zeros, (self.rank, self.features))
        # contract_ind = tuple(range(0, len(axis)))
        # lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))
        return lora_linear(
            inputs,
            kernel,
            lora_a=lora_a,
            lora_b=lora_b,
            alpha=self.alpha,
            rank=self.rank,
            axis=self.axis,
        )


# class HyperLoraWeights(nn.Module):
#
#     config: HyperT5Config  # TODO: use and generalize this
#     rank: int = 0
#
#     def setup(self):
#         cfg = self.config
#
#         self.lora_a_gen = SimpleLinear(
#             output_dim=cfg.emb_dim * self.rank,
#             act_fn="linear",
#             dropout_rate=cfg.dropout_rate,
#             dtype=cfg.dtype,
#             kernel_axes=("mlp", "embed"),  # TODO: what should they be?
#             kernel_init=nn.initializers.lecun_normal(),  # TODO: needs to be normal gaussian.
#             name="lora_a",
#         )
#
#         self.lora_b_gen = SimpleLinear(
#             output_dim=cfg.emb_dim * self.features,
#             act_fn="linear",
#             dropout_rate=cfg.dropout_rate,
#             dtype=cfg.dtype,
#             kernel_axes=("mlp", "embed"),  # TODO: what should they be?
#             kernel_init=nn.initializers.lecun_normal(),  # TODO: needs to be normal gaussian.
#             name="lora_b",
#         )
