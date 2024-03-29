# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Layers with changes for my model
"""
import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import lax
from t5x.examples.t5.layers import (
    DenseGeneral,
    _convert_to_activation_function,
    combine_biases,
    combine_masks,
    dot_product_attention,
    dynamic_vector_slice_in_dim,
)
from typing_extensions import TypeAlias

# from flax.linen.partitioning import param_with_axes, with_sharding_constraint
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


# Type annotations
Array: TypeAlias = jnp.ndarray
DType: TypeAlias = jnp.dtype
PRNGKey: TypeAlias = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]


class SimpleLinear(nn.Module):
    """Feed-forward block that allows output values to be set.
    TODO: hypernet init

    Attributes:
      output_dim: Output dimension.
      activations: Type of activations for layer. Can be string or flax module.
      kernel_init: Kernel function, passed to the dense layers.
      deterministic: Whether the dropout layers should be deterministic.
      intermediate_dropout_rate: Dropout rate used after the intermediate layers.
      dtype: Type for the dense layer.
    """

    output_dim: int = 2048
    act_fn: Union[str, Callable] = "gelu"
    kernel_init: Initializer = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    kernel_axes: Tuple[str, ...] = ()

    @nn.compact
    def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
        """Applies SimpleLinear module."""
        # Iterate over specified MLP input activation functions.
        # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
        x = DenseGeneral(
            self.output_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=self.kernel_axes,
            name="wi",
        )(inputs)
        x = _convert_to_activation_function(self.act_fn)(x)
        output = nn.Dropout(rate=self.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic
        )  # Broadcast along length.

        return output


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
      intermediate_dim: Shared dimension of hidden layers.
      activations: Type of activations for each layer.  Each element is either
        'linear', a string function name in flax.linen, or a function.
      kernel_init: Kernel function, passed to the dense layers.
      deterministic: Whether the dropout layers should be deterministic.
      intermediate_dropout_rate: Dropout rate used after the intermediate layers.
      dtype: Type for the dense layer.
    """

    intermediate_dim: int = 2048
    output_dim: Optional[int] = None  # by default we preserve the input dim
    activations: Sequence[Union[str, Callable]] = ("relu",)
    kernel_init: Initializer = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
    intermediate_dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
        """Applies Transformer MlpBlock module."""
        # Iterate over specified MLP input activation functions.
        # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
        activations = []
        for idx, act_fn in enumerate(self.activations):
            dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
            x = DenseGeneral(
                self.intermediate_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                kernel_axes=("embed", "mlp"),
                name=dense_name,
            )(inputs)
            x = _convert_to_activation_function(act_fn)(x)
            activations.append(x)

        # Take elementwise product of above intermediate activations.
        x = functools.reduce(operator.mul, activations)
        # Apply dropout and final dense output projection.
        x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic
        )  # Broadcast along length.

        # CHANGE from t5x
        # Removing the sharding constraint as we require to use this layer for shapes of ('batch', 'mlp'),
        # which makes below constraint invalid.
        # x = with_sharding_constraint(x, ('batch', 'length', 'mlp'))

        output = DenseGeneral(
            inputs.shape[-1] if self.output_dim is None else self.output_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("mlp", "embed"),
            name="wo",
        )(x)
        return output


class MultiHeadDotProductAttentionWithPrefix(nn.Module):
    """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
    """

    num_heads: int
    head_dim: int
    dtype: DType = jnp.float32
    dropout_rate: float = 0.0
    kernel_init: Initializer = nn.initializers.variance_scaling(1.0, "fan_in", "normal")
    float32_logits: bool = False  # computes logits in float32 for stability.
    use_prefix: bool = True  # use prefix or not

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        key_prefix: Array,
        value_prefix: Array,
        mask: Optional[Array] = None,
        bias: Optional[Array] = None,
        *,
        decode: bool = False,
        deterministic: bool = False,
    ) -> Array:
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        There are two modes: decoding and non-decoding (e.g., training). The mode is
        determined by `decode` argument. For decoding, this method is called twice,
        first to initialize the cache and then for an actual decoding process. The
        two calls are differentiated by the presence of 'cached_key' in the variable
        dict. In the cache initialization stage, the cache variables are initialized
        as zeros and will be filled in the subsequent decoding process.

        In the cache initialization call, `inputs_q` has a shape [batch, length,
        q_features] and `inputs_kv`: [batch, length, kv_features]. During the
        incremental decoding stage, query, key and value all have the shape [batch,
        1, qkv_features] corresponding to a single step.

        Args:
          inputs_q: input queries of shape `[batch, q_length, q_features]`.
          inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
          key_prefix: key prefix of shape `[batch, num_prefix_tokens, num_heads, head_dim]`.
          value_prefix: value prefix of shape `[batch, num_prefix_tokens, num_heads, head_dim]`.
          mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
          bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
          decode: Whether to prepare and use an autoregressive cache.
          deterministic: Disables dropout if set to True.

        Returns:
          output of shape `[batch, length, q_features]`.
        """
        projection = functools.partial(
            DenseGeneral,
            axis=-1,
            features=(self.num_heads, self.head_dim),
            kernel_axes=("embed", "joined_kv"),
            dtype=self.dtype,
        )

        # NOTE: T5 does not explicitly rescale the attention logits by
        #       1/sqrt(depth_kq)!  This is folded into the initializers of the
        #       linear transformations, which is equivalent under Adafactor.
        depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
        query_init = lambda *args: self.kernel_init(*args) / depth_scaling  # noqa: E731

        # Project inputs_q to multi-headed q/k/v
        # dimensions are then [batch, length, num_heads, head_dim]
        query = projection(kernel_init=query_init, name="query")(inputs_q)
        key = projection(kernel_init=self.kernel_init, name="key")(inputs_kv)
        value = projection(kernel_init=self.kernel_init, name="value")(inputs_kv)

        query = with_sharding_constraint(query, ("batch", "length", "heads", "kv"))
        key = with_sharding_constraint(key, ("batch", "length", "heads", "kv"))
        value = with_sharding_constraint(value, ("batch", "length", "heads", "kv"))

        if decode:
            # Detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable("cache", "cached_key")
            # The key and value have dimension [batch, length, num_heads, head_dim],
            # but we cache them as [batch, num_heads, head_dim, length] as a TPU
            # fusion optimization. This also enables the "scatter via one-hot
            # broadcast" trick, which means we do a one-hot broadcast instead of a
            # scatter/gather operations, resulting in a 3-4x speedup in practice.
            swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])  # noqa: E731
            cached_key = self.variable(
                "cache", "cached_key", jnp.zeros, swap_dims(key.shape), key.dtype
            )
            cached_value = self.variable(
                "cache", "cached_value", jnp.zeros, swap_dims(value.shape), value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )
            if is_initialized:
                batch, num_heads, head_dim, length = cached_key.value.shape
                # During fast autoregressive decoding, we feed one position at a time,
                # and cache the keys and values step by step.
                # Sanity shape check of cached key against input query.
                expected_shape = (batch, 1, num_heads, head_dim)
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        "expected query shape %s instead got %s." % (expected_shape, query.shape)
                    )

                # Create a OHE of the current index. NOTE: the index is increased below.
                cur_index = cache_index.value
                one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
                # In order to update the key, value caches with the current key and
                # value, we move the length axis to the back, similar to what we did for
                # the cached ones above.
                # Note these are currently the key and value of a single position, since
                # we feed one position at a time.
                one_token_key = jnp.moveaxis(key, -3, -1)
                one_token_value = jnp.moveaxis(value, -3, -1)
                # Update key, value caches with our new 1d spatial slices.
                # We implement an efficient scatter into the cache via one-hot
                # broadcast and addition.
                key = cached_key.value + one_token_key * one_hot_indices
                value = cached_value.value + one_token_value * one_hot_indices
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # Move the keys and values back to their original shapes.
                key = jnp.moveaxis(key, -1, -3)
                value = jnp.moveaxis(value, -1, -3)

                # Causal mask for cached decoder self-attention: our single query
                # position should only attend to those key positions that have already
                # been generated and cached, not the remaining zero elements.
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(length) <= cur_index,
                        # (1, 1, length) represent (head dim, query length, key length)
                        # query length is 1 because during decoding we deal with one
                        # index.
                        # The same mask is applied to all batch elements and heads.
                        (batch, 1, 1, length),
                    ),
                )

                # Grab the correct relative attention bias during decoding. This is
                # only required during single step decoding.
                if bias is not None:
                    # The bias is a full attention matrix, but during decoding we only
                    # have to take a slice of it.
                    # This is equivalent to bias[..., cur_index:cur_index+1, :].
                    bias = dynamic_vector_slice_in_dim(
                        jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2
                    )

        # CHANGE from t5x
        # ADD PREFIXES ###
        # key has dim [batch, len, num_heads, head_dim], and we add prefixes
        if self.use_prefix:
            key = jnp.concatenate([key_prefix, key], axis=1)
            value = jnp.concatenate([value_prefix, value], axis=1)
        ####################

        # Convert the boolean attention mask to an attention bias.
        if mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                mask > 0,
                jnp.full(mask.shape, 0.0).astype(self.dtype),
                jnp.full(mask.shape, -1e10).astype(self.dtype),
            )
        else:
            attention_bias = None

        # Add provided bias term (e.g. relative position embedding).
        if bias is not None:
            attention_bias = combine_biases(attention_bias, bias)

        # CHANGE from t5x
        # PREFIX CHANGE
        # Avoid attention bias affecting the prefixes by prepending 0s
        # attention_bias has shape [batch, num_heads, q_length, kv_length]
        if attention_bias is not None and self.use_prefix:
            num_prefix_toks = key_prefix.shape[1]
            batch, num_heads, q_length, _ = attention_bias.shape
            attention_bias = jnp.concatenate(
                [jnp.empty((batch, num_heads, q_length, num_prefix_toks)), attention_bias],
                axis=-1,
            )
        ###

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.0:
            dropout_rng = self.make_rng("dropout")

        # Apply attention.
        x = dot_product_attention(
            query,
            key,
            value,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            dtype=self.dtype,
            float32_logits=self.float32_logits,
        )

        # Back to the original inputs dimensions.
        out = DenseGeneral(
            features=inputs_q.shape[-1],  # output dim is set to the input dim.
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            kernel_axes=("joined_kv", "embed"),
            dtype=self.dtype,
            name="out",
        )(x)
        return out
