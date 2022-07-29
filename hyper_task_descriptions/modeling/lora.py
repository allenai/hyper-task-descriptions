import functools
from typing import Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import lax
from typing_extensions import TypeAlias

from hyper_task_descriptions.modeling.layers import Initializer
from t5x.examples.t5.layers import (
    _canonicalize_tuple,
    _normalize_axes,
    combine_biases,
    combine_masks,
    dot_product_attention,
    dynamic_vector_slice_in_dim,
)

param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


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
    x = lax.dot_general(inputs, lora_a, dimension_numbers=dimension_numbers)

    b_axis = _normalize_axes((-1,), x.ndim)
    b_contract_ind = tuple(range(0, len(b_axis)))
    b_dimension_numbers = ((b_axis, b_contract_ind), ((), ()))
    x = lax.dot_general(x, lora_b, dimension_numbers=b_dimension_numbers)

    output = output + x * (alpha / rank)

    return output


def batch_lora_linear(
    inputs: NumArray,
    kernel: NumArray,
    lora_a: NumArray,
    lora_b: NumArray,
    alpha: int,
    rank: int,
    axis: Union[Iterable[int], int] = -1,
) -> NumArray:

    axis = _canonicalize_tuple(axis)
    k_axis = _normalize_axes(axis, inputs.ndim)
    contract_ind = tuple(range(0, len(k_axis)))
    dimension_numbers = ((k_axis, contract_ind), ((), ()))

    # Linear computation: output = W0x
    output = lax.dot_general(inputs, kernel, dimension_numbers)
    # output = output + bias

    # Lora addition: output += BAx
    a_axis = _normalize_axes(axis, lora_a.ndim)
    a_contract_ind = tuple(range(1, 1 + len(a_axis))) #_normalize_axes((-2,), lora_a.ndim)
    a_dimension_numbers = ((a_axis, a_contract_ind), ((0,), (0,)))

    x = lax.dot_general(inputs, lora_a, dimension_numbers=a_dimension_numbers)

    b_axis = _normalize_axes((-1,), x.ndim)
    # b_contract_ind = tuple(range(0, len(b_axis)))
    b_contract_ind = tuple(range(1, 1 + len(b_axis))) #_normalize_axes((-3,), lora_b.ndim) #tuple(range(lora_b.ndim-1, len(b_axis)))
    b_dimension_numbers = ((b_axis, b_contract_ind), ((0,), (0,)))
    x = lax.dot_general(x, lora_b, dimension_numbers=b_dimension_numbers)

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
    hyper_gen: bool = False  # TODO: handle this better?

    @nn.compact
    def __call__(
        self, inputs: NumArray, lora_a: Optional[NumArray] = None, lora_b: Optional[NumArray] = None
    ) -> NumArray:
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

        if not self.hyper_gen:
            lora_a_shape = tuple([inputs.shape[ax] for ax in axis]) + tuple([self.rank])
            lora_a_param_shape = (np.prod([inputs.shape[ax] for ax in axis]), self.rank)
            lora_a = param_with_axes(
                "lora_a", self.lora_a_init, lora_a_param_shape, axes=self.kernel_axes
            )  # TODO: what should the axes be?
            lora_a = jnp.asarray(lora_a, self.dtype)
            lora_a = jnp.reshape(lora_a, lora_a_shape)

            lora_b_shape = tuple([self.rank]) + features
            lora_b_param_shape = (self.rank, np.prod(features))
            lora_b = param_with_axes(
                "lora_b", nn.initializers.zeros, lora_b_param_shape
            )  # TODO: axes?
            lora_b = jnp.asarray(lora_b, self.dtype)
            lora_b = jnp.reshape(lora_b, lora_b_shape)

            return lora_linear(
                inputs,
                kernel,
                lora_a=lora_a,
                lora_b=lora_b,
                alpha=self.alpha,
                rank=self.rank,
                axis=self.axis,
            )
        else:
            return batch_lora_linear(
                inputs,
                kernel,
                lora_a=lora_a,
                lora_b=lora_b,
                alpha=self.alpha,
                rank=self.rank,
                axis=self.axis,
            )


class LoraMultiHeadDotProductAttention(nn.Module):
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
    rank: int
    hyper_gen: bool = False
    dtype: jnp.dtype = jnp.float32
    dropout_rate: float = 0.0
    kernel_init: Initializer = nn.initializers.variance_scaling(1.0, "fan_in", "normal")
    float32_logits: bool = False  # computes logits in float32 for stability.

    @nn.compact
    def __call__(
        self,
        inputs_q: NumArray,
        inputs_kv: NumArray,
        mask: Optional[NumArray] = None,
        bias: Optional[NumArray] = None,
        lora_qa: Optional[NumArray] = None,
        lora_qb: Optional[NumArray] = None,
        lora_ka: Optional[NumArray] = None,
        lora_kb: Optional[NumArray] = None,
        lora_va: Optional[NumArray] = None,
        lora_vb: Optional[NumArray] = None,
        lora_oa: Optional[NumArray] = None,
        lora_ob: Optional[NumArray] = None,
        *,
        decode: bool = False,
        deterministic: bool = False
    ) -> NumArray:
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
            mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
            bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
            decode: Whether to prepare and use an autoregressive cache.
            deterministic: Disables dropout if set to True.

        Returns:
            output of shape `[batch, length, q_features]`.
        """
        projection = functools.partial(
            LoraDenseGeneral,
            rank=self.rank,
            axis=-1,
            features=(self.num_heads, self.head_dim),
            kernel_axes=("embed", "joined_kv"),
            dtype=self.dtype,
            hyper_gen=self.hyper_gen,
        )

        # NOTE: T5 does not explicitly rescale the attention logits by
        #       1/sqrt(depth_kq)!  This is folded into the initializers of the
        #       linear transformations, which is equivalent under Adafactor.
        depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
        query_init = lambda *args: self.kernel_init(*args) / depth_scaling  # noqa: E731

        # Project inputs_q to multi-headed q/k/v
        # dimensions are then [batch, length, num_heads, head_dim]

        # TODO: allow turning off lora for q, k, v, o.
        query = projection(kernel_init=query_init, name="query")(
            inputs_q, lora_a=lora_qa, lora_b=lora_qb
        )
        key = projection(kernel_init=self.kernel_init, name="key")(
            inputs_kv, lora_a=lora_ka, lora_b=lora_kb
        )
        value = projection(kernel_init=self.kernel_init, name="value")(
            inputs_kv, lora_a=lora_va, lora_b=lora_vb
        )

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
        out = LoraDenseGeneral(
            features=inputs_q.shape[-1],  # output dim is set to the input dim.
            axis=(-2, -1),
            rank=self.rank,
            kernel_init=self.kernel_init,
            kernel_axes=("joined_kv", "embed"),
            dtype=self.dtype,
            name="out",
            hyper_gen=self.hyper_gen,
        )(x, lora_a=lora_oa, lora_b=lora_ob)
        return out


# class HyperLoraWeights(nn.Module):
#
#     config: HyperT5Config  # TODO: use rank within this.
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
#
#     def __call__(self, *args, **kwargs):
#
#         lora_a = self.lora_a_gen(intermediate_embeddings, deterministic=deterministic)
#         adapter_down = jnp.reshape(adapter_down, (-1, total_layers, cfg.emb_dim, cfg.adapter_size))
