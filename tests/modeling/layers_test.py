import dataclasses

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze
from jax import random

from hyper_task_descriptions.modeling import layers

# from t5x.examples.t5.layers import MultiHeadDotProductAttention


def test_simple_linear():
    module = layers.SimpleLinear(
        output_dim=16,
        act_fn="relu",
        kernel_init=nn.initializers.xavier_uniform(),
        dtype=jnp.float32,
    )

    inputs = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32,
    )  # 2 x 3 x 2
    params = module.init(random.PRNGKey(0), inputs, deterministic=True)
    assert "wi" in params["params_axes"]
    assert params["params"]["wi"]["kernel"].shape == (2, 16)
    result = module.apply(params, inputs, deterministic=True)
    assert result.shape == (2, 3, 16)


def test_mlp_block():
    module = layers.MlpBlock(
        intermediate_dim=4,
        activations=("relu",),
        kernel_init=nn.initializers.xavier_uniform(),
        dtype=jnp.float32,
    )
    inputs = np.array(
        [
            # Batch 1.
            [[1, 1], [1, 1], [1, 2]],
            # Batch 2.
            [[2, 2], [3, 1], [2, 2]],
        ],
        dtype=np.float32,
    )  # 2 x 3 x 2
    params = module.init(random.PRNGKey(0), inputs, deterministic=True)

    assert "wi" in params["params_axes"] and "wo" in params["params_axes"]
    assert params["params_axes"]["wi"]["kernel_axes"].names == ("embed", "mlp")
    assert params["params_axes"]["wo"]["kernel_axes"].names == ("mlp", "embed")

    assert params["params"]["wi"]["kernel"].shape == (2, 4)
    assert params["params"]["wo"]["kernel"].shape == (4, 2)

    result = module.apply(params, inputs, deterministic=True)

    assert inputs.shape == result.shape


@dataclasses.dataclass(frozen=True)
class SelfAttentionArgs:
    num_heads: int = 1
    batch_size: int = 2
    # qkv_features: int = 3
    head_dim: int = 3
    # out_features: int = 4
    q_len: int = 5
    features: int = 6
    dropout_rate: float = 0.1
    deterministic: bool = False
    decode: bool = False
    float32_logits: bool = False

    def __post_init__(self):
        # If we are doing decoding, the query length should be 1, because are doing
        # autoregressive decoding where we feed one position at a time.
        assert not self.decode or self.q_len == 1

    def init_args(self):
        return dict(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout_rate=self.dropout_rate,
            float32_logits=self.float32_logits,
        )

    def apply_args(self):
        inputs_q = jnp.ones((self.batch_size, self.q_len, self.features))
        mask = jnp.ones((self.batch_size, self.num_heads, self.q_len, self.q_len))
        bias = jnp.ones((self.batch_size, self.num_heads, self.q_len, self.q_len))
        return {
            "inputs_q": inputs_q,
            "mask": mask,
            "bias": bias,
            "deterministic": self.deterministic,
        }


@pytest.mark.parametrize("f", [20, 22])
def test_multihead_dot_product_attention_with_prefix(f):
    # b: batch, f: emb_dim, q: q_len, k: kv_len, h: num_head, d: head_dim, p: num_prefix_tokens
    b, q, h, d, k, p = 2, 3, 4, 5, 6, 7

    base_args = SelfAttentionArgs(num_heads=h, head_dim=d, dropout_rate=0)
    args = base_args.init_args()

    np.random.seed(0)
    inputs_q = np.random.randn(b, q, f)
    inputs_kv = np.random.randn(b, k, f)
    key_prefix = np.random.randn(b, p, h, d)
    value_prefix = np.random.randn(b, p, h, d)

    # Projection: [b, q, f] -> [b, q, h, d]
    # So the kernels have to be [f, h, d]
    query_kernel = np.random.randn(f, h, d)
    key_kernel = np.random.randn(f, h, d)
    value_kernel = np.random.randn(f, h, d)
    # `out` calculation: [b, q, h, d] -> [b, q, f]
    # So kernel has to be [h, d, f]
    out_kernel = np.random.randn(h, d, f)

    params = {
        "query": {"kernel": query_kernel.reshape(f, -1)},
        "key": {"kernel": key_kernel.reshape(f, -1)},
        "value": {"kernel": value_kernel.reshape(f, -1)},
        "out": {"kernel": out_kernel.reshape(-1, f)},
    }
    y = layers.MultiHeadDotProductAttentionWithPrefix(**args).apply(
        {"params": freeze(params)}, inputs_q, inputs_kv, key_prefix, value_prefix
    )

    query = np.einsum("bqf,fhd->bqhd", inputs_q, query_kernel)
    key = np.einsum("bkf,fhd->bkhd", inputs_kv, key_kernel)
    value = np.einsum("bkf,fhd->bkhd", inputs_kv, value_kernel)

    key = np.concatenate([key_prefix, key], axis=1)
    value = np.concatenate([value_prefix, value], axis=1)

    logits = np.einsum("bqhd,bkhd->bhqk", query, key)
    weights = nn.softmax(logits, axis=-1)
    combined_value = np.einsum("bhqk,bkhd->bqhd", weights, value)
    y_expected = np.einsum("bqhd,hdf->bqf", combined_value, out_kernel)
    np.testing.assert_allclose(y, y_expected, rtol=1e-4, atol=1e-4)

    # TODO: test with bias
