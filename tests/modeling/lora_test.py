import jax
import jax.numpy as jnp
import numpy as np

from hyper_task_descriptions.common.testing import get_prng_key
from hyper_task_descriptions.modeling.lora import (
    LoraDenseGeneral,
    LoraMultiHeadDotProductAttention,
    lora_linear,
)


def test_lora_linear():
    batch_size, in_features, out_features = 3, 4, 5
    rank = 2

    key = get_prng_key(23)
    inputs = jax.random.normal(key, (batch_size, in_features))
    W = jax.random.normal(key, (in_features, out_features))
    A = jax.random.normal(key, (in_features, rank))
    B = jax.random.normal(key, (rank, out_features))

    output = lora_linear(inputs, W, A, B, 1, rank)
    assert output.shape == (batch_size, out_features)
    expected_output = (inputs @ W) + (inputs @ A @ B) * (1 / rank)  # W0x + (BAx*scaling)
    assert jnp.all(output == expected_output)


def test_lora_dense_general():
    batch_size, in_features, out_features = 3, 4, 5
    rank = 2

    inputs = jnp.array(np.random.randn(batch_size, in_features))

    lora_dense = LoraDenseGeneral(out_features, rank=rank, hyper_gen=False)
    key = get_prng_key(23)
    params = lora_dense.init(key, inputs)
    assert "lora_a" in params["params"].keys()

    output = lora_dense.apply(params, inputs)
    assert output.shape == (batch_size, out_features)

    lora_dense = LoraDenseGeneral(out_features, rank=rank, hyper_gen=True)
    A = jax.random.normal(key, (batch_size, in_features, rank))
    B = jax.random.normal(key, (batch_size, rank, out_features))
    key = get_prng_key(23)
    params = lora_dense.init(key, inputs, lora_a=A, lora_b=B)
    assert "lora_a" not in params["params"].keys()


def test_lora_multihead_dot_product_attention():
    batch_size, q_len, q_features, kv_len, kv_features = 3, 4, 5, 6, 7
    num_heads, head_dim = 8, 16
    rank = 2

    inputs_q = jnp.array(np.random.randn(batch_size, q_len, q_features))
    inputs_kv = jnp.array(np.random.randn(batch_size, kv_len, kv_features))

    lora_multihead = LoraMultiHeadDotProductAttention(
        num_heads=num_heads, head_dim=head_dim, rank=rank
    )
    key = get_prng_key(23)
    params = lora_multihead.init(key, inputs_q, inputs_kv)

    output = lora_multihead.apply(params, inputs_q, inputs_kv)
    assert output.shape == (batch_size, q_len, q_features)

