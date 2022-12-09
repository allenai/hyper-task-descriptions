import jax
import jax.numpy as jnp
import numpy as np
from t5x.examples.t5.layers import DenseGeneral, MultiHeadDotProductAttention

from hyper_task_descriptions.common.testing import get_prng_key
from hyper_task_descriptions.modeling.lora import (
    LoraDenseGeneral,
    LoraMultiHeadDotProductAttentionWithPrefix,
    batch_ia3_linear,
    batch_lora_ia3_linear,
    batch_lora_linear,
    ia3_linear,
    lora_ia3_linear,
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

    np.testing.assert_allclose(output, expected_output, rtol=1e-6)


def test_ia3_linear():
    batch_size, in_features, out_features = 3, 4, 5
    rank = 2

    key = get_prng_key(23)
    inputs = jax.random.normal(key, (batch_size, in_features))
    W = jax.random.normal(key, (in_features, out_features))
    A = jax.random.normal(key, (in_features, rank))
    B = jax.random.normal(key, (rank, out_features))

    output = ia3_linear(inputs, W, A, B, 1, rank)
    assert output.shape == (batch_size, out_features)
    expected_output = inputs @ (W * (A @ B) * (1 / rank))

    np.testing.assert_allclose(output, expected_output, rtol=1e-6)


def test_lora_ia3_linear():
    batch_size, in_features, out_features = 3, 4, 5
    rank = 2
    ia3_rank = 4

    key = get_prng_key(23)
    inputs = jax.random.normal(key, (batch_size, in_features))
    W = jax.random.normal(key, (in_features, out_features))
    A = jax.random.normal(key, (in_features, rank))
    B = jax.random.normal(key, (rank, out_features))

    ia3_A = jax.random.normal(key, (in_features, ia3_rank))
    ia3_B = jax.random.normal(key, (ia3_rank, out_features))

    output = lora_ia3_linear(inputs, W, A, B, ia3_A, ia3_B, 1, rank, ia3_rank)
    assert output.shape == (batch_size, out_features)

    ia3_kernel = W * (ia3_A @ ia3_B) * (1 / ia3_rank)
    lora_kernel = ia3_kernel + ((A @ B) * (1 / rank))
    expected_output = inputs @ lora_kernel

    np.testing.assert_allclose(output, expected_output, rtol=1e-6)


def test_batch_lora_ia3_linear():
    batch_size, in_features, out_features = 3, 4, 5
    rank = 2
    ia3_rank = 4

    key = get_prng_key(23)
    inputs = jax.random.normal(key, (batch_size, in_features))
    W = jax.random.normal(key, (in_features, out_features))
    A = jax.random.normal(key, (batch_size, in_features, rank))
    B = jax.random.normal(key, (batch_size, rank, out_features))

    ia3_A = jax.random.normal(key, (batch_size, in_features, ia3_rank))
    ia3_B = jax.random.normal(key, (batch_size, ia3_rank, out_features))

    output = batch_lora_ia3_linear(inputs, W, A, B, ia3_A, ia3_B, 1, rank, ia3_rank)
    assert output.shape == (batch_size, out_features)


def test_batch_ia3_linear():
    batch_size, in_features, out_features = 3, 4, 5
    rank = 2

    key = get_prng_key(23)
    inputs = jax.random.normal(key, (batch_size, in_features))
    W = jax.random.normal(key, (in_features, out_features))
    A = jax.random.normal(key, (batch_size, in_features, rank))
    B = jax.random.normal(key, (batch_size, rank, out_features))

    output = batch_ia3_linear(inputs, W, A, B, 1, rank)
    assert output.shape == (batch_size, out_features)


def test_batch_lora_linear():
    batch_size, in_features, out_features = 3, 4, 5
    rank = 2

    key = get_prng_key(23)
    inputs = jax.random.normal(key, (batch_size, in_features))
    W = jax.random.normal(key, (in_features, out_features))
    A = jax.random.normal(key, (batch_size, in_features, rank))
    B = jax.random.normal(key, (batch_size, rank, out_features))

    output = batch_lora_linear(inputs, W, A, B, 1, rank)
    assert output.shape == (batch_size, out_features)


def test_lora_dense_general():
    batch_size, in_features, out_features = 3, 4, 5
    rank = 2

    inputs = jnp.array(np.random.randn(batch_size, in_features))

    lora_dense = LoraDenseGeneral(out_features, rank=rank, manual_lora=True)
    key = get_prng_key(23)
    params = lora_dense.init(key, inputs)
    assert "lora_a" in params["params"].keys()

    output = lora_dense.apply(params, inputs)
    assert output.shape == (batch_size, out_features)

    # Sanity check: lora's init B is zeros. So, the output should
    # be the same as DenseGeneral.
    dense = DenseGeneral(out_features)
    key = get_prng_key(23)
    dparams = dense.init(key, inputs)
    doutput = dense.apply(dparams, inputs)
    assert (output == doutput).all()

    lora_dense = LoraDenseGeneral(out_features, rank=rank, manual_lora=False)
    A = jax.random.normal(key, (batch_size, in_features, rank))
    B = jax.random.normal(key, (batch_size, rank, out_features))
    key = get_prng_key(23)
    params = lora_dense.init(key, inputs, lora_a=A, lora_b=B)
    assert "lora_a" not in params["params"].keys()


def test_lora_multihead_dot_product_attention_with_prefix():
    batch_size, q_len, q_features, kv_len, kv_features = 3, 4, 5, 6, 7
    num_heads, head_dim = 8, 16
    lora_ranks = (2, None, 2, None)

    inputs_q = jnp.array(np.random.randn(batch_size, q_len, q_features))
    inputs_kv = jnp.array(np.random.randn(batch_size, kv_len, kv_features))

    lora_multihead = LoraMultiHeadDotProductAttentionWithPrefix(
        num_heads=num_heads,
        head_dim=head_dim,
        lora_ranks=lora_ranks,
        use_prefix=False,
        manual_lora=True,
    )
    key = get_prng_key(23)
    params = lora_multihead.init(key, inputs_q, inputs_kv)

    assert "lora_a" in params["params"]["query"]
    assert "lora_a" not in params["params"]["key"]
    assert "lora_a" in params["params"]["value"]
    assert "lora_a" not in params["params"]["out"]
    output = lora_multihead.apply(params, inputs_q, inputs_kv)
    assert output.shape == (batch_size, q_len, q_features)

    # Sanity check
    multihead = MultiHeadDotProductAttention(num_heads=num_heads, head_dim=head_dim)
    key = get_prng_key(23)
    params = multihead.init(key, inputs_q, inputs_kv)

    moutput = multihead.apply(params, inputs_q, inputs_kv)
    assert (output == moutput).all()
