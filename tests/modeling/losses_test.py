import jax.numpy as jnp
from hyper_task_descriptions.modeling.losses import (
    cosine_similarity,
    cosine_similarity_loss,
    cosine_similarity_one_to_many,
    # safe_norm,
)


def test_cosine_similarity():

    preds = jnp.array([1, 2, 3])
    targets = jnp.array([1, 2, 3])
    assert jnp.allclose(cosine_similarity(preds, targets), jnp.array(1))

    targets = jnp.array([-1, 0, 1])
    # [1, 2, 3] dot [-1, 0, 1] = -1 + 0 + 3 = 2
    # ||[1, 2, 3]|| = sqrt(14)
    # ||[-1, 0, 1]|| = sqrt(2)
    # cosine_sim = 2/(sqrt(14)*sqrt(2)) = 1/sqrt(7) ~= 0.37796444
    assert jnp.allclose(cosine_similarity(preds, targets), jnp.array(0.37796444))


def test_cosine_similarity_one_to_many():

    preds = jnp.array([1, 2, 3])
    targets = jnp.array([[1, 2, 3], [-1, 0, 1]])
    assert jnp.allclose(cosine_similarity_one_to_many(preds, targets), jnp.array([1, 0.37796444]))


def test_cosine_similarity_loss():
    preds = jnp.array([[1, 2, 3], [1, 2, 3]])
    targets = jnp.array([[1, 2, 3], [-1, 0, 1]])
    gt_sim = jnp.array([1, 0.37796444])

    jnp.allclose(cosine_similarity_loss(preds, targets, gt_sim), jnp.array(0))

    gt_sim = jnp.array([0.5, 0.37796444])
    expected_loss = ((1 - 0.5) ** 2 + 0) / 2
    jnp.allclose(cosine_similarity_loss(preds, targets, gt_sim), jnp.array(expected_loss))


def test_safe_norm():
    pass
