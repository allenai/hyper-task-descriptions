from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp


def cosine_similarity_loss(
    pred_vectors: jnp.ndarray,
    target_vectors: jnp.ndarray,
    ground_truth_similarity: jnp.ndarray,
) -> jnp.ndarray:
    cosine_sim = jax.vmap(cosine_similarity_one_to_many, in_axes=[0, None])(
        pred_vectors, target_vectors
    )
    loss = jnp.mean((cosine_sim - ground_truth_similarity) ** 2)
    return loss


def cosine_similarity_one_to_many(
    pred_vector: jnp.ndarray,
    target_vectors: jnp.ndarray,
) -> jnp.ndarray:
    cosine_sim = cosine_similarity(
        pred_vector[
            None,
        ],
        target_vectors,
    )
    return cosine_sim


def cosine_similarity(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    """
    Computes the cosine similarity between targets and predictions.
    Adapted from optax: https://github.com/deepmind/optax/blob/master/optax/_src/loss.py
    eps default adjusted to 1e-8 to match pytorch.
    """
    # vectorize norm fn, to treat all dimensions except the last as batch dims.
    batched_norm_fn = jnp.vectorize(safe_norm, signature="(k)->()", excluded={1})
    # normalise the last dimension of targets and predictions.
    unit_targets = targets / jnp.expand_dims(batched_norm_fn(targets, epsilon), axis=-1)
    unit_predictions = predictions / jnp.expand_dims(batched_norm_fn(predictions, epsilon), axis=-1)
    # return cosine similarity.
    return jnp.sum(unit_targets * unit_predictions, axis=-1)


# taken whole-cloth from optax.
# https://github.com/deepmind/optax/blob/master/optax/_src/numerics.py#L48
def safe_norm(
    x: jnp.ndarray,
    min_norm: float,
    ord: Optional[Union[int, float, str]] = None,  # pylint: disable=redefined-builtin
    axis: Union[None, Tuple[int, ...], int] = None,
    keepdims: bool = False,
) -> jnp.ndarray:
    """Returns jnp.maximum(jnp.linalg.norm(x), min_norm) with correct gradients.
    The gradients of `jnp.maximum(jnp.linalg.norm(x), min_norm)` at 0.0 is `NaN`,
    because jax will evaluate both branches of the `jnp.maximum`. This function
    will instead return the correct gradient of 0.0 also in such setting.
    Args:
      x: jax array.
      min_norm: lower bound for the returned norm.
      ord: {non-zero int, inf, -inf, optional. Order of the norm.
        inf means numpy’s inf object. The default is None.
      axis: {None, int, 2-tuple of ints}, optional. If axis is an integer, it
        specifies the axis of x along which to compute the vector norms. If axis
        is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix
        norms of these matrices are computed. If axis is None then either a vector
        norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. The
        default is None.
      keepdims: bool, optional. If this is set to True, the axes which are normed
        over are left in the result as dimensions with size one. With this option
        the result will broadcast correctly against the original x.
    Returns:
      The safe norm of the input vector, accounting for correct gradient.
    """
    norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
    x = jnp.where(norm <= min_norm, jnp.ones_like(x), x)
    norm = jnp.squeeze(norm, axis=axis) if not keepdims else norm
    masked_norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    return jnp.where(norm <= min_norm, min_norm, masked_norm)
