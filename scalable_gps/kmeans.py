import jax
import jax.numpy as jnp
from functools import partial
import chex
from jax import random
from typing import NamedTuple


class kmeans_init_state(NamedTuple):
    centroids: list
    points: chex.Array
    key: chex.PRNGKey


@jax.jit
def vector_quantize(
    points: chex.Array, centroids: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """Assign each point to the closest centroid and return the distance to the centroid"""
    chex.assert_rank(points, 2)
    chex.assert_rank(centroids, 2)
    chex.assert_equal_shape_suffix([points, centroids], 1)

    assignment = jax.vmap(
        lambda point: jnp.argmin(jax.vmap(jnp.linalg.norm)(centroids - point))
    )(points)

    dists = jax.vmap(jnp.linalg.norm)(centroids[assignment, :] - points)
    return assignment, dists


def add_centroid(i: int, state: kmeans_init_state) -> kmeans_init_state:
    """Iteration of Kmeans++ initialisation"""
    current_key, key = random.split(state.key)
    _, distortions = vector_quantize(state.points, state.centroids)
    probs = distortions / distortions.sum()
    new_cluster_idx = random.choice(
        current_key, jnp.arange(len(state.points)), shape=(), p=probs
    )

    centroids = state.centroids.at[i].set(state.points[new_cluster_idx])
    return kmeans_init_state(centroids=centroids, points=state.points, key=key)


@partial(jax.jit, static_argnums=(2,))
def kmeanspp(points: chex.Array, key: chex.PRNGKey, k: int) -> chex.Array:
    """Randomly initialise centroids to be datapoints distant from each other."""
    init_cetroids = jnp.zeros((k, points.shape[1]))
    init_cetroids = init_cetroids.at[0].set(points[0])
    state = kmeans_init_state(
        centroids=init_cetroids,
        points=points,
        key=key,
    )

    state = jax.lax.fori_loop(1, k, add_centroid, state)
    return state.centroids


@partial(jax.jit, static_argnums=(2, 3))
def kmeans(
    key: chex.PRNGKey, points: chex.Array, k: int, thresh: float = 1e-5
) -> tuple[chex.Array, chex.Array]:
    """Iterate untill loss improvement is less than thresh.
    Return centroids and loss value.

    Partially based on https://colab.research.google.com/drive/1AwS4haUx6swF82w3nXr6QKhajdF8aSvA
    """

    def improve_centroids(val):
        prev_centroids, prev_distn, _ = val
        assignment, distortions = vector_quantize(points, prev_centroids)

        counts = (
            (assignment[jnp.newaxis, :] == jnp.arange(k)[:, jnp.newaxis])
            .sum(axis=1, keepdims=True)
            .clip(min=1.0)
        )  # clip to change 0/0 later to 0/1

        new_centroids = jnp.zeros_like(prev_centroids)
        dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        new_centroids = jax.lax.scatter_add(
            new_centroids, jnp.expand_dims(assignment, -1), points, dnums
        )
        new_centroids /= counts
        return new_centroids, jnp.mean(distortions), prev_distn

    # Run one iteration to initialize distortions and cause it'll never hurt...
    initial_centroids = kmeanspp(points, key, k)

    initial_val = improve_centroids((initial_centroids, jnp.inf, None))
    # ...then iterate until convergence!
    centroids, distortion, _ = jax.lax.while_loop(
        lambda val: (val[2] - val[1]) > thresh,
        improve_centroids,
        initial_val,
    )
    return centroids, distortion
