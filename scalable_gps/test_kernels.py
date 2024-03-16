import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from scalable_gps.kernels import TanimotoKernel, TanimotoL1Kernel


@pytest.mark.parametrize("Kernel", [TanimotoKernel, TanimotoL1Kernel])
def test_scalar_kernel(Kernel):
    kernel = Kernel({"signal_scale": 1.0})
    result = kernel.kernel_fn(
        jnp.array([[1.0, 2.0, 3.0]]), jnp.array([[3.0, 2.0, 1.0]])
    )

    assert jnp.isclose(jnp.squeeze(result), 0.5).all()


def test_random_features():
    # Compute many random features

    x1 = jnp.array([[1.0, 2.0, 3.0]])
    x2 = jnp.array([[3.0, 2.0, 1.0]])
    modulo_value = 8

    kernel = TanimotoKernel({"signal_scale": 1.0})

    feature_params = kernel.feature_params_fn(
        jr.PRNGKey(0), n_features=1000, n_input_dims=3, modulo_value=modulo_value
    )

    x1_features = kernel.feature_fn(x1, feature_params)

    x2_features = kernel.feature_fn(x2, feature_params)

    print(x1.shape, x2.shape, x1_features.shape, x2_features.shape)

    # Compute inner product between features
    rf_prod = jnp.mean(
        x1_features * x2_features
    )  # mean is because the features are not properly normalized

    # Test: is random feature inner product correct?
    print("test", rf_prod)
    assert jnp.all(jnp.abs(rf_prod - 0.5) < 0.05)
