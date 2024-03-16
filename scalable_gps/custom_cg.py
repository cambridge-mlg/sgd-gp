import operator
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import device_put, lax
from jax.tree_util import Partial, tree_leaves, tree_map, tree_structure

_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_vdot = partial(jnp.vdot, precision=lax.Precision.HIGHEST)


# aliases for working with pytrees
def _vdot_real_part(x, y):
    """Vector dot-product guaranteed to have a real valued result despite
    possibly complex input. Thus neglects the real-imaginary cross-terms.
    The result is a real float.
    """
    # all our uses of vdot() in CG are for computing an operator of the form
    #  z^H M z
    #  where M is positive definite and Hermitian, so the result is
    # real valued:
    # https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Definitions_for_complex_matrices
    result = _vdot(x.real, y.real)
    if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
        result += _vdot(x.imag, y.imag)
    return result


def _vdot_real_tree(x, y):
    return sum(tree_leaves(tree_map(_vdot_real_part, x, y)))


def _mul(scalar, tree):
    return tree_map(partial(operator.mul, scalar), tree)


_add = partial(tree_map, operator.add)
_sub = partial(tree_map, operator.sub)


@Partial
def _identity(x):
    return x


def _normalize_matvec(f):
    """Normalize an argument for computing matrix-vector products."""
    if callable(f):
        return f
    elif isinstance(f, (np.ndarray, jax.Array)):
        if f.ndim != 2 or f.shape[0] != f.shape[1]:
            raise ValueError(
                f"linear operator must be a square matrix, but has shape: {f.shape}"
            )
        return partial(_dot, f)
    elif hasattr(f, "__matmul__"):
        if hasattr(f, "shape") and len(f.shape) != 2 or f.shape[0] != f.shape[1]:
            raise ValueError(
                f"linear operator must be a square matrix, but has shape: {f.shape}"
            )
        return partial(operator.matmul, f)
    else:
        raise TypeError(f"linear operator must be either a function or ndarray: {f}")


def _shapes(pytree):
    return map(jnp.shape, tree_leaves(pytree))


def _cg_solve(
    A, b, x0=None, cg_state=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity
):
    # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
    bs = _vdot_real_tree(b, b)
    atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma.real if M is _identity else _vdot_real_tree(r, r)
        return (rs > atol2) & (k < maxiter)

    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p)
        alpha = gamma / _vdot_real_tree(p, Ap).astype(dtype)
        x_ = _add(x, _mul(alpha, p))
        r_ = _sub(r, _mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = _vdot_real_tree(r_, z_).astype(dtype)
        beta_ = gamma_ / gamma
        p_ = _add(z_, _mul(beta_, p))
        return x_, r_, gamma_, p_, k + 1

    if cg_state is not None:
        x0, r0, gamma0, p0, k0 = cg_state
        dtype = jnp.result_type(*tree_leaves(p0))
    else:
        r0 = _sub(b, A(x0))
        p0 = z0 = M(r0)
        dtype = jnp.result_type(*tree_leaves(p0))
        gamma0 = _vdot_real_tree(r0, z0).astype(dtype)
        k0 = 0
    initial_value = (x0, r0, gamma0, p0, k0)

    x, r, gamma, p, k = lax.while_loop(cond_fun, body_fun, initial_value)

    return x, (x, r, gamma, p, k)


def _isolve(
    _isolve_solve,
    A,
    b,
    x0=None,
    *,
    cg_state=None,
    tol=1e-5,
    atol=0.0,
    maxiter=None,
    M=None,
    check_symmetric=False,
):
    if x0 is None:
        x0 = tree_map(jnp.zeros_like, b)

    b, x0 = device_put((b, x0))

    if maxiter is None:
        size = sum(bi.size for bi in tree_leaves(b))
        maxiter = 10 * size  # copied from scipy

    if M is None:
        M = _identity
    A = _normalize_matvec(A)
    M = _normalize_matvec(M)

    if tree_structure(x0) != tree_structure(b):
        raise ValueError(
            "x0 and b must have matching tree structure: "
            f"{tree_structure(x0)} vs {tree_structure(b)}"
        )

    # if _shapes(x0) != _shapes(b):
    #     raise ValueError(
    #         'arrays in x0 and b must have matching shapes: '
    #         f'{_shapes(x0)} vs {_shapes(b)}')

    isolve_solve = partial(
        _isolve_solve,
        x0=x0,
        cg_state=cg_state,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=M,
    )

    # real-valued positive-definite linear operators are symmetric
    def real_valued(x):
        return not issubclass(x.dtype.type, np.complexfloating)

    symmetric = all(map(real_valued, tree_leaves(b))) if check_symmetric else False
    x, cg_state = lax.custom_linear_solve(
        A,
        b,
        solve=isolve_solve,
        transpose_solve=isolve_solve,
        symmetric=symmetric,
        has_aux=True,
    )

    return x, cg_state


def custom_cg(
    A, b, x0=None, *, cg_state=None, tol=1e-5, atol=0.0, maxiter=None, M=None
):
    """Use Conjugate Gradient iteration to solve ``Ax = b``.

    The numerics of JAX's ``cg`` should exact match SciPy's ``cg`` (up to
    numerical precision), but note that the interface is slightly different: you
    need to supply the linear operator ``A`` as a function instead of a sparse
    matrix or ``LinearOperator``.

    Derivatives of ``cg`` are implemented via implicit differentiation with
    another ``cg`` solve, rather than by differentiating *through* the solver.
    They will be accurate only if both solves converge.

    Parameters
    ----------
    A: ndarray, function, or matmul-compatible object
        2D array or function that calculates the linear map (matrix-vector
        product) ``Ax`` when called like ``A(x)`` or ``A @ x``. ``A`` must represent
        a hermitian, positive definite matrix, and must return array(s) with the
        same structure and shape as its argument.
    b : array or tree of arrays
        Right hand side of the linear system representing a single vector. Can be
        stored as an array or Python container of array(s) with any shape.

    Returns
    -------
    x : array or tree of arrays
        The converged solution. Has the same structure as ``b``.
    info : None
        Placeholder for convergence information. In the future, JAX will report
        the number of iterations when convergence is not achieved, like SciPy.

    Other Parameters
    ----------------
    x0 : array or tree of arrays
        Starting guess for the solution. Must have the same structure as ``b``.
    tol, atol : float, optional
        Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
        We do not implement SciPy's "legacy" behavior, so JAX's tolerance will
        differ from SciPy unless you explicitly pass ``atol`` to SciPy's ``cg``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : ndarray, function, or matmul-compatible object
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.

    See also
    --------
    scipy.sparse.linalg.cg
    jax.lax.custom_linear_solve
    """
    return _isolve(
        _cg_solve,
        A=A,
        b=b,
        x0=x0,
        cg_state=cg_state,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        M=M,
        check_symmetric=True,
    )
