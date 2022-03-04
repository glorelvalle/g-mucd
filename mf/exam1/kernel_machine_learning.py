from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance


def linear_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
) -> np.ndarray:
    return X @ X_prime.T


def exponential_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    l: float
) -> np.ndarray:
    d = distance.cdist(X, X_prime, metric='minkowski', p=1)
    return A * np.exp(- d / l)


def rbf_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    ls: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_prime:
        Data matrix
    A:
        Output variance
    ls:
        Kernel lengthscale

    Returns
    -------
    kernel matrix

    Notes
    -------
    Alternative parametrization (e.g. en sklearn)
    gamma = 0.5 / ls**2

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_process_regression as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> X_prime = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> kernel_matrix = gp.rbf_kernel(X, X_prime, A, l)
    >>> print(kernel_matrix)
    """
    d = distance.cdist(X, X_prime, metric='euclidean')
    return A * np.exp(-0.5 * (d / ls)**2)


def kernel_pca(
    X: np.ndarray,
    X_test: np.ndarray,
    kernel: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_test:
        data matrix
    A:
        output variance
    ls:
        kernel lengthscale

    Returns
    -------
    X_test_hat:
        Projection of X_test on the principal components
    lambda_eigenvals:
        Eigenvalues of the centered kernel
    alpha_eigenvecs:
        Principal components. These are the eigenvectors
        of the centered kernel with the RKHS normalization

    Notes
    -------
    In the corresponding method of sklearn the eigenvectors
    are normalized in l2.

    """

    N = np.shape(X)[0]
    L = np.shape(X_test)[0]

    K = kernel(X, X)
    K_test = kernel(X_test, X)

    ones_N_N = np.ones((N, N))
    ones_L_N = np.ones((L, N))

    aux =  K @ ones_N_N
    K_centered = K - (aux + aux.T - ones_N_N @ aux / N) / N

    K_test_centered = K_test - (
        K_test @ ones_N_N + ones_L_N @ (K - aux / N)
    ) / N

    lambda_eigenvals, alpha_eigenvecs = np.linalg.eigh(K_centered)
    # np.linalg.eigh returns eigenvalues in ascending order

    # Eliminate eigenvalues that are small relative to the largest one
    TOL_REL = 1.0e-12
    index = (lambda_eigenvals / lambda_eigenvals[-1]) > TOL_REL
    lambda_eigenvals = lambda_eigenvals[index]
    alpha_eigenvecs = alpha_eigenvecs[:, index]

    # Order eigenvalues in descending order
    lambda_eigenvals = lambda_eigenvals[::-1]
    alpha_eigenvecs = alpha_eigenvecs[:, ::-1]
    RKHS_norm_factor = np.sqrt(lambda_eigenvals)
    alpha_eigenvecs = alpha_eigenvecs / RKHS_norm_factor

    # Projection on the principal components
    X_test_hat = K_test_centered @ alpha_eigenvecs

    return X_test_hat, lambda_eigenvals, alpha_eigenvecs
