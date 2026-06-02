import numpy as np
from sklearn.utils.extmath import squared_norm
from math import sqrt


def NNDSVDA(X, eps=1e-8, n_component = 30):
    """
    Perform NNDSVDA (Non-negative Double Singular Value Decomposition with zeros
    filled with the average of X) initialization for Non-negative Matrix Factorization (NMF).

    Parameters:
    -----------
    X : ndarray of shape (M, N)
        The input data matrix to be factorized.
    eps : float, optional, default=1e-8
        A small constant to avoid numerical issues with very small values.

    Returns:
    --------
    W : ndarray of shape (M, n_components)
        The initialized non-negative matrix W.
    H : ndarray of shape (n_components, N)
        The initialized non-negative matrix H.
    """

    # Calculate the number of components (rank of factorization)
    #n_components = int(np.ceil(np.sqrt(X.shape[1])))
    n_components = n_component
    if n_components > X.shape[0]:
        n_components = X.shape[0]

    # Perform Singular Value Decomposition (SVD) on matrix X
    u, s, v = np.linalg.svd(X, full_matrices=False)
    U = u[:, :n_components]  # Extract the first n_components of U
    V = v[:n_components, :]  # Extract the first n_components of V
    S = s[:n_components]     # Extract the first n_components singular values

    # Initialize W and H as zero matrices of appropriate shapes
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # Initialize the first column of W and first row of H
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    # Loop through remaining components to initialize W and H
    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # Split vectors into positive and negative parts
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # Calculate norms of the positive and negative parts
        x_p_nrm, y_p_nrm = sqrt(squared_norm(x_p)), sqrt(squared_norm(y_p))
        x_n_nrm, y_n_nrm = sqrt(squared_norm(x_n)), sqrt(squared_norm(y_n))

        # Compute the magnitudes of the positive and negative parts
        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # Choose the larger magnitude for the update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        # Update W and H using the chosen vectors
        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    # Replace small values in W and H with the average of X to avoid numerical issues
    avg = np.mean(X)
    W[W < eps] = avg
    H[H < eps] = avg

    return W, H
