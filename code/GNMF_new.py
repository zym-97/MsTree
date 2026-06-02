import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph

from gnmf import GNMF
import nndsvda


def run_GNMF(
    X,
    locations,
    knn_k=200,
    n_components=None,
    l=1.0,
    tol=1e-3,
    symmetrize=True,
    return_sparse_A=False,
    X_orientation="genes_by_spots",
    verbose=False,
):
    """
    GNMF embedding callable from R via reticulate.

    Parameters
    ----------
    X : array-like
        Expression matrix.
        If X_orientation == "genes_by_spots": shape (n_genes, n_spots)
        If X_orientation == "spots_by_genes": shape (n_spots, n_genes)
    locations : array-like
        Spatial coordinates, shape (n_spots, 2) or (n_spots, 3)
    knn_k : int
        Number of neighbors in KNN graph for adjacency A.
    n_components : int or None
        Latent dimension for GNMF. If None, use int(sqrt(n_spots)).
    l : float
        Regularization parameter in GNMF implementation.
    tol : float
        Tolerance for GNMF convergence.
    symmetrize : bool
        Whether to symmetrize adjacency A via max(A, A.T).
    return_sparse_A : bool
        Whether to also return sparse adjacency A (and D, L) for debugging.
    X_orientation : str
        "genes_by_spots" or "spots_by_genes".
    verbose : bool
        Print debug info.

    Returns
    -------
    H : np.ndarray
        Embedding matrix with shape (n_components, n_spots) if X is genes_by_spots.
        If X is spots_by_genes, H will still be returned as (n_components, n_spots)
        for consistency with your downstream R code expecting dim x spots.
    """

    # Convert inputs to numpy arrays
    X = np.asarray(X)
    locations = np.asarray(locations)

    if X_orientation not in ("genes_by_spots", "spots_by_genes"):
        raise ValueError("X_orientation must be 'genes_by_spots' or 'spots_by_genes'.")

    # Standardize X to genes_by_spots for your current pipeline
    # (so n_spots is X.shape[1])
    if X_orientation == "spots_by_genes":
        # transpose to genes_by_spots
        X = X.T

    n_genes, n_spots = X.shape

    if verbose:
        print("X shape (genes_by_spots):", X.shape)
        print("locations shape:", locations.shape)

    if locations.shape[0] != n_spots:
        raise ValueError(
            f"locations rows ({locations.shape[0]}) must equal n_spots ({n_spots})."
        )

    # -------- 1) Build adjacency A via KNN graph --------
    # Ensure knn_k < n_spots to avoid sklearn error
    knn_k_eff = int(min(knn_k, max(1, n_spots - 1)))

    A = kneighbors_graph(locations, knn_k_eff, mode="connectivity", include_self=False)
    if symmetrize:
        # A is sparse; use maximum with transpose
        A = A.maximum(A.T)

    # Ensure CSR sparse
    A = A.tocsr()

    if verbose:
        sparsity = A.nnz / (A.shape[0] * A.shape[1])
        print(f"A sparsity: {sparsity:.6f} (nnz={A.nnz})")

    n = A.shape[0]

    # -------- 2) Degree matrix D and normalized Laplacian L --------
    degrees = np.array(A.sum(axis=1)).flatten()
    # avoid division by zero
    degrees_safe = np.where(degrees > 0, degrees, 1.0)

    D = sp.diags(degrees_safe)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees_safe))

    I = sp.eye(n, format="csr")
    L = I - (D_inv_sqrt @ A @ D_inv_sqrt)

    # -------- 3) Choose n_components --------
    if n_components is None:
        n_components = int(np.sqrt(n_spots))
        n_components = max(2, n_components)

    n_components = int(n_components)

    # -------- 4) Initialize W, H with NNDSVDA --------
    # Your nndsvda expects X (genes x spots) and returns W, H
    W, H = nndsvda.NNDSVDA(X, n_component=n_components)

    # -------- 5) GNMF --------
    myGNMF = GNMF(n_components=W.shape[1], l=l, tol=tol)
    W, H = myGNMF.fit_transform(X, W, H, L, A, D)

    # H shape should be (n_components, n_spots) in your setup
    if verbose:
        print("H shape:", H.shape)

    if return_sparse_A:
        return H, W, A, D, L
    return H


# Optional convenience wrapper to accept pandas DataFrame from R
def run_GNMF_df(df_expr, df_locations, **kwargs):
    """
    df_expr: pandas DataFrame (genes x spots) OR (spots x genes)
    df_locations: pandas DataFrame with x,y(,z), rows=spots
    """
    X = df_expr.values
    loc = df_locations.values
    return run_GNMF(X, loc, **kwargs)
