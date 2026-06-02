from scipy.spatial.distance import cdist



import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.neighbors import kneighbors_graph
from gnmf import GNMF
import nndsvda
import scipy.sparse as sp
from cal_A import choose_n_components



# 1. Input X
df = pd.read_csv("/data/simulation/gene10000/sp_sim_count_ieffect5.csv")
print(df.head())
X = df.iloc[:, 1:].values


# 2. Calculate the adjacency matrix A
df1 = pd.read_csv("/data/simulation/gene10000/sp_sim_location.csv")
print(df1.head())
locations = df1.iloc[:, 1:3].values

k = 20  
A = kneighbors_graph(locations, k, mode='connectivity').toarray()
A = np.maximum(A,A.T) 
A = sp.csr_matrix(A) 
sparsity = A.nnz / (A.shape[0] * A.shape[1])
n = A.shape[0]


# 3. Calculate the degree matrix D and the Laplace matrix L
degrees = np.array(A.sum(axis=1)).flatten()
D = sp.diags(degrees)  
D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))

I = sp.eye(n)  # 单位矩阵 I
L = I - D_inv_sqrt @ A @ D_inv_sqrt


# 4. Initialize W, H
s = X.shape[1]  
k = int(np.sqrt(s))  

W, H = nndsvda.NNDSVDA(X, n_component=k)


# 5. Run GNMF
myGNMF = GNMF(n_components=W.shape[1], l=1.0, tol = 1e-3)
W, H = myGNMF.fit_transform(X, W, H, L, A, D)

df_H = pd.DataFrame(H)
df_H.columns = df.columns[1:]

# 7. Save
df_H.to_csv("/data/simulation/gene10000/sp_sim_ieffect5_embedding.csv", index=True)

