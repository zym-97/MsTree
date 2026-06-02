import numpy as np
from scipy.spatial.distance import cdist


def compute_gaussian_adjacency(X, sigma=1.0):
    """
    计算基于高斯核的邻接矩阵
    X: 坐标矩阵 (m, d)，m个点，每个点d维坐标
    sigma: 高斯核参数，控制相似度
    返回邻接矩阵 A (m, m)
    """
    pairwise_distances = cdist(X, X, metric='euclidean')  # 计算所有点对之间的欧式距离
    A = np.exp(-pairwise_distances ** 2 / (2 * sigma ** 2))  # 高斯核计算相似度
    np.fill_diagonal(A, 0)  # 自己到自己的邻接值设为0
    return A



def choose_n_components(X, threshold=0.9):
    """
    自动选择保留足够信息量所需的最小降维维度
    X: 输入矩阵（基因 × spot）
    threshold: 保留的能量比例（如0.9表示保留90%的信息）
    """
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    explained_variance = np.cumsum(s**2) / np.sum(s**2)

    # 找到第一个使得累计能量超过阈值的索引（+1 是维数）
    n_components = np.searchsorted(explained_variance, threshold) + 1

    return n_components, explained_variance
