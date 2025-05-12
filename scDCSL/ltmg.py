# ltmg.py
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def fit_ltgm_per_gene(expr, max_components=4, min_cells=10):
    """
    拟合某基因的左截断高斯混合模型，返回该基因在每个细胞中的转录调控状态（TRS）标签。

    参数:
        expr: shape=(n_cells,) 原始表达向量
        max_components: 最多拟合的高斯分布数
        min_cells: 至少非零表达细胞数，太少则返回全0标签

    返回:
        trs_labels: shape=(n_cells,) 每个细胞的离散状态编号
    """
    expr = np.asarray(expr)

    # 若表达细胞数太少，跳过该基因
    if np.count_nonzero(expr) < min_cells:
        return np.zeros_like(expr, dtype=int)

    valid_expr = expr[expr > 0].reshape(-1, 1)
    max_k = min(max_components, valid_expr.shape[0])

    if max_k < 1:
        return np.zeros_like(expr, dtype=int)

    lowest_bic = np.inf
    best_gmm = None

    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='full')
            gmm.fit(valid_expr)
            bic = gmm.bic(valid_expr)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
        except:
            continue

    if best_gmm is None:
        return np.zeros_like(expr, dtype=int)

    full_expr = expr.reshape(-1, 1)
    posterior = best_gmm.predict_proba(full_expr)  # shape: (n_cells, k)
    trs_labels = posterior.argmax(axis=1)  # 最大后验分量编号

    return trs_labels


def build_trs_mask(expression_matrix, max_components=4, min_cells=10, normalize=True):
    """
    对整个表达矩阵进行 LTMG 拟合，生成 TRS 标签矩阵。

    参数:
        expression_matrix: shape=(n_cells, n_genes)
        max_components: 每个基因最多拟合多少个分量
        min_cells: 拟合前所需的最少非零表达细胞
        normalize: 是否将标签归一化为0~1（作为soft权重）

    返回:
        trs_mask: shape=(n_cells, n_genes)
    """
    n_cells, n_genes = expression_matrix.shape
    trs_mask = np.zeros((n_cells, n_genes), dtype=int)

    for j in tqdm(range(n_genes)):
        gene_expr = expression_matrix[:, j]
        trs_mask[:, j] = fit_ltgm_per_gene(gene_expr,
                                           max_components=max_components,
                                           min_cells=min_cells)

    if normalize:
        trs_mask = trs_mask.astype(float)
        for j in range(n_genes):
            max_val = np.max(trs_mask[:, j])
            if max_val > 0:
                trs_mask[:, j] /= max_val

    return trs_mask
