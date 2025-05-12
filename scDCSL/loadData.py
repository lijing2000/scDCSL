import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
import pandas as pd
from natsort import natsorted
from preprocess import prepro
from preprocess import normalize
import opt
def loadData(path):
    """ load miRNA-CTS dataset from config files """
    FILE = open(path)
    lines = FILE.readlines()
    FILE.close()

    sample_labels, cell_labels, cluster_labels, gene_labels = [], [], [], []
    set_idx = 0
    for l, line in enumerate(lines[0:3]):
        # print(l)
        tokens = line.strip().split(",")
        tokens = [int(s) for s in tokens[1:]]
        if l == 0:
            sample_labels = np.array(tokens)
        if l == 1:
            cell_labels = np.array(tokens)
        if l == 2:
            cluster_labels = np.array(tokens)
        # print(sample_labels)
    print(np.shape(sample_labels)[0])
    cluster_labels = cluster_labels - np.min(cluster_labels)
    matrix = np.zeros((len(lines) - 3, np.shape(sample_labels)[0]), np.float64)
    for l, line in enumerate(lines[3:]):
        tokens = line.strip().split(",")
        gene_labels.append(tokens[0])
        tokens = [np.float64(s) for s in tokens[1:]]
        matrix[l] = np.array(tokens)
    gene_labels = np.array(gene_labels)

    return sample_labels, cell_labels, cluster_labels, gene_labels, matrix


def loadAnnData(path):
    sample_labels, cell_labels, cluster_labels, gene_labels, matrix = loadData(path)
    # print("cell_label:", cell_labels)
    # print("sample_labels:", sample_labels)
    # print("cluster_labels:", cluster_labels)
    # print("gene_labels:", gene_labels)
    # print("matrix:", np.shape(matrix))
    # print(matrix)
    # 设置观测样本的数量
    n_obs = np.shape(sample_labels)[0]
    # obs用于保存观测量的信息
    obs = pd.DataFrame()
    obs["cell_labels"] = cell_labels
    obs["sample_labels"] = sample_labels



    # for i in range(n_obs):
    #    obs[cell_label[i]]= cell_label[i]
    # 设置特征名var_names
    var_names = gene_labels
    print("gene_labels:", gene_labels)
    print(var_names)

    # 特征数量
    n_vars = len(var_names)  # 234

    # 将特征定义到 Dataframe
    var = pd.DataFrame(index=var_names)
    var["gene_labels"] = gene_labels
    #print("000000")
    #print("var.head()", var.head())  # 现在var没有columns(列索引), 只有index(行索引)
    #print("111111")
    X = matrix.T

    # 显示在所有细胞中在每个单细胞中产生最高计数分数的基因
    adata = ad.AnnData(X, obs=obs, var=var, dtype='float')

    adata.obs['cluster_labels'] = pd.Categorical(
        values=cluster_labels.astype('U'),
        categories=natsorted(map(str, np.unique(cluster_labels))),
    )

    return adata




def loadH5AnnData(path):
    x, y = prepro(path)

    print("Cell number:", x.shape[0])
    print("Gene number", x.shape[1])
    x = np.ceil(x).astype(int)
    cluster_number = len(np.unique(y))
    print("Cluster number:", cluster_number)
    opt.args.n_clusters = cluster_number
    adata = sc.AnnData(x)
    adata.obs['cell_labels'] = y

    adata_nom = normalize(adata, copy=True, highly_genes=opt.args.highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    return adata_nom,adata