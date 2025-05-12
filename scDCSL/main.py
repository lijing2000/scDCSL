# === main.py ===
from __future__ import print_function, division
import torch
import numpy as np
import scanpy as sc
import time
import tracemalloc
import h5py

from utils import *
import opt
from scDFCN import scDFCN
from train import train
from ltmg import build_trs_mask
from preprocess import normalize
from plot_ltmg_fit import plot_gene_gmm_fit

tracemalloc.start()

if __name__ == "__main__":
    start_time = time.time()
    setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt.args.name == "Macosko":
        data_mat = h5py.File(opt.args.data_file)
        x = np.array(data_mat['X'])
        y = np.array(data_mat['Y'])
        data_mat.close()
        adata = sc.AnnData(x)
        adata.obs['cell_labels'] = y
        adata_raw = read_dataset(adata, transpose=False, test_split=False, copy=True)
        adata = normalize(adata_raw, copy=True, highly_genes=opt.args.highly_genes,
                          size_factors=True, normalize_input=True, logtrans_input=True)
        x = adata.X
        y = adata.obs['cell_labels']
    else:
        file_path = f"data/{opt.args.name}.{opt.args.load_type}"
        dataset = load_data_origin_data(file_path, opt.args.load_type, scaling=True)
        x = dataset.x
        y = dataset.y
        adata = sc.AnnData(x)
        adata.obs['cell_labels'] = y

        x1 = dataset.x1
        y1 = dataset.y1
        adata_raw = sc.AnnData(x1)
        adata_raw.obs['cell_labels'] = y1

        adata = normalize(adata_raw, copy=True, highly_genes=opt.args.highly_genes,
                          size_factors=True, normalize_input=True, logtrans_input=True)

    # 构建图结构
    A, A_norm = load_graph(x)
    x_tensor = torch.tensor(x, dtype=torch.float32).to(opt.args.device)
    A_norm_tensor = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)

    # 构建 LTMG 转录状态标签矩阵
    print("Building TRS mask using LTMG...")
    trs_mask = build_trs_mask(x, max_components=4, min_cells=10)
    trs_mask_tensor = torch.tensor(trs_mask, dtype=torch.float32).to(opt.args.device)

    # 初始化模型
    model = scDFCN(n_node=x.shape[0]).to(opt.args.device)
    model.pretrain(LoadDataset(x_tensor), A_norm_tensor)

    # 训练模型
    adata_embed, cluster_labels, accs, nmis, aris, gammas = train(
        model, LoadDataset(x_tensor), x_tensor, y, A, A_norm_tensor,
        trs_mask=trs_mask_tensor, alpha=opt.args.ltmg_alpha)

    # 保存结果
    adata.obs['predicted_clusters'] = cluster_labels
    print("Best ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}".format(max(accs), nmis[np.argmax(accs)], aris[np.argmax(accs)]))
    print("Best Epoch:", np.argmax(accs))

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f}s")

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1e9:.3f} GB")
    print(f"Peak memory usage: {peak / 1e9:.3f} GB")


    gene_id = 10
    gene_expr = x[:, gene_id]  # 假设 x 是 numpy array
    plot_gene_gmm_fit(gene_expr, gene_name=f"Gene_{gene_id}")
