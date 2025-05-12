# === train.py ===
import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from utils import eva
import opt
import scanpy as sc
from ltmg import build_trs_mask

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def train(model, dataset, x, y, A, A_norm, trs_mask=None, alpha=0.0):
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde, _ = model(x, A_norm)
    kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(opt.args.device)

    acc_results, nmi_results, ari_results, gamma_list = [], [], [], []
    y_pred_last = kmeans.labels_
    num = x.shape[0]

    for epoch in range(opt.args.epoch):
        x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde, gamma = model(x, A_norm)
        gamma_list.append(gamma)
        p = target_distribution(q.data)

        # AE loss with LTMG TRS
        if trs_mask is not None and alpha > 0:
            loss_ae = ((1 - alpha) * F.mse_loss(x_hat, x, reduction='none') +
                       alpha * F.mse_loss(x_hat, x, reduction='none') * trs_mask).mean()
        else:
            loss_ae = F.mse_loss(x_hat, x)

        loss_w = F.mse_loss(z_hat, torch.spmm(A_norm, x))
        loss_a = F.mse_loss(adj_hat, A_norm.to_dense())
        loss_igae = loss_w + opt.args.gamma_value * loss_a
        loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = loss_ae + loss_igae + opt.args.lambda_value * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        labels = KMeans(n_clusters=opt.args.n_clusters, n_init=20).fit_predict(z_tilde.data.cpu().numpy())
        acc, nmi, ari = eva(y, labels, epoch)
        acc_results.append(acc)
        nmi_results.append(nmi)
        ari_results.append(ari)

        delta_label = np.sum(labels != y_pred_last).astype(np.float32) / num
        if epoch > 0 and delta_label < 1e-3:
            print('Reach tolerance threshold. Stopping training.')
            break
        y_pred_last = labels

    adata = sc.AnnData(z_tilde.cpu().detach().numpy())
    adata.obs['cell_labels'] = labels
    return adata, labels, acc_results, nmi_results, ari_results, gamma_list
