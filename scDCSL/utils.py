import torch
import opt
import random
import numpy as np
import h5py
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import scale, minmax_scale
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
# from sklearn import metrics
# from munkres import Munkres
from loadData import loadH5AnnData
import scanpy as sc
import anndata
from scipy.optimize import linear_sum_assignment


def setup():
    print("setting:")
    setup_seed(opt.args.seed)
    if opt.args.name == 'baron_mouse':
        opt.args.n_clusters = 13
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.01
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'deng':
        opt.args.n_clusters = 6
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5


    elif opt.args.name == 'zeisel':
        opt.args.n_clusters = 9
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        # opt.args.lr = 1e-5

    elif opt.args.name == 'Quake_10x_Limb_Muscle':
        opt.args.n_clusters = 6
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'Quake_Smart-seq2_Limb_Muscle':
        opt.args.n_clusters=6
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-4

    elif opt.args.name == 'Romanov':
        opt.args.n_clusters = 7
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5
    elif opt.args.name == 'romanov':
        opt.args.n_clusters = 7
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'darmanis':
        opt.args.n_clusters = 8
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'biase':
        opt.args.n_clusters = 8
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'Muraro':
        opt.args.n_clusters = 9
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-4

    elif opt.args.name == 'Young':
        opt.args.n_clusters = 11
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-4

    elif opt.args.name == 'Macosko':
        opt.args.n_clusters = 39
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'Shekhar':
        opt.args.n_clusters = 12
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'Pollen':
        opt.args.n_clusters = 11
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'Quake_Smart-seq2_Trachea':
        opt.args.n_clusters = 4
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5
    elif opt.args.name == 'Quake_Smart-seq2_Lung':
        opt.args.n_clusters = 11
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5
    elif opt.args.name == 'Quake_Smart-seq2_Heart':
        opt.args.n_clusters = 8
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-4

    elif opt.args.name == 'Quake_10x_Bladder':
        opt.args.n_clusters = 4
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 100
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'simulation':
        opt.args.n_clusters = 10
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'sim1':
        opt.args.n_clusters = 15
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == '1M':
        opt.args.n_clusters = 35
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5

    elif opt.args.name == 'GSE194122s1rna':
        opt.args.n_clusters = 10
        opt.args.n_input = opt.args.highly_genes
        opt.args.lambda_value = 0.1
        opt.args.gamma_value = 0.1
        opt.args.lr = 1e-5



    else:
        print("error!")
        print("please add the new dataset's parameters")
        print("------------------------------")
        print("dataset       : ")
        print("device        : ")
        print("random seed   : ")
        print("clusters      : ")
        print("alpha value   : ")
        print("lambda value  : ")
        print("gamma value   : ")
        print("learning rate : ")
        print("------------------------------")
        exit(0)

    opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("random seed   : {}".format(opt.args.seed))
    print("clusters      : {}".format(opt.args.n_clusters))
    print("lambda value  : {}".format(opt.args.lambda_value))
    print("gamma value   : {:.0e}".format(opt.args.gamma_value))
    print("learning rate : {:.0e}".format(opt.args.lr))
    print("------------------------------")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class load_data_origin_data(Dataset):
    def __init__(self, dataset, load_type, take_log=False, scaling=False):
        def load_txt():
            self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
            self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

        def load_h5():
            adata,adata_raw = loadH5AnnData('{}'.format(dataset))
            self.x = adata.X
            self.y = adata.obs['cell_labels'].values
            self.x1 = adata_raw.X
            self.y1 = adata_raw.obs['cell_labels'].values

        def load_csv():
            pre_process_paras = {'take_log': take_log, 'scaling': scaling}
            self.pre_process_paras = pre_process_paras
            print(pre_process_paras)
            dataset_list, dataset_list_raw = pre_processing_single(dataset, pre_process_paras, type='csv')
            self.x = dataset_list[0]['gene_exp'].transpose().astype(np.float32)
            self.x1 = dataset_list_raw[0]['gene_exp'].transpose().astype(np.float32)
            # self.x = x.T
            print(self.x.shape)
            self.y = dataset_list[0]['cell_labels'].astype(np.int32)
            self.y1 = dataset_list_raw[0]['cell_labels'].astype(np.int32)
            self.cluster_label = dataset_list[0]['cluster_labels'].astype(np.int32)
        def load_h5ad():
            adata = anndata.read_h5ad('{}'.format(dataset))
            self.x = adata.X
            self.y = adata.obs['name'].values

        if load_type == "csv":
            load_csv()
        elif load_type == "h5":
            load_h5()
        elif load_type == "txt":
            load_txt()
        elif load_type == "h5ad":
            load_h5ad()

def read_csv(filename, take_log):
    """ Read TPM data of a dataset saved in csv format
    Format of the csv:
    first row: sample labels
    second row: cell labels
    third row: cluster labels from Seurat
    first column: gene symbols
    Args:
        filename: name of the csv file
        take_log: whether do log-transformation on input data
    Returns:
        dataset: a dict with keys 'gene_exp', 'gene_sym', 'sample_labels', 'cell_labels', 'cluster_labels'
    """
    dataset = {}
    df = pd.read_csv(filename, header=None)
    dat = df[df.columns[1:]].values
    dataset['sample_labels'] = dat[0, :].astype(int)
    dataset['cell_labels'] = dat[1, :].astype(int)
    dataset['cluster_labels'] = dat[2, :].astype(int)
    gene_sym = df[df.columns[0]].tolist()[3:]
    gene_exp = dat[3:, :]


    if take_log:
            gene_exp = np.log2(gene_exp + 1)
    dataset['gene_exp'] = gene_exp
    dataset['gene_sym'] = gene_sym
    return dataset

def read_txt(filename, take_log):
    dataset = {}
    # df = pd.read_table(filename, header=None)
    # dat = df[df.columns[1:]].values
    # dataset['cell_labels'] = dat[8, 1:]
    # gene_sym = df[df.columns[0]].tolist()[11:]
    # gene_exp = dat[11:, 1:].astype(np.float32)
    # if take_log:
    #     gene_exp = np.log2(gene_exp + 1)
    # dataset['gene_exp'] = gene_exp
    # dataset['gene_sym'] = gene_sym
    # dataset['cell_labels'] = convert_strclass_to_numclass(dataset['cell_labels'])
    #
    # save_csv(gene_exp, gene_sym,  dataset['cell_labels'])

    return dataset

def pre_processing_single(dataset_file_list, pre_process_paras, type=opt.args.load_type):
    """ pre-processing of multiple datasets
    Args:
        dataset_file_list: list of filenames of datasets
        pre_process_paras: dict, parameters for pre-processing
    Returns:
        dataset_list: list of datasets
    """
    # parameters
    take_log = pre_process_paras['take_log']
    scaling = pre_process_paras['scaling']
    dataset_list = []
    dataset_list_raw = []
    data_file = dataset_file_list

    if type == 'csv':
        dataset = read_csv(data_file, take_log)
    elif type == 'txt':
        dataset = read_txt(data_file, take_log)


    dataset['gene_exp'] = dataset['gene_exp'].astype(np.float)
    dataset_list_raw.append(dataset)

    if scaling:  # scale to [0,1]
        minmax_scale(dataset['gene_exp'], feature_range=(0, 1), axis=1, copy=False)


    dataset_list.append(dataset)
    return dataset_list, dataset_list_raw

def select(dataset):
    data = dataset.x
    # 识别高度可变基因，并进行过滤：
    sc.pp.highly_variable_genes(data, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(data)
    # 保存原始数据后再把data变成只有高变基因
    data.raw = data
    # 过滤
    ##注意切片data[obs:var]
    data = data[:, data.var['highly_variable']]
    return dataset

def load_graph(count,k=opt.args.k, pca=50, mode="connectivity"):
    
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n

def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10

# def load_graph(dataset, k=None, n=10, label=None):
#     import os
#     graph_path = os.getcwd()
#     if k:
#         path = graph_path + '/{}{}_graph.txt'.format(dataset, k)
#     else:
#         path =graph_path +  '/{}_graph.txt'.format(dataset)
#
#
#     idx = np.array([i for i in range(n)], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt(path, dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(n, n), dtype=np.float32)
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#     adj = adj + sp.eye(adj.shape[0])
#     adj = norm_adj(adj)
#     # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import os
    print("delete file: ", path)
    os.remove(path)

    return adj


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D

def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a


def torch_to_numpy(t):
    """
    torch tensor to numpy array
    :param t: the torch tensor
    :return: numpy array
    """
    return t.numpy()

class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

def pretrain(model, dataset, A_norm):
    print("Pretraining...")
    model.pretrain(LoadDataset(dataset.x), A_norm)




def load_pretrain_parameter(model):
    """
    load pretrained parameters
    Args:
        model: Dual Correlation Reduction Network
    Returns: model
    """
    pretrained_dict = torch.load('model_pretrain/{}_pretrain.pkl'.format(opt.args.name), map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def model_init(model,dataset, x, y, A_norm):
    """
    load the pre-train model and calculate similarity and cluster centers
    Args:
        model: Dual Correlation Reduction Network
        X: input feature matrix
        y: input label
        A_norm: normalized adj
    Returns: embedding similarity matrix
    """
    # load pre-train model
    model = load_pretrain_parameter(model)
    # with torch.no_grad():
    #     x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde = model(x, A_norm)
    # kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    # cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
    # model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(opt.args.device)
    # eva(y, cluster_id, 'Initialization')






def eva(y_true, y_pred, epoch=0):
    acc= cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
    #       ', f1 {:.4f}'.format(f1))
    print('Epoch_{}'.format(epoch), ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
    return acc, nmi, ari

def cluster_acc(y_true, y_pred):
    # y_true = y_true - np.min(y_true)
    #
    # l1 = list(set(y_true))
    # numclass1 = len(l1)
    #
    # l2 = list(set(y_pred))
    # numclass2 = len(l2)
    #
    # ind = 0
    # if numclass1 != numclass2:
    #     for i in l1:
    #         if i in l2:
    #             pass
    #         else:
    #             y_pred[ind] = i
    #             ind += 1
    #
    # l2 = list(set(y_pred))
    # numclass2 = len(l2)
    #
    # if numclass1 != numclass2:
    #     print('error')
    #     return
    #
    # cost = np.zeros((numclass1, numclass2), dtype=int)
    # for i, c1 in enumerate(l1):
    #     mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
    #     for j, c2 in enumerate(l2):
    #         mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
    #         cost[i][j] = len(mps_d)
    #
    # m = Munkres()
    # cost = cost.__neg__().tolist()
    # indexes = m.compute(cost)
    #
    # new_predict = np.zeros(len(y_pred))
    # for i, c in enumerate(l1):
    #     c2 = l2[indexes[i][1]]
    #
    #     ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
    #     new_predict[ai] = c
    #
    # acc = metrics.accuracy_score(y_true, new_predict)
    # f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    # precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    # recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    # f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    # precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    # recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    # return acc, f1_macro
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def clustering(Z, y):
    """
    clustering based on embedding
    Args:
        Z: the input embedding
        y: the ground truth

    Returns: acc, nmi, ari, f1, clustering centers
    """
    model = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    cluster_id = model.fit_predict(Z.data.cpu().numpy())
    acc, nmi, ari= eva(y, cluster_id, show_details=opt.args.show_training_details)
    return acc, nmi, ari, model.cluster_centers_


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def read_dataset(adata, transpose=False, test_split=False, copy=False):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    # if adata.X.size < 50e6: # check if adata.X is integer only if array is small
    #     if sp.sparse.issparse(adata.X):
    #         assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
    #     else:
    #         assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose: adata = adata.transpose()
    from sklearn.model_selection import train_test_split
    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata




def save(cluster_lables):
    test = pd.DataFrame(columns=[opt.args.name], data=cluster_lables)
    test.to_csv('scDFCN.csv', index=False, sep=',')