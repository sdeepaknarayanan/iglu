import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import numpy as np
import scipy.sparse as sp

dataset = PygNodePropPredDataset(name='ogbn-proteins',
                                 transform=T.ToSparseTensor())
data = dataset[0]

# Move edge features to node features.
data.x = data.adj_t.mean(dim=1)
data.adj_t.set_value_(None)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train']



feat_data = data.x.numpy()

np.save("feat.npy", feat_data)
np.save("Labels.npy", data.y.numpy())

np.save("SortedTrainIndex.npy", split_idx['train'].numpy())
np.save("SortedValidIndex.npy", split_idx['valid'].numpy())
np.save("SortedTestIndex.npy", split_idx['test'].numpy())


row, col, val = data.adj_t.coo()
val = np.ones(len(row))
adj_mat = sp.coo_matrix((val, (row.numpy(), col.numpy())))
adj_mat_ = adj_mat.tocsr()


sp.save_npz("adj_full.npz", adj_mat_)

adj_train = adj_mat_[train_idx][:, train_idx]

sp.save_npz("adj_train.npz", adj_train)

