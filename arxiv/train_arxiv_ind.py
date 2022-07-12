
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.sparse as sp
from copy import deepcopy
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import time
from sklearn import metrics
import random
import json
import sys
import os
import scipy.sparse as sp

dataset = 'arxiv'

def set_gpu(gpus):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Check if tensorflow is running on a gpu
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
set_gpu("1")




labels = np.load("./Datasets/ogbn-arxiv_undirected/labels.npy")
train_ix = np.load("./Datasets/ogbn-arxiv_undirected/TrainSortedIndex.npy")
valid_ix = np.load("./Datasets/ogbn-arxiv_undirected/ValidSortedIndex.npy")
test_ix = np.load("./Datasets/ogbn-arxiv_undirected/TestSortedIndex.npy")

feat_data = np.load("./Datasets/ogbn-arxiv_undirected/feats.npy")
indices = {}
indices['train'] = train_ix
indices['valid'] = valid_ix
indices['test'] = test_ix
train_adj = sp.load_npz("./Datasets/ogbn-arxiv_undirected/adj_train.npz")
full_adj = sp.load_npz("./Datasets/ogbn-arxiv_undirected/adj_full.npz")


def normalize(adj):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
s.fit(feat_data[train_ix])
feat_data= s.transform(feat_data)

train_feat = np.zeros(feat_data.shape)
train_feat[train_ix] = feat_data[train_ix]

train_adj = train_adj + train_adj.T.multiply(train_adj.T > train_adj) - train_adj.multiply(train_adj.T > train_adj)
full_adj = full_adj + full_adj.T.multiply(full_adj.T > full_adj) - full_adj.multiply(full_adj.T > full_adj)

import sys
from utils import *

adj_normalize = 'GCN'
if adj_normalize == "GCN":
    v = np.zeros(train_adj.shape[0])
    v[train_ix] = 1
    t=sp.diags(v)
    train_laplacian = normalize(train_adj + t)
    full_laplacian = normalize(full_adj + sp.eye(full_adj.shape[0]))
elif adj_normalize =="SAINT":
    v = np.zeros(train_adj.shape[0])
    v[train_ix] = 1
    t=sp.diags(v)
    train_laplacian = saint_normalize(train_adj + t)
    full_laplacian = saint_normalize(full_adj + sp.eye(full_adj.shape[0]))


def get_sparse_adj(A):
    from scipy.sparse import csr_matrix
    return csr_matrix((A[1], (A[0][:, 0], A[0][:, 1])), A[2])

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return (indices, coo.data, coo.shape)

def get_tensor_values(adj, num_adj):
    adj = adj.astype(np.float32)
    num_nodes = train_laplacian.shape[0]
    ix = [(i*num_nodes//num_adj, (i+1) * num_nodes//num_adj) for i in range(num_adj)]
    adjs = [adj[ix[i][0]:ix[i][1], :] for i in range(num_adj)]
    sparse_vals = {ix: tf.SparseTensorValue(*convert_sparse_matrix_to_sparse_tensor(adjs[ix]))                   for ix in range(num_adj)}
    return sparse_vals

def MyLayerNorm(x):
    mean, variance = tf.nn.moments(x, axes=[1], keep_dims=True)
    
    offset = zeros([1, x.get_shape()[1]], name='offset')
    scale  = ones([1, x.get_shape()[1]], name='scale')
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-9), offset, scale

def MyLayerNorm1(x):
    mean, variance = tf.nn.moments(x, axes=[0], keep_dims=True)
    
    offset = tf.get_variable(name='testoffset', shape=[1, x.get_shape()[1]], dtype=tf.float32, 
                           initializer=tf.zeros_initializer(), trainable=False)
    scale  = tf.get_variable(name='testscale', shape=[1, x.get_shape()[1]], dtype=tf.float32, 
                           initializer=tf.ones_initializer(), trainable=False)
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-9), offset, scale


def zeros(shape, name=None):
    """All zeros."""
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32, 
                           initializer=tf.zeros_initializer())


def ones(shape, name=None):
    """All ones."""
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32, 
                           initializer=tf.ones_initializer())

vals = get_tensor_values(train_laplacian, num_adj = 1)

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

from copy import deepcopy

mask = [False for i in range((train_laplacian.shape[0]))]
train_mask = deepcopy(mask)
valid_mask = deepcopy(mask)
test_mask = deepcopy(mask)

train_labels = np.zeros((train_laplacian.shape[0], 40))
train_labels[train_ix] = labels[train_ix]
train_labels = train_labels.astype(np.float32)

valid_labels = np.zeros((train_laplacian.shape[0], 40))
valid_labels[valid_ix] = labels[valid_ix]
valid_labels = valid_labels.astype(np.float32)

test_labels = np.zeros((train_laplacian.shape[0], 40))
test_labels[test_ix] = labels[test_ix].astype(np.float32)
test_labels = test_labels.astype(np.float32)



for elem in train_ix:
    train_mask[elem] = True
for elem in valid_ix:
    valid_mask[elem] = True
for elem in test_ix:
    test_mask[elem] = True

label_ph = tf.placeholder(tf.float32, shape=[None, None])
mask_ph = tf.placeholder(tf.float32, shape=[None, None])


tf.reset_default_graph()

Q = glorot([128, 256], name = "R")
R = glorot([256, 256], name = "R")
w = glorot([256, 40], name = "w")

b1 = tf.get_variable(name="b1", dtype=tf.float32, shape=[256,], initializer=tf.zeros_initializer())
b2 = tf.get_variable(name="b2", dtype=tf.float32, shape=[256,], initializer=tf.zeros_initializer())
b3 = tf.get_variable(name="b3", dtype=tf.float32, shape=[40,], initializer=tf.zeros_initializer())


X1 = tf.Variable(np.zeros((full_laplacian.shape[0], 128)).astype(np.float32), trainable = False)
X2 = tf.Variable(np.zeros((full_laplacian.shape[0], 256)).astype(np.float32), trainable = False)
X3 = tf.Variable(np.zeros((full_laplacian.shape[0], 256)).astype(np.float32), trainable = False)
alpha_3_var = tf.Variable(np.zeros((full_laplacian.shape[0], 256)).astype(np.float32), trainable = False)
alpha_2_var = tf.Variable(np.zeros((full_laplacian.shape[0], 256)).astype(np.float32), trainable = False)

AX3 = tf.concat([tf.sparse_tensor_dense_matmul(vals[i], X3) for i in range(len(vals))], axis = 0)
preds = tf.matmul(AX3, w) +b3
train_loss = masked_softmax_cross_entropy(preds, train_labels, train_mask)

AX2 = tf.concat([tf.sparse_tensor_dense_matmul(vals[i], X2) for i in range(len(vals))], axis = 0)
AX2R = tf.matmul(AX2, R) +b2
AX2R, L2_O, L2_S = MyLayerNorm(AX2R)
X3_to_assign = tf.nn.relu(AX2R)
X3_assign_op = X3.assign(X3_to_assign)

with tf.variable_scope("Layer2"):
    AX1 = tf.concat([tf.sparse_tensor_dense_matmul(vals[i], X1) for i in range(len(vals))], axis = 0)
    AX1Q = tf.matmul(AX1, Q) +b1
    AX1Q, L1_O, L1_S = MyLayerNorm(AX1Q)
    X2_to_assign = tf.nn.relu(AX1Q)
    X2_assign_op = X2.assign(X2_to_assign)


alpha_3 = tf.gradients(train_loss, X3)[0]
alpha_3_assign_op = alpha_3_var.assign(alpha_3)

alpha_ix = tf.placeholder(tf.int32, shape = [None])
alpha_3_to_use = tf.gather(alpha_3_var, alpha_ix)

alpha_2 = tf.gradients(alpha_3_var * X3_to_assign, X2)[0]
alpha_2_assign_op = alpha_2_var.assign(alpha_2)
alpha_2_to_use = tf.gather(alpha_2_var, alpha_ix)

drop_ph = tf.placeholder(tf.float32)


ix1 = tf.placeholder(tf.int32, shape = [None])
ix2 = tf.placeholder(tf.int32, shape = [None])
A_ph = tf.sparse_placeholder(tf.float32)
AX1_0 = tf.sparse_tensor_dense_matmul(A_ph, tf.gather(X1, ix1))
AX1_f = AX1_0
AX1Q_f = tf.matmul(AX1_f, Q) +b1
mean, variance = tf.nn.moments(AX1Q_f, axes=[1], keep_dims=True)
AX1Q_f = tf.nn.batch_normalization(AX1Q_f, mean, variance, L1_O, L1_S, 1e-9)
X2_f = tf.nn.relu(AX1Q_f)
X2_f = tf.nn.dropout(X2_f, rate = drop_ph)

ix3 = tf.placeholder(tf.int32, shape = [None])
ix4 = tf.placeholder(tf.int32, shape = [None])
A1_ph = tf.sparse_placeholder(tf.float32)
AX2_0 = tf.sparse_tensor_dense_matmul(A1_ph, tf.gather(X2, ix3))
AX2_f = AX2_0
AX2R_f = tf.matmul(AX2_f, R) + b2
mean, variance = tf.nn.moments(AX2R_f, axes=[1], keep_dims=True)
AX2R_f = tf.nn.batch_normalization(AX2R_f, mean, variance, L2_O, L2_S, 1e-9)
X3_f = tf.nn.relu(AX2R_f)
X3_f = tf.nn.dropout(X3_f, rate = drop_ph)

ix5 = tf.placeholder(tf.int32, shape = [None])
ix6 = tf.placeholder(tf.int32, shape = [None])
A2_ph = tf.sparse_placeholder(tf.float32)
AX3_0 = tf.sparse_tensor_dense_matmul(A2_ph, tf.gather(X3, ix5))
AX3_f = AX3_0

preds_mb = tf.matmul(AX3_f, w) +b3

label_ph = tf.placeholder(tf.int32)
loss_mb = tf.nn.softmax_cross_entropy_with_logits(logits=preds_mb, labels = label_ph)

grad_w = tf.gradients(loss_mb, [w,b3])
grad_R = tf.gradients(X3_f * alpha_3_to_use, [R, L2_O, L2_S, b2])
grad_Q = tf.gradients(X2_f * alpha_2_to_use, [Q, L1_O, L1_S, b1])


lr_ph = tf.placeholder(tf.float32)
optimizer = {
    'R':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_R, [R, L2_O, L2_S, b2]))),
    'Q':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_Q, [Q, L1_O, L1_S, b1]))),
    'w':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_w, [w, b3])))
}

def mini_batch_exact(batch_nodes, adj_matrix, depth = 1):
    sampled_nodes = []
    previous_nodes = batch_nodes
    adjs = []
    for d in range(depth):
        U = adj_matrix[previous_nodes, :]
        after_nodes = []
        for U_row in U:
            indices = U_row.indices
            after_nodes.append(indices)
        after_nodes = np.unique(np.concatenate(after_nodes))
        after_nodes = np.concatenate(
            [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
        adj = U[:, after_nodes]
        adjs += [adj]
        sampled_nodes.append(previous_nodes)
        previous_nodes = after_nodes
    adjs.reverse()
    sampled_nodes.reverse()
    return adjs, previous_nodes, sampled_nodes

def get_sampled_adjs_1(batch, laplacian_matrix, samp_num_list, depth = 1):
    adjs, prev_nodes, sampled_nodes = mini_batch_exact(batch,
                                                 adj_matrix = laplacian_matrix,
                                                 depth = depth)
    adjs = [sparse_to_tuple(v) for v in adjs]
    return ((adjs, prev_nodes, sampled_nodes), batch)

train_batches = []
train_index = np.random.permutation(np.arange(len(train_ix)))

ix = 0
for _ in range(0, len(train_ix), 10000):
    train_batches.append(train_ix[train_index[ix: min(ix + 10000, len(train_ix))]]) 
    ix += 10000

arg_list = [(batch, train_laplacian, [10]) for _, batch in enumerate(train_batches)]

import multiprocessing
with multiprocessing.Pool(processes=30) as pool:
    results = pool.starmap(get_sampled_adjs_1,arg_list)
train_data_batch = {
    ix: ((results[ix][0][0], results[ix][0][1], results[ix][0][2]), results[ix][1])
for ix, batch in enumerate(train_batches)
}


from sklearn import metrics
def calc_f1( y_true, y_pred,is_sigmoid):
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        return metrics.f1_score(y_true, y_pred, average="micro")
    else:
        y_pred_met = deepcopy(y_pred)
        y_pred_met[y_pred_met > 0] = 1
        y_pred_met[y_pred_met <= 0] = 0
        return metrics.f1_score(y_true, y_pred_met, average="micro"), metrics.f1_score(y_true, y_pred_met, average="macro")
    
    
train_feat = np.zeros((train_laplacian.shape[0], 128))
train_feat[train_ix] = feat_data[train_ix]


vals_f = get_tensor_values(full_laplacian, num_adj = 1)

X0_full = np.concatenate([feat_data], axis = 1).astype(np.float32)
X2_full = tf.concat([tf.sparse_tensor_dense_matmul(vals_f[i], X0_full) for i in range(len(vals_f))], axis = 0)
X2_full = tf.matmul(X2_full, Q) +b1
mean, variance = tf.nn.moments(X2_full, axes=[1], keep_dims=True)
X2_full = tf.nn.batch_normalization(X2_full, mean, variance, L1_O, L1_S, 1e-9)
X2_full = tf.nn.relu(X2_full)

X3_full = tf.concat([tf.sparse_tensor_dense_matmul(vals_f[i], X2_full) for i in range(len(vals_f))], axis = 0)
X3_full = tf.matmul(X3_full, R) +b2
mean, variance = tf.nn.moments(X3_full, axes=[1], keep_dims=True)
X3_full = tf.nn.batch_normalization(X3_full, mean, variance, L2_O, L2_S, 1e-9)

X3_full = tf.nn.relu(X3_full)


X4_full = tf.concat([tf.sparse_tensor_dense_matmul(vals_f[i], X3_full) for i in range(len(vals_f))], axis = 0)
preds_f = tf.matmul(X4_full, w) +b3


init = tf.global_variables_initializer()

sess = tf.Session()

X1_ass_op = X1.assign(np.concatenate([train_feat], axis = 1).astype(np.float32))

sess.run(init)



lr = {'w':1e-3, 'R':1e-3, 'Q':1e-3}
lr_decay_rate = 0.95

for run in range(5):

    lr = {'w':1e-3, 'R':1e-3, 'Q':1e-3}
    lr_decay_rate = 0.95

    drop_rate = 0.70

    sess.run(init)
    _ = sess.run(X1_ass_op)
    _ = sess.run(X2_assign_op)
    _ = sess.run(X3_assign_op)
    for epoch in range(200):

        if(epoch==0):
            _ = sess.run(X1_ass_op)
            _ = sess.run(X2_assign_op)
            _ = sess.run(X3_assign_op)

        _  = sess.run(alpha_3_assign_op)
        _ = sess.run(alpha_2_assign_op)

        if epoch == 15:
            for key in lr:
                lr[key]=1e-4


        batch_index = np.arange(len(train_data_batch))
        order = np.random.choice(batch_index, len(train_data_batch), replace = False)


        for index, key in enumerate(order):
            (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], train_data_batch[key][1]
            sess.run(optimizer['Q'], {ix1:prev_nodes, 
                                      ix2:train_batch,
                                   A_ph:tf.SparseTensorValue(*adj_mats[0]),
                                   lr_ph:lr['R'], 
                                      alpha_ix: train_batch, drop_ph:drop_rate})

        _ = sess.run(X2_assign_op)

        for index, key in enumerate(order):
            (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], train_data_batch[key][1]
            sess.run(optimizer['R'], {ix3:prev_nodes, 
                                      ix4:train_batch,
                                   A1_ph:tf.SparseTensorValue(*adj_mats[0]),
                                   lr_ph:lr['R'], 
                                      alpha_ix: train_batch, drop_ph:drop_rate})

        _ = sess.run(X3_assign_op)

        for index, key in enumerate(order):
            (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], train_data_batch[key][1]
            sess.run(optimizer['w'], {ix5:prev_nodes, ix6:train_batch,
                                   A2_ph:tf.SparseTensorValue(*adj_mats[0]),
                                   lr_ph:lr['w'], label_ph:labels[train_batch], drop_ph:drop_rate})


        print(f"Time Taken for epoch: {epoch} is {end_epoch_time - start_epoch_time} seconds.")
        full_preds_ = sess.run(preds_f)

        valid_preds = full_preds_[valid_ix]
        test_preds = full_preds_[test_ix]
        valid_f1 = calc_f1(labels[valid_ix], valid_preds, False)
        test_f1 = calc_f1(labels[test_ix], test_preds, False)
        train_f1 = calc_f1(labels[train_ix], full_preds_[train_ix], False)
        loss_value = sess.run(train_loss)

        print("Train Loss:", loss_value)
        print("Train - Micro - F1", train_f1)
        print("Valid - Micro - F1", valid_f1)
        print("Test - Micro - F1", test_f1)
        print()


