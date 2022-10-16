import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import argparse
import os
import scipy.sparse as sp
from copy import deepcopy
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import time
import sys
from utils import *
import multiprocessing
from sklearn import metrics

def set_gpu(gpus):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Check if tensorflow is running on a gpu
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")


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
    sparse_vals = {ix: tf.SparseTensorValue(*convert_sparse_matrix_to_sparse_tensor(adjs[ix])) \
                  for ix in range(num_adj)}
    return sparse_vals


def MyLayerNorm(x):
    mean, variance = tf.nn.moments(x, axes=[1], keep_dims=True)
    
    offset = zeros([1, x.get_shape()[1]], name='offset')
    scale  = ones([1, x.get_shape()[1]], name='scale')
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-9), offset, scale

def MyLayerNorm1(x):
    mean, variance = tf.nn.moments(x, axes=[1], keep_dims=True)
    
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

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

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

parser = argparse.ArgumentParser(description='Parameters for running IGLU')
parser.add_argument('--batchsize', metavar='INT', default = 10000, dest='batch_size', type=int,
help = 'Number of Nodes per minibatch.')
parser.add_argument('--learning_rate', metavar='FLOAT', default = 1e-2, type = float, dest='lr', help = 'Learning Rate')
parser.add_argument('--gpuid', metavar='0|1|2|3',default = 1,  type = int, dest='gpuid', help = 'GPU ID')
parser.add_argument('--data_path', metavar='STRING',  type = str, dest='data_path', help = 'Path to the dataset')

args = parser.parse_args()
        
set_gpu(str(args.gpuid))

#Load Data
start_data_time = time.time()
labels = np.load(args.data_path+"/labels.npy")
train_ix = np.load(args.data_path+"/TrainSortedIndex.npy")
valid_ix = np.load(args.data_path+"/ValidSortedIndex.npy")
test_ix = np.load(args.data_path+"/TestSortedIndex.npy")
feat_data = np.load(args.data_path+"/feat_preprocessed.npy")
train_adj = sp.load_npz(args.data_path+"/train_adj.npz")
full_adj = sp.load_npz(args.data_path+"/full_adj.npz")
print("Time to load data:", time.time()-start_data_time)
indices = {}
indices['train'] = train_ix
indices['valid'] = valid_ix
indices['test'] = test_ix

adj_normalize = 'SAINT'

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

vals = get_tensor_values(train_laplacian, num_adj = 8)



mask = [False for i in range((train_laplacian.shape[0]))]
train_mask = deepcopy(mask)
valid_mask = deepcopy(mask)
test_mask = deepcopy(mask)

train_labels = np.zeros((train_laplacian.shape[0], 41))
train_labels[train_ix] = labels[train_ix]
train_labels = train_labels.astype(np.float32)

valid_labels = np.zeros((train_laplacian.shape[0], 41))
valid_labels[valid_ix] = labels[valid_ix]
valid_labels = valid_labels.astype(np.float32)

test_labels = np.zeros((train_laplacian.shape[0], 41))
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

R = glorot([1204, 128], name = "R")
w = glorot([256, 41], name = "w")

train_feat = np.zeros((train_laplacian.shape[0], feat_data.shape[1]))
train_feat[train_ix] = feat_data[train_ix]

AX0 = np.concatenate([train_laplacian.dot(train_feat), train_feat], axis=1)

AX2 = tf.Variable(AX0.astype(np.float32), trainable = False)
X3 = tf.Variable(np.zeros((full_laplacian.shape[0], 128)).astype(np.float32), trainable = False)
alpha_3_var = tf.Variable(np.zeros((full_laplacian.shape[0], 128)).astype(np.float32), trainable = False)

AX3 = tf.concat([tf.concat([tf.sparse_tensor_dense_matmul(vals[i], X3) for i in range(len(vals))], axis = 0),
        X3], axis=1)
preds = tf.matmul(AX3, w)
train_loss = masked_softmax_cross_entropy(preds, train_labels, train_mask)


AX2R = tf.matmul(AX2, R)
AX2R, L2_O, L2_S = MyLayerNorm(AX2R)
X3_to_assign = tf.nn.relu(AX2R)
X3_assign_op = X3.assign(X3_to_assign)

alpha_3 = tf.gradients(train_loss, X3)[0]
alpha_3_assign_op = alpha_3_var.assign(alpha_3)

alpha_ix = tf.placeholder(tf.int32, shape = [None])
alpha_3_to_use = tf.gather(alpha_3_var, alpha_ix)

ix1 = tf.placeholder(tf.int32, shape = [None])
AX2_f = tf.gather(AX2, ix1)
AX2R_f = tf.matmul(AX2_f, R)
mean, variance = tf.nn.moments(AX2R_f, axes=[1], keep_dims=True)
AX2R_f = tf.nn.batch_normalization(AX2R_f, mean, variance, L2_O, L2_S, 1e-9)
X3_f = tf.nn.relu(AX2R_f)

ix3 = tf.placeholder(tf.int32, shape = [None])
ix4 = tf.placeholder(tf.int32, shape = [None])
A1_ph = tf.sparse_placeholder(tf.float32)
AX3_0 = tf.sparse_tensor_dense_matmul(A1_ph, tf.gather(X3, ix3))
X3_1 = tf.gather(X3, ix4)
AX3_f = tf.concat([AX3_0, X3_1], axis = 1)
preds_mb = tf.matmul(AX3_f, w)
label_ph = tf.placeholder(tf.int32)
loss_mb = tf.nn.softmax_cross_entropy_with_logits(logits=preds_mb, labels = label_ph)

grad_w = tf.convert_to_tensor(tf.gradients(loss_mb, w)[0])
grad_R = tf.gradients(X3_f * alpha_3_to_use, [R, L2_O, L2_S])

lr_ph = tf.placeholder(tf.float32)
optimizer = {
    'R':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_R, [R, L2_O, L2_S]))),
    'w':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients([(grad_w, w)])
}


train_batches = []
train_index = np.random.permutation(np.arange(len(train_ix)))

ix = 0
for _ in range(0, len(train_ix), args.batch_size):
    train_batches.append(train_ix[train_index[ix: min(ix + args.batch_size, len(train_ix))]]) 
    ix += args.batch_size

arg_list = [(batch, train_laplacian, [10]) for _, batch in enumerate(train_batches)]

import multiprocessing
with multiprocessing.Pool(processes=3) as pool:
    results = pool.starmap(get_sampled_adjs_1,arg_list)
train_data_batch = {
    ix: ((results[ix][0][0], results[ix][0][1], results[ix][0][2]), results[ix][1])
for ix, batch in enumerate(train_batches)
}
    
    
train_feat = np.zeros((232965, 602))
train_feat[train_ix] = feat_data[train_ix]

vals_f = get_tensor_values(full_laplacian, num_adj = 8)

X0_full = np.concatenate([feat_data], axis = 1).astype(np.float32)
X3_full = tf.concat([
    tf.concat([tf.sparse_tensor_dense_matmul(vals_f[i], X0_full) for i in range(len(vals_f))], axis = 0),
    X0_full], axis=1)
X3_full = tf.matmul(X3_full, R)
mean, variance = tf.nn.moments(X3_full, axes=[1], keep_dims=True)
X3_full = tf.nn.batch_normalization(X3_full, mean, variance, L2_O, L2_S, 1e-9)

X3_full = tf.nn.relu(X3_full)

X4_full = tf.concat([
    tf.concat([tf.sparse_tensor_dense_matmul(vals_f[i], X3_full) for i in range(len(vals_f))], axis = 0),
    X3_full], axis=1)
preds_f = tf.matmul(X4_full, w)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


lr = {'w':args.lr, 'R':args.lr}

lr_decay = 0.90
valid_ = []
test_ = []
wctime_ = []
train_ = []
train_loss_list = []

strict_opt_time = []

for epoch in range(1, 50):
    
    if epoch == 3:
        for key in lr:
            lr[key] = 5e-3
    if epoch==5:
        for key in lr:
            lr[key] = 1e-3
            

    epoch_opt_time = 0
    if epoch==0:
    	_ = sess.run(X3_assign_op)
    start_epoch_time = time.time()
    batch_index = np.arange(len(train_data_batch))
    order = np.random.choice(batch_index, len(train_data_batch), replace = False)

    
    start_alpha_assign_time = time.time()
    _  = sess.run(alpha_3_assign_op)
    epoch_opt_time+= time.time() - start_alpha_assign_time

    
    
    start_R_time = time.time()
    
    r_opt_time = 0
    for index, key in enumerate(order):
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], train_data_batch[key][1]
        
        start_r_opt_time = time.time()
        sess.run(optimizer['R'], {ix1:train_batch, 
                               lr_ph:lr['R'], 
                                  alpha_ix: train_batch})
        r_opt_time+= time.time()-start_r_opt_time
        

    epoch_opt_time+= r_opt_time
    
    
    start_assign_time = time.time()
    _ = sess.run(X3_assign_op)

    epoch_opt_time+= time.time() - start_assign_time
    
    order = np.random.choice(batch_index, len(train_data_batch), replace = False)

    
    start_w_time = time.time()
    w_opt_time = 0
    for index, key in enumerate(order):
        
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], train_data_batch[key][1]
        
        start_w_opt_time = time.time()
        sess.run(optimizer['w'], {ix3:prev_nodes, ix4:train_batch,
                               A1_ph:tf.SparseTensorValue(*adj_mats[0]),
                               lr_ph:lr['w'], label_ph:labels[train_batch]})
        w_opt_time+= time.time()-start_w_opt_time
    
    epoch_opt_time +=w_opt_time
    
        
    end_epoch_time = time.time()

    print(f"Time Taken for epoch: {epoch} is {epoch_opt_time} seconds.")

    full_preds_ = sess.run(preds_f)

    valid_preds = full_preds_[valid_ix]
    test_preds = full_preds_[test_ix]
    train_preds = sess.run(preds)
    valid_f1 = calc_f1(labels[valid_ix], valid_preds, False)
    test_f1 = calc_f1(labels[test_ix], test_preds, False)
    train_f1 = calc_f1(labels[train_ix], train_preds[train_ix], False)

    print("Train - Micro - F1", train_f1)
    print("Valid - Micro - F1", valid_f1)
    print("Test - Micro - F1", test_f1)
    train_loss_val = sess.run(train_loss)
    print("Train Loss:", train_loss_val)    