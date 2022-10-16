import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import argparse
import numpy as np
import os
import scipy.sparse as sp
from copy import deepcopy
from sklearn.metrics import f1_score
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

def MyLayerNorm(x, ctr_val):
    mean, variance = tf.nn.moments(x, axes=[1], keep_dims=True)
    
    offset = zeros([1, x.get_shape()[1]], name=('offset'+str(ctr_val)))
    scale  = ones([1, x.get_shape()[1]], name=('scale'+str(ctr_val)))
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
parser.add_argument('--learning_rate', metavar='FLOAT', default = 1e-3, type = float, dest='lr', help = 'Learning Rate')
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
train_adj = sp.load_npz(args.data_path+"/adj_train.npz")
full_adj = sp.load_npz(args.data_path+"/adj_full.npz")
print("Time to load data:", time.time()-start_data_time)
indices = {}
indices['train'] = train_ix
indices['valid'] = valid_ix
indices['test'] = test_ix

from utils import *
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
    
vals = get_tensor_values(train_laplacian, num_adj = 1)


mask = [False for i in range((train_laplacian.shape[0]))]
train_mask = deepcopy(mask)
valid_mask = deepcopy(mask)
test_mask = deepcopy(mask)

train_labels = np.zeros((train_laplacian.shape[0], 7))
train_labels[train_ix] = labels[train_ix]
train_labels = train_labels.astype(np.float32)

valid_labels = np.zeros((train_laplacian.shape[0], 7))
valid_labels[valid_ix] = labels[valid_ix]
valid_labels = valid_labels.astype(np.float32)

test_labels = np.zeros((train_laplacian.shape[0], 7))
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

train_feat = np.zeros((train_laplacian.shape[0], feat_data.shape[1]))
train_feat[train_ix] = feat_data[train_ix]

tf.reset_default_graph()

#3 layer weight matrices
Q1 = glorot([500,128 ], name = "Q1")
Q2 = glorot([500,128 ], name = "Q2")

R1 = glorot([256, 128], name = "R1")
R2 = glorot([256, 128], name = "R2")

w = glorot([256, 7], name = "w")

X0 = tf.Variable(np.zeros((full_laplacian.shape[0], 500)).astype(np.float32), trainable = False)
X1 = tf.Variable(np.zeros((full_laplacian.shape[0], 256)).astype(np.float32), trainable = False)
X2 = tf.Variable(np.zeros((full_laplacian.shape[0], 256)).astype(np.float32), trainable = False)
alpha_2_var = tf.Variable(np.zeros((full_laplacian.shape[0], 256)).astype(np.float32), trainable = False)

AX0 = train_laplacian.dot(train_feat)
AX0_var = tf.Variable(AX0.astype(np.float32), trainable = False)

AX0Q1 = tf.matmul(AX0_var, Q1)
Z11r = tf.nn.relu(AX0Q1)
X11_outs, L11_O, L11_S = MyLayerNorm(Z11r, 1)

X0Q2 = tf.matmul(X0, Q2)
Z12r = tf.nn.relu(X0Q2)
X12_outs, L12_O, L12_S = MyLayerNorm(Z12r, 2)

X1_outs = tf.concat([X11_outs, X12_outs], axis=1)

X1_to_assign = X1_outs 
X1_assign_op = X1.assign(X1_to_assign)

#Layer 2 computation graph
X1R1 = tf.matmul(X1,R1)
AX1R1 = tf.concat([tf.sparse_tensor_dense_matmul(vals[i], X1R1) for i in range(len(vals))], axis=0)
Z21r = tf.nn.relu(AX1R1)
X21_outs, L21_O, L21_S = MyLayerNorm(Z21r, 3)


X1R2 = tf.matmul(X1, R2)
Z22r = tf.nn.relu(X1R2)
X22_outs, L22_O, L22_S = MyLayerNorm(Z22r, 4)

X2_outs = tf.nn.l2_normalize(tf.concat([X21_outs, X22_outs], axis=1),1)

preds = tf.matmul(X2_outs, w)

train_loss = masked_softmax_cross_entropy(preds, train_labels, train_mask) 

alpha_2 = tf.gradients(train_loss, X1)[0]
alpha_2_assign_op = alpha_2_var.assign(alpha_2)

alpha_2_ix = tf.placeholder(tf.int32, shape= [None])
alpha_2_to_use = tf.gather(alpha_2_var, alpha_2_ix)

#Full Computation Graph 
ix1 = tf.placeholder(tf.int32, shape= [None])
AX0_f = tf.gather(AX0_var, ix1)
AX0Q1_f = tf.nn.relu(tf.matmul(AX0_f, Q1))
mean, variance = tf.nn.moments(AX0Q1_f, axes=[1], keep_dims = True)
AX0Q1_f = tf.nn.batch_normalization(AX0Q1_f, mean, variance, L11_O, L11_S, 1e-9)

X0_f = tf.gather(X0, ix1)
X0Q2_f = tf.nn.relu(tf.matmul(X0_f, Q2))
mean, variance = tf.nn.moments(X0Q2_f, axes = [1], keep_dims = True)
X0Q2_f = tf.nn.batch_normalization(X0Q2_f, mean, variance, L12_O, L12_S, 1e-9)

X1_f = tf.concat([AX0Q1_f, X0Q2_f], axis = 1)

#layer 2
ix3 = tf.placeholder(tf.int32, shape = [None])
ix4 = tf.placeholder(tf.int32, shape = [None])

X11_f = tf.gather(X1, ix3)
X11R1_f = tf.matmul(X11_f, R1)
A1_ph = tf.sparse_placeholder(tf.float32)
AX1R1_f = tf.nn.relu(tf.sparse_tensor_dense_matmul(A1_ph, X11R1_f))
mean, variance = tf.nn.moments(AX1R1_f, axes=[1], keep_dims = True)
AX1R1_f = tf.nn.batch_normalization(AX1R1_f, mean, variance, L21_O, L21_S, 1e-9)

X12_f = tf.gather(X1, ix4)
X12R2_f = tf.nn.relu(tf.matmul(X12_f, R2))
mean, variance = tf.nn.moments(X12R2_f, axes=[1], keep_dims = True)
X12R2_f = tf.nn.batch_normalization(X12R2_f, mean, variance, L22_O, L22_S, 1e-9)

X2_f = tf.nn.l2_normalize(tf.concat([AX1R1_f, X12R2_f], axis=1),1)

preds_mb = tf.matmul(X2_f, w)

label_ph = tf.placeholder(tf.int32)
loss_mb = tf.nn.softmax_cross_entropy_with_logits(logits = preds_mb, labels = label_ph)

grad_R = tf.gradients(loss_mb, [w, R1, R2, L21_O, L21_S, L22_O, L22_S])

grad_Q = tf.gradients(tf.multiply(alpha_2_to_use, X1_f), [Q1, Q2, L11_O, L11_S, L12_O, L12_S])

variables = tf.trainable_variables()
print(variables)

lr_ph = tf.placeholder(tf.float32)

optimizer = {
    
    'Q':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_Q, [Q1, Q2, L11_O, L11_S, L12_O, L12_S]))),
    
    'R':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_R, [w, R1, R2, L21_O, L21_S, L22_O, L22_S]))),
    
}

train_batches = []
train_index = np.random.permutation(np.arange(len(train_ix)))

ix = 0
for _ in range(0, len(train_ix), args.batch_size):
    train_batches.append(train_ix[train_index[ix: min(ix + args.batch_size, len(train_ix))]]) 
    ix += args.batch_size

arg_list = [(batch, train_laplacian, [10]) for _, batch in enumerate(train_batches)]

with multiprocessing.Pool(processes=30) as pool:
    results = pool.starmap(get_sampled_adjs_1,arg_list)
train_data_batch = {
    ix: ((results[ix][0][0], results[ix][0][1], results[ix][0][2]), results[ix][1])
for ix, batch in enumerate(train_batches)
}

vals_f = get_tensor_values(full_laplacian, num_adj = 1)

AX0_full = full_laplacian.dot(feat_data)
AX0_full_var = tf.Variable(AX0_full.astype(np.float32), trainable = False)

Z11_full = tf.nn.relu(tf.matmul(AX0_full_var, Q1))
mean, variance = tf.nn.moments(Z11_full, axes=[1], keep_dims=True)
Z11_full_out = tf.nn.batch_normalization(Z11_full, mean, variance, L11_O, L11_S, 1e-9)

X0_full = feat_data.astype(np.float32)
Z12_full = tf.nn.relu(tf.matmul(X0_full,Q2))
mean, variance = tf.nn.moments(Z12_full, axes=[1], keep_dims=True)
Z12_full_out = tf.nn.batch_normalization(Z12_full, mean, variance, L12_O, L12_S, 1e-9)

X1_full = tf.concat([Z11_full_out, Z12_full_out], axis=1)


X1R1_full = tf.matmul(X1_full, R1)
X1R2_full = tf.matmul(X1_full, R2)

Z21_full = tf.nn.relu(tf.concat([tf.sparse_tensor_dense_matmul(vals_f[i], X1R1_full) for i in range(len(vals_f))], axis = 0))
mean, variance = tf.nn.moments(Z21_full, axes=[1], keep_dims=True)
Z21_full_out = tf.nn.batch_normalization(Z21_full, mean, variance, L21_O, L21_S, 1e-9)

Z22_full = tf.nn.relu(X1R2_full)
mean, variance = tf.nn.moments(Z22_full, axes=[1], keep_dims=True)
Z22_full_out = tf.nn.batch_normalization(Z22_full, mean, variance, L22_O, L22_S, 1e-9)

X2_full = tf.nn.l2_normalize(tf.concat([Z21_full_out, Z22_full_out],axis=1),1)

preds_f = tf.matmul(X2_full, w)

sess = tf.Session()

init = tf.global_variables_initializer()


sess.run(init)
_ = sess.run(X0.assign(train_feat.astype(np.float32)))

lr_value = args.lr

train_loss_list = []
train_f1_list = []
valid_f1_list = []
test_f1_list = []
epoch_times_list = []

for epoch in range(50):

    if epoch==0:
        start_epoch_time_0 = time.time()
        _ = sess.run(X1_assign_op)
        print("Epoch 0 time:", (time.time()-start_epoch_time_0))

    if epoch ==20:
        lr_value = 1e-4


        
    start_epoch_time = time.time()

    batch_index = np.arange(len(train_data_batch))

    #Update alpha 2 values
    start_alpha_assign_time = time.time()
    _  = sess.run(alpha_2_assign_op)

    #Update Q
    start_Q = time.time()
    order = np.random.choice(batch_index, len(train_data_batch), replace = False)
    for index, key in enumerate(order):
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], \
                                                      train_data_batch[key][1]
        sess.run(optimizer['Q'], feed_dict = {ix1:train_batch, alpha_2_ix: train_batch, lr_ph:lr_value})


    start_x1_assign_op = time.time()
    _ = sess.run(X1_assign_op)

    # Update R.
    start_R = time.time()
    order = np.random.choice(batch_index, len(train_data_batch), replace = False)
    for index, key in enumerate(order):
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], \
                                                      train_data_batch[key][1]

        sess.run(optimizer['R'], 
         feed_dict = {ix3:prev_nodes,
                      ix4:train_batch,
                      A1_ph: tf.SparseTensorValue(*adj_mats[0]),
                      lr_ph:lr_value,
                    label_ph:labels[train_batch] })


    end_epoch_time = time.time()


    print(f"Time Taken for epoch: {epoch} is {end_epoch_time - start_epoch_time} seconds.")
    epoch_times_list.append((end_epoch_time - start_epoch_time))

    loss_value = sess.run(train_loss)
    full_preds = sess.run(preds_f)

    valid_preds = full_preds[valid_ix]
    test_preds = full_preds[test_ix]
    train_preds = sess.run(preds)[train_ix]
    valid_f1 = calc_f1(labels[valid_ix], valid_preds, False)
    test_f1 = calc_f1(labels[test_ix], test_preds, False)
    train_f1 = calc_f1(labels[train_ix], train_preds, False)

    print("Train Loss:", loss_value)
    print("Train - Micro - F1", train_f1)
    print("Valid - Micro - F1", valid_f1)
    print("Test - Micro - F1", test_f1)

    train_loss_list.append(loss_value)
    train_f1_list.append(train_f1)
    valid_f1_list.append(valid_f1)
    test_f1_list.append(test_f1)

    print()