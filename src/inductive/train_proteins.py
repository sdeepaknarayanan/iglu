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
from sklearn.metrics import roc_auc_score

    
def get_sparse_adj(A):
    from scipy.sparse import csr_matrix
    return csr_matrix((A[1], (A[0][:, 0], A[0][:, 1])), A[2])

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return (indices, coo.data, coo.shape)

def MyLayerNorm(x, ctr_val):
    mean, variance = tf.nn.moments(x, axes=[0], keep_dims=True)
    
    offset = zeros([1, x.get_shape()[1]], name=('offset'+str(ctr_val)))
    scale  = ones([1, x.get_shape()[1]], name=('scale'+str(ctr_val)))
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

def set_gpu(gpus):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Check if tensorflow is running on a gpu
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

def get_tensor_values(adj, num_adj):
    adj = adj.astype(np.float32)
    num_nodes = adj.shape[0]
    ix = [(i*num_nodes//num_adj, (i+1) * num_nodes//num_adj) for i in range(num_adj)]
    adjs = [adj[ix[i][0]:ix[i][1], :] for i in range(num_adj)]
    sparse_vals = {ix: tf.SparseTensorValue(*convert_sparse_matrix_to_sparse_tensor(adjs[ix])) \
                  for ix in range(num_adj)}
    return sparse_vals

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_sigmoid_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= tf.reshape(mask, [-1, 1])
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
parser.add_argument('--batchsize', metavar='INT', default = 512, dest='batch_size', type=int,
help = 'Number of Nodes per minibatch.')
parser.add_argument('--learning_rate', metavar='FLOAT', default = 1e-2, type = float, dest='lr', help = 'Learning Rate')
parser.add_argument('--gpuid', metavar='0|1|2|3',default = 1,  type = int, dest='gpuid', help = 'GPU ID')
parser.add_argument('--data_path', metavar='STRING',  type = str, dest='data_path', help = 'Path to the dataset')

args = parser.parse_args()
        
set_gpu(str(args.gpuid))

#Load Data
start_data_time = time.time()
labels = np.load(args.data_path+"/Labels.npy")
train_ix = np.load(args.data_path+"/SortedTrainIndex.npy")
valid_ix = np.load(args.data_path+"/SortedValidIndex.npy")
test_ix = np.load(args.data_path+"/SortedTestIndex.npy")
feat_data = np.load(args.data_path+"/feat.npy")
train_adj = sp.load_npz(args.data_path+"/adj_train.npz")
full_adj = sp.load_npz(args.data_path+"/adj_full.npz")
print("Time to load data:", time.time()-start_data_time)

indices = {}
indices['train'] = train_ix
indices['valid'] = valid_ix
indices['test'] = test_ix


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
    

train_laplacian = train_laplacian[:train_ix.shape[0]][:, :train_ix.shape[0]]

adj= train_laplacian
num_adj = 32
adj = adj.astype(np.float32)

num_nodes = adj.shape[0]
num_nodes = 86619
ix = [(i*num_nodes//num_adj, (i+1) * num_nodes//num_adj) for i in range(num_adj)]
adjs = [adj[ix[i][0]:ix[i][1], :] for i in range(num_adj)]
sparse_vals = {ix: tf.SparseTensorValue(*convert_sparse_matrix_to_sparse_tensor(adjs[ix])) \
              for ix in range(num_adj)}

vals = sparse_vals

train_feat = np.zeros((train_laplacian.shape[0], feat_data.shape[1]))
train_feat[train_ix] = feat_data[train_ix]

tf.reset_default_graph()

#3 layer weight matrices
Q1 = glorot([8,256 ], name = "Q1")
Q2 = glorot([8,256 ], name = "Q2")

R1 = glorot([256, 256], name = "R1")
R2 = glorot([256, 256], name = "R2")

S1 = glorot([256, 256], name = "S1")
S2 = glorot([256, 256], name = "S2")

w  = glorot([256, 112], name = "w")


X0 = tf.Variable(np.zeros((train_laplacian.shape[0], 8)).astype(np.float32), trainable = False)
X1 = tf.Variable(np.zeros((train_laplacian.shape[0], 256)).astype(np.float32), trainable = False)
X2 = tf.Variable(np.zeros((train_laplacian.shape[0], 256)).astype(np.float32), trainable = False)
alpha_2_var = tf.Variable(np.zeros((train_laplacian.shape[0], 256)).astype(np.float32), trainable = False)
alpha_1_var = tf.Variable(np.zeros((train_laplacian.shape[0], 256)).astype(np.float32), trainable = False)

AX0 = train_laplacian.dot(train_feat)
AX0_var = tf.Variable(AX0.astype(np.float32), trainable = False)

AX0Q1 = tf.matmul(AX0_var, Q1)
Z11r = tf.nn.relu(AX0Q1)
X11_outs, L11_O, L11_S = MyLayerNorm(Z11r, 1)

X0Q2 = tf.matmul(X0, Q2)
Z12r = tf.nn.relu(X0Q2)
X12_outs, L12_O, L12_S = MyLayerNorm(Z12r, 2)

X1_outs = tf.add(X11_outs, X12_outs)

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


X2_outs = tf.add(X21_outs, X22_outs)

X2_to_assign = X2_outs
X2_assign_op = X2.assign(X2_to_assign)

X2S1 = tf.matmul(X2,S1)
AX2S1 = tf.concat([tf.sparse_tensor_dense_matmul(vals[i], X2S1) for i in range(len(vals))], axis=0)
Z31r = tf.nn.relu(AX2S1)
X31_outs, L31_O, L31_S = MyLayerNorm(Z31r, 5)

X2S2 = tf.matmul(X2, S2)
Z32r = tf.nn.relu(X2S2)
X32_outs, L32_O, L32_S = MyLayerNorm(Z32r, 6)

X3_outs = tf.nn.l2_normalize(tf.add(X31_outs, X32_outs),1)

train_preds = tf.matmul(X3_outs, w)

label_train = tf.gather(labels.astype(np.float32), train_ix)
train_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= train_preds, labels = label_train))

alpha_2 = tf.gradients(train_loss, X2)[0]
alpha_2_assign_op = alpha_2_var.assign(alpha_2)

alpha_2_ix = tf.placeholder(tf.int32, shape= [None])
alpha_2_to_use = tf.gather(alpha_2_var, alpha_2_ix)


alpha_1 = tf.gradients(tf.multiply(alpha_2_to_use, X2_to_assign), X1)[0]
alpha_1_assign_op = alpha_1_var.assign(alpha_1)

alpha_1_ix = tf.placeholder(tf.int32, shape = [None])
alpha_1_to_use = tf.gather(alpha_1_var, alpha_1_ix)


#Full Computation Graph 
ix1 = tf.placeholder(tf.int32, shape= [None])
AX0_f = tf.gather(AX0_var, ix1)
AX0Q1_f = tf.nn.relu(tf.matmul(AX0_f, Q1))
mean, variance = tf.nn.moments(AX0Q1_f, axes=[0], keep_dims = True)
AX0Q1_f = tf.nn.batch_normalization(AX0Q1_f, mean, variance, L11_O, L11_S, 1e-9)

X0_f = tf.gather(X0, ix1)
X0Q2_f = tf.nn.relu(tf.matmul(X0_f, Q2))
mean, variance = tf.nn.moments(X0Q2_f, axes = [0], keep_dims = True)
X0Q2_f = tf.nn.batch_normalization(X0Q2_f, mean, variance, L12_O, L12_S, 1e-9)

X1_f = tf.add(AX0Q1_f, X0Q2_f)

#layer 2
ix3 = tf.placeholder(tf.int32, shape = [None])
ix4 = tf.placeholder(tf.int32, shape = [None])

X11_f = tf.gather(X1, ix3)
X11R1_f = tf.matmul(X11_f, R1)
A1_ph = tf.sparse_placeholder(tf.float32)
AX1R1_f = tf.nn.relu(tf.sparse_tensor_dense_matmul(A1_ph, X11R1_f))
mean, variance = tf.nn.moments(AX1R1_f, axes=[0], keep_dims = True)
AX1R1_f = tf.nn.batch_normalization(AX1R1_f, mean, variance, L21_O, L21_S, 1e-9)

X12_f = tf.gather(X1, ix4)
X12R2_f = tf.nn.relu(tf.matmul(X12_f, R2))
mean, variance = tf.nn.moments(X12R2_f, axes=[0], keep_dims = True)
X12R2_f = tf.nn.batch_normalization(X12R2_f, mean, variance, L22_O, L22_S, 1e-9)

X2_f = tf.add(AX1R1_f, X12R2_f)

#layer 2
ix5 = tf.placeholder(tf.int32, shape = [None])
ix6 = tf.placeholder(tf.int32, shape = [None])

X21_f = tf.gather(X2, ix5)
X21S1_f = tf.matmul(X21_f, S1)
A2_ph = tf.sparse_placeholder(tf.float32)
AX2S1_f = tf.nn.relu(tf.sparse_tensor_dense_matmul(A2_ph, X21S1_f))
mean, variance = tf.nn.moments(AX2S1_f, axes=[0], keep_dims = True)
AX2S1_f = tf.nn.batch_normalization(AX2S1_f, mean, variance, L31_O, L31_S, 1e-9)

X22_f = tf.gather(X2, ix6)
X22S2_f = tf.nn.relu(tf.matmul(X22_f, S2))
mean, variance = tf.nn.moments(X22S2_f, axes=[0], keep_dims = True)
X22S2_f = tf.nn.batch_normalization(X22S2_f, mean, variance, L32_O, L32_S, 1e-9)

X3_f = tf.nn.l2_normalize(tf.add(AX2S1_f, X22S2_f),1)

preds_mb = tf.matmul(X3_f, w)

label_ph = tf.placeholder(tf.float32)
loss_mb = tf.nn.sigmoid_cross_entropy_with_logits(logits = preds_mb, labels = label_ph)

#Defining the gradients 

grad_S = tf.gradients(loss_mb, [w, S1, S2, L31_O, L31_S, L32_O, L32_S])

grad_R = tf.gradients(tf.multiply(alpha_2_to_use, X2_f), [R1, R2, L21_O, L21_S, L22_O, L22_S])

grad_Q = tf.gradients(tf.multiply(alpha_1_to_use, X1_f), [Q1, Q2, L11_O, L11_S, L12_O, L12_S])

variables = tf.trainable_variables()
print(variables)

lr_ph = tf.placeholder(tf.float32)

optimizer = {
    
    'Q':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_Q, [Q1, Q2, L11_O, L11_S, L12_O, L12_S]))),
    
    'R':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_R, [R1, R2, L21_O, L21_S, L22_O, L22_S]))),
    
    'S':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_S, [w, S1, S2, L31_O, L31_S, L32_O, L32_S]))),

}

batch_size = args.batch_size

train_batches = []
train_index = np.random.permutation(np.arange(len(train_ix)))

ix = 0
for _ in range(0, len(train_ix), batch_size):
    train_batches.append(train_ix[train_index[ix: min(ix + batch_size, len(train_ix))]]) 
    ix += batch_size

arg_list = [(batch, train_laplacian, [10]) for _, batch in enumerate(train_batches)]

with multiprocessing.Pool(processes=3) as pool:
    results = pool.starmap(get_sampled_adjs_1,arg_list)
train_data_batch = {
    ix: ((results[ix][0][0], results[ix][0][1], results[ix][0][2]), results[ix][1])
for ix, batch in enumerate(train_batches)
}

vals_f = get_tensor_values(full_laplacian, num_adj = 20)

#Defining the computation graph for Inference

AX0_full = full_laplacian.dot(feat_data)
AX0_full_var = tf.Variable(AX0_full.astype(np.float32), trainable = False)

Z11_full = tf.nn.relu(tf.matmul(AX0_full_var, Q1))
mean, variance = tf.nn.moments(Z11_full, axes=[0], keep_dims=True)
Z11_full_out = tf.nn.batch_normalization(Z11_full, mean, variance, L11_O, L11_S, 1e-9)


X0_full = feat_data.astype(np.float32)
Z12_full = tf.nn.relu(tf.matmul(X0_full,Q2))
mean, variance = tf.nn.moments(Z12_full, axes=[0], keep_dims=True)
Z12_full_out = tf.nn.batch_normalization(Z12_full, mean, variance, L12_O, L12_S, 1e-9)


X1_full = tf.add(Z11_full_out, Z12_full_out)


X1R1_full = tf.matmul(X1_full, R1)
X1R2_full = tf.matmul(X1_full, R2)

Z21_full = tf.nn.relu(tf.concat([tf.sparse_tensor_dense_matmul(vals_f[i], X1R1_full) for i in range(len(vals_f))], axis = 0))
mean, variance = tf.nn.moments(Z21_full, axes=[0], keep_dims=True)
Z21_full_out = tf.nn.batch_normalization(Z21_full, mean, variance, L21_O, L21_S, 1e-9)

Z22_full = tf.nn.relu(X1R2_full)
mean, variance = tf.nn.moments(Z22_full, axes=[0], keep_dims=True)
Z22_full_out = tf.nn.batch_normalization(Z22_full, mean, variance, L22_O, L22_S, 1e-9)

X2_full = tf.add(Z21_full_out, Z22_full_out)

X2S1_full = tf.matmul(X2_full, S1)
X2S2_full = tf.matmul(X2_full, S2)

Z31_full = tf.nn.relu(tf.concat([tf.sparse_tensor_dense_matmul(vals_f[i], X2S1_full) for i in range(len(vals_f))], axis = 0))
mean, variance = tf.nn.moments(Z31_full, axes=[0], keep_dims=True)
Z31_full_out = tf.nn.batch_normalization(Z31_full, mean, variance, L31_O, L31_S, 1e-9)

Z32_full = tf.nn.relu(X2S2_full)
mean, variance = tf.nn.moments(Z32_full, axes=[0], keep_dims=True)
Z32_full_out = tf.nn.batch_normalization(Z32_full, mean, variance, L32_O, L32_S, 1e-9)

X3_full = tf.nn.l2_normalize(tf.add(Z31_full_out, Z32_full_out),1)

preds_f = tf.matmul(X3_full, w)


sess = tf.Session()

init = tf.global_variables_initializer()



sess.run(init)
_ = sess.run(X0.assign(train_feat.astype(np.float32)))

_ = sess.run(X1_assign_op)

_ = sess.run(X2_assign_op)

lr_value = args.lr
lr_decay = 0.9


for epoch in range(200):


    if epoch%4==0:
        lr_value = lr_value*lr_decay

    start_epoch_time = time.time()

    batch_index = np.arange(len(train_data_batch))

    _  = sess.run(alpha_2_assign_op)

    _  = sess.run(alpha_1_assign_op, feed_dict = {alpha_2_ix:train_ix})

    #Update Q
    order = np.random.choice(batch_index, len(train_data_batch), replace = False)
    for index, key in enumerate(order):
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], \
                                                      train_data_batch[key][1]
        sess.run(optimizer['Q'], feed_dict = {ix1:train_batch, alpha_1_ix: train_batch, lr_ph:lr_value})


    _ = sess.run(X1_assign_op)


    # Update R.
    order = np.random.choice(batch_index, len(train_data_batch), replace = False)
    for index, key in enumerate(order):
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], \
                                                      train_data_batch[key][1]

        sess.run(optimizer['R'], 
         feed_dict = {ix3:prev_nodes,
                      ix4:train_batch,
                      A1_ph: tf.SparseTensorValue(*adj_mats[0]),
                      lr_ph:lr_value,
                      alpha_2_ix: train_batch
                     })


    _ = sess.run(X2_assign_op)

    # Update S.
    order = np.random.choice(batch_index, len(train_data_batch), replace = False)
    for index, key in enumerate(order):
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], \
                                                      train_data_batch[key][1]

        sess.run(optimizer['S'], 
         feed_dict = {ix5:prev_nodes,
                      ix6:train_batch,
                      A2_ph: tf.SparseTensorValue(*adj_mats[0]),
                      lr_ph:lr_value,
                    label_ph:labels[train_batch] })


    end_epoch_time = time.time()


    print(f"Time Taken for epoch: {epoch} is {end_epoch_time - start_epoch_time} seconds.")

    loss_value = sess.run(train_loss)
    full_preds = sess.run(preds_f)

    valid_preds = full_preds[valid_ix]
    test_preds = full_preds[test_ix]

    train_preds_outs = sess.run(train_preds)


    valid_f1 = roc_auc_score(labels[valid_ix], valid_preds)
    test_f1 = roc_auc_score(labels[test_ix], test_preds)
    train_f1 = roc_auc_score(labels[train_ix], train_preds_outs)

    print("Train Loss:", loss_value)
    print("Train - ROC-AUC", train_f1)
    print("Valid - ROC-AUC", valid_f1)
    print("Test - ROC-AUC", test_f1)

    print()