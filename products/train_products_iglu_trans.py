#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np


# In[2]:


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.sparse as sp
from copy import deepcopy
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import time
from sklearn import metrics


# In[5]:


import numpy as np
import random
import json
import sys
import os


labels = np.load("./data/ogbn_products/Labels.npy")
train_ix = np.load("./data/ogbn_products/TrainSortedIndex.npy")
valid_ix = np.load("./data/ogbn_products/ValidSortedIndex.npy")
test_ix = np.load("./data/ogbn_products/TestSortedIndex.npy")


# In[7]:


feat_data = np.load("./data/ogbn_products/feats.npy")


# In[8]:


from sklearn.preprocessing import StandardScaler
s = StandardScaler()
s.fit(feat_data[train_ix])
feat_trn = s.transform(feat_data)


# In[9]:


indices = {}
indices['train'] = train_ix
indices['valid'] = valid_ix
indices['test'] = test_ix


# In[10]:


import scipy.sparse as sp


# In[11]:


train_adj = sp.load_npz("./data/ogbn_products/adj_train.npz")
full_adj = sp.load_npz("./data/ogbn_products/adj_full.npz")   


# In[12]:


from utils import *


# In[13]:


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


# In[14]:


def get_sparse_adj(A):
    from scipy.sparse import csr_matrix
    return csr_matrix((A[1], (A[0][:, 0], A[0][:, 1])), A[2])

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return (indices, coo.data, coo.shape)

def get_tensor_values(adj, num_adj):
    adj = adj.astype(np.float32)
    num_nodes = adj.shape[0]
    ix = [(i*num_nodes//num_adj, (i+1) * num_nodes//num_adj) for i in range(num_adj)]
    adjs = [adj[ix[i][0]:ix[i][1], :] for i in range(num_adj)]
    sparse_vals = {ix: tf.SparseTensorValue(*convert_sparse_matrix_to_sparse_tensor(adjs[ix]))                   for ix in range(num_adj)}
    return sparse_vals


# In[15]:


def MyLayerNorm(x, name):
    mean, variance = tf.nn.moments(x, axes=[1], keep_dims=True)
    
    offset = zeros([1, x.get_shape()[1]], name=name+'_offset')
    scale  = ones([1, x.get_shape()[1]], name=name+'_scale')
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


# In[16]:


vals = get_tensor_values(full_laplacian, num_adj = 12)
    
value = vals[0]
mat = sp.coo_matrix((value.values, (np.array(value.indices[:, 0]).reshape(-1),    np.array(value.indices[:, 1]).reshape(-1))), value.dense_shape)
mat = mat.tocsr()
vals1 = get_tensor_values(mat, num_adj = 5)

value = vals[1]
mat1 = sp.coo_matrix((value.values, (np.array(value.indices[:, 0]).reshape(-1),    np.array(value.indices[:, 1]).reshape(-1))), value.dense_shape)
mat1 = mat1.tocsr()

vals2 = get_tensor_values(mat1, num_adj = 2)

for key in range(2):
    vals1[len(vals1)] = vals2[key]
for key in range(2, 12):
    vals1[len(vals1)] = vals[key]

# In[18]:


vals = deepcopy(vals1)


#  ###  ```Alpha Computation Graph```

# In[19]:


tf.reset_default_graph()
sess = tf.Session()
# In[20]:


Q1 = glorot([100, 256], name = "Q1")
Q2 = glorot([100, 256], name = "Q2")
R1 = glorot([256, 256], name = "R1")
R2 = glorot([256, 256], name = "R2")
w1 = glorot([256, 256], name = "w1")
w2 = glorot([256, 256], name = "w2")
w = glorot([256, 47], name = "w2")


# In[21]:


N = train_adj.shape[0]


# In[22]:


X0 = tf.Variable(tf.ones([N, 100], tf.float32), name="X1", trainable=False)
X1 = tf.Variable(tf.ones([N, 256], tf.float32), name="X1", trainable=False)
X_general = tf.Variable(tf.ones([N, 256], tf.float32), name="XG", trainable=False)
alpha_3_var = tf.Variable(tf.ones([N, 256], tf.float32), name="Alpha3", trainable=False)
alpha_2_var = tf.Variable(tf.ones([N, 256], tf.float32), name="Alpha2", trainable=False)


# In[23]:


var_X0 = tf.contrib.framework.zero_initializer(X0)
var_X1 = tf.contrib.framework.zero_initializer(X1)
var_Xg = tf.contrib.framework.zero_initializer(X_general)
var_a3 = tf.contrib.framework.zero_initializer(alpha_3_var)
var_a2 = tf.contrib.framework.zero_initializer(alpha_2_var)

_ = sess.run(var_X0)
_ = sess.run(var_X1)
_ = sess.run(var_Xg)
_ = sess.run(var_a3)
_ = sess.run(var_a2)

# In[26]:


drop_rate = tf.placeholder(tf.float32)
data_ix = tf.placeholder(tf.int32)


# In[27]:


# First Layer
ix1 = tf.placeholder(tf.int32, shape = [None])
ix2 = tf.placeholder(tf.int32, shape = [None])
A_ph = tf.sparse_placeholder(tf.float32)

X1_1 = tf.gather(X0, ix1)
X1_2 = tf.gather(X0, ix2)
X1_1 = tf.nn.dropout(X1_1, rate = drop_rate)
X1_2 = tf.nn.dropout(X1_2, rate = drop_rate)

AX1_0 = tf.sparse_tensor_dense_matmul(A_ph, X1_1)
AX1Q1_f = tf.matmul(AX1_0, Q1) 
AX1Q1_f = tf.nn.relu(AX1Q1_f)
X1Q2 = tf.matmul(X1_2, Q2)
X1Q2 = tf.nn.relu(X1Q2)

AX1Q1_f, L1_O1, L1_S1 = MyLayerNorm(AX1Q1_f, name = "Layer1C")
X1Q2, L1_O2, L1_S2 = MyLayerNorm(X1Q2, name = "Layer1D")

X1_f = AX1Q1_f + X1Q2

# Second Layer
ix3 = tf.placeholder(tf.int32, shape = [None])
ix4 = tf.placeholder(tf.int32, shape = [None])
A1_ph = tf.sparse_placeholder(tf.float32)

X2_1 = tf.gather(X1, ix3)
X2_2 = tf.gather(X1, ix4)
X2_1 = tf.nn.dropout(X2_1, rate = drop_rate)
X2_2 = tf.nn.dropout(X2_2, rate = drop_rate)

AX2_0 = tf.sparse_tensor_dense_matmul(A1_ph, X2_1)
AX2R1_f = tf.matmul(AX2_0, R1) 
AX2R1_f = tf.nn.relu(AX2R1_f)
X2R2 = tf.matmul(X2_2, R2)
X2R2 = tf.nn.relu(X2R2)

AX2R1_f, L2_O1, L2_S1 = MyLayerNorm(AX2R1_f, name = "Layer2C")
X2R2, L2_O2, L2_S2 = MyLayerNorm(X2R2, name = "Layer2D")

X2_f = AX2R1_f + X2R2

# Third Layer
ix5 = tf.placeholder(tf.int32, shape = [None])
ix6 = tf.placeholder(tf.int32, shape = [None])
A2_ph = tf.sparse_placeholder(tf.float32)

X3_1 = tf.gather(X_general, ix5)
X3_2 = tf.gather(X_general, ix6)
X3_1 = tf.nn.dropout(X3_1, rate = drop_rate)
X3_2 = tf.nn.dropout(X3_2, rate = drop_rate)

AX3_0 = tf.sparse_tensor_dense_matmul(A2_ph, X3_1)
AX3w1_f = tf.matmul(AX3_0, w1) 
AX3w1_f = tf.nn.relu(AX3w1_f)

X3w2 = tf.matmul(X3_2, w2)
X3w2 = tf.nn.relu(X3w2)

AX3w1_f, L3_O1, L3_S1 = MyLayerNorm(AX3w1_f, name = "Layer3C")
X3w2, L3_O2, L3_S2 = MyLayerNorm(X3w2, name = "Layer3D")

X3_f = AX3w1_f + X3w2
X3_f = tf.nn.l2_normalize(X3_f, axis = 1)


preds_mb = tf.matmul(X3_f, w)

label_ph = tf.placeholder(tf.int32)
loss_mb = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=preds_mb, labels = label_ph)
)

alpha_3 = tf.gradients(loss_mb, X3_1)[0]
alpha_3_assign_op = tf.scatter_update(alpha_3_var, data_ix, alpha_3)

alpha_ix = tf.placeholder(tf.int32, shape = [None])
alpha_3_to_use = tf.gather(alpha_3_var, alpha_ix)

alpha_2 = tf.gradients(alpha_3_to_use * X2_f, X2_1)[0]
alpha_2_assign_op = tf.scatter_update(alpha_2_var, data_ix, alpha_2)
alpha_2_to_use = tf.gather(alpha_2_var, alpha_ix)


# In[28]:


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
                                                 laplacian_matrix,
                                                 depth = depth)
    adjs = [sparse_to_tuple(v) for v in adjs]
    return ((adjs, prev_nodes, sampled_nodes), batch)

train_batches = []
train_index = np.random.permutation(np.arange(len(train_ix)))

ix = 0
for _ in range(0, len(train_ix), 25000):
    train_batches.append(train_ix[train_index[ix: min(ix + 25000, len(train_ix))]]) 
    ix += 25000

arg_list = [(batch, full_laplacian, [5]) for _, batch in enumerate(train_batches)]

import multiprocessing
with multiprocessing.Pool(processes=3) as pool:
    results = pool.starmap(get_sampled_adjs_1,arg_list)
train_data_batch = {
    ix: ((results[ix][0][0], results[ix][0][1], results[ix][0][2]), results[ix][1])
for ix, batch in enumerate(train_batches)
}


# ## For Training Layers that are inner!

# In[29]:


train_batches = []
train_index = np.random.permutation(np.arange((full_laplacian.shape[0])))

ix = 0
for _ in range(0, full_laplacian.shape[0], 150000):
    train_batches.append(train_index[ix: min(ix + 150000, full_laplacian.shape[0])]) 
    ix += 150000

arg_list = [(batch, full_laplacian, [5]) for _, batch in enumerate(train_batches)]

import multiprocessing
with multiprocessing.Pool(processes=3) as pool:
    results = pool.starmap(get_sampled_adjs_1,arg_list)
train_data_batch_full = {
    ix: ((results[ix][0][0], results[ix][0][1], results[ix][0][2]), results[ix][1])
for ix, batch in enumerate(train_batches)
}


# ### Initialize all Trainable variables

# In[31]:


__ = sess.run([var.initializer for var in tf.trainable_variables()])


# In[33]:


phs = {ix:tf.sparse_placeholder(tf.float32) for ix in range(len(vals))}


# ### ```Alpha 3 Computation Graph```

# In[34]:


AX_gen =  tf.concat(
    [tf.sparse_tensor_dense_matmul(phs[ix], X_general)  for ix in range(len(vals))], axis = 0)

AX3W1 = tf.nn.relu(
    tf.matmul(AX_gen, w1)
)
X3W2 = tf.nn.relu(
    tf.matmul(X_general, w2)
)
mean, variance = tf.nn.moments(AX3W1, axes=[1], keep_dims=True)
AX3W1 = tf.nn.batch_normalization(AX3W1, mean, variance, L3_O1, L3_S1, 1e-9)

mean, variance = tf.nn.moments(X3W2, axes=[1], keep_dims=True)
X3W2 = tf.nn.batch_normalization(X3W2, mean, variance, L3_O2, L3_S2, 1e-9)

final_emb = AX3W1 + X3W2
final_emb = tf.nn.l2_normalize(final_emb, axis = 1)
preds = tf.matmul(final_emb, w)

label_train = tf.gather(labels, train_ix)
train_pred_vals = tf.gather(preds, train_ix)
train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= train_pred_vals, labels = label_train))


# ### ```Alpha 2 Computation Graph```

# In[35]:


AX_1 =  tf.concat(
    [tf.sparse_tensor_dense_matmul(phs[ix], X1)  for ix in range(len(vals))], axis = 0)


AX2R1 = tf.nn.relu(
    tf.matmul(AX_1, R1)
)
X2R2 = tf.nn.relu(
    tf.matmul(X1, R2)
)
mean, variance = tf.nn.moments(AX2R1, axes=[1], keep_dims=True)
AX2R1 = tf.nn.batch_normalization(AX2R1, mean, variance, L2_O1, L2_S1, 1e-9)

mean, variance = tf.nn.moments(X2R2, axes=[1], keep_dims=True)
X2R2 = tf.nn.batch_normalization(X2R2, mean, variance, L2_O2, L2_S2, 1e-9)

X3_to_assign = AX2R1 + X2R2

X3_assign_op = X_general.assign(X3_to_assign)


# In[36]:


AX_0 =  tf.concat(
    [tf.sparse_tensor_dense_matmul(phs[ix], X0)  for ix in range(len(vals))], axis = 0)

AX1Q1 = tf.nn.relu(
    tf.matmul(AX_0, Q1)
)
X1Q2 = tf.nn.relu(
    tf.matmul(X0, Q2)
)
mean, variance = tf.nn.moments(AX1Q1, axes=[1], keep_dims=True)
AX1Q1 = tf.nn.batch_normalization(AX1Q1, mean, variance, L1_O1, L1_S1, 1e-9)

mean, variance = tf.nn.moments(X1Q2, axes=[1], keep_dims=True)
X1Q2 = tf.nn.batch_normalization(X1Q2, mean, variance, L1_O2, L1_S2, 1e-9)

X2_to_assign = AX1Q1 + X1Q2

X2_assign_op = X1.assign(X2_to_assign)


# In[39]:


grad_w = tf.gradients(loss_mb , [w, w2, w1, L3_O1, L3_O2, L3_S1, L3_S2])
grad_R = tf.gradients(X2_f * alpha_3_to_use, [R2, R1, L2_O1, L2_O2, L2_S1, L2_S2])
grad_Q = tf.gradients(X1_f * alpha_2_to_use, [Q2, Q1, L1_O1, L1_O2, L1_S1, L1_S2])


# In[40]:


lr_ph = tf.placeholder(tf.float32)
optimizer = {
    'R':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_R, [R2, R1, L2_O1, L2_O2, L2_S1, L2_S2]))),
    'Q':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_Q, [Q2, Q1, L1_O1, L1_O2, L1_S1, L1_S2]))),
    'w':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_w, [w, w2, w1, L3_O1, L3_O2, L3_S1, L3_S2])))
}


# In[41]:


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
    


# In[43]:


import scipy.sparse as sp


# In[49]:


feat_assign_op = X0.assign(feat_trn)


# In[50]:


_ = sess.run(feat_assign_op)
_ = sess.run(X2_assign_op, {phs[ix]:vals[ix] for ix in range(len(vals))})
_ = sess.run(X3_assign_op, {phs[ix]:vals[ix] for ix in range(len(vals))})


# In[52]:


lr = {'w':5e-3, 'R':5e-3, 'Q':5e-3}


# In[53]:


for epoch in range(150):
    if epoch == 100:
        lr = {'w':1e-3, 'R':1e-3, 'Q':1e-3}


    batch_index = np.arange(len(train_data_batch))
    order = np.random.choice(batch_index, len(train_data_batch), replace = False)

    for index, key in enumerate(order):
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], train_data_batch[key][1]
        sess.run(alpha_3_assign_op, {ix5:prev_nodes, ix6:train_batch, data_ix: prev_nodes, drop_rate:0, 
                                    A2_ph:tf.SparseTensorValue(*adj_mats[0]),
                                    label_ph: labels[train_batch]})
        
    batch_index = np.arange(len(train_data_batch_full))
    order = np.random.choice(batch_index, len(train_data_batch_full), replace = False)

    for index, key in enumerate(order):
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch_full[key][0], train_data_batch_full[key][1]
        sess.run(alpha_2_assign_op, {ix3:prev_nodes , ix4:train_batch, data_ix: prev_nodes, drop_rate:0, 
                                    A1_ph:tf.SparseTensorValue(*adj_mats[0]),
                                    alpha_ix:train_batch})
    
    batch_index = np.arange(len(train_data_batch_full))
    order = np.random.choice(batch_index, len(train_data_batch_full), replace = False)


    for index, key in enumerate(order):
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch_full[key][0], train_data_batch_full[key][1]
        sess.run(optimizer['Q'], {ix1:prev_nodes, ix2:train_batch, A_ph:tf.SparseTensorValue(*adj_mats[0]),
                                lr_ph:lr['Q'], alpha_ix: train_batch, drop_rate:0.3})

    
    sess.run(X2_assign_op, {phs[ix]:vals[ix] for ix in range(len(vals))})


    for index, key in enumerate(order):
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch_full[key][0], train_data_batch_full[key][1]
        sess.run(optimizer['R'], {ix3:prev_nodes, ix4:train_batch, A1_ph:tf.SparseTensorValue(*adj_mats[0]),
                                lr_ph:lr['R'], alpha_ix: train_batch, drop_rate:0.3})


    sess.run(X3_assign_op, {phs[ix]:vals[ix] for ix in range(len(vals))})

    batch_index = np.arange(len(train_data_batch))
    order = np.random.choice(batch_index, len(train_data_batch), replace = False)

    for index, key in enumerate(order):
        (adj_mats, prev_nodes, samp_nodes), train_batch = train_data_batch[key][0], train_data_batch[key][1]
        sess.run(optimizer['w'], {ix5:prev_nodes , ix6:train_batch, drop_rate:0.3, 
                                    A2_ph:tf.SparseTensorValue(*adj_mats[0]),
                                    label_ph: labels[train_batch],
                                 lr_ph:lr['w']})
        
    full_preds, train_loss_val = sess.run([preds, train_loss], {phs[ix]:vals[ix] for ix in range(len(vals))})

    valid_preds = full_preds[valid_ix]
    test_preds = full_preds[test_ix]
    train_preds = full_preds[train_ix]

    valid_f1 = calc_f1(labels[valid_ix], valid_preds, False)
    test_f1 = calc_f1(labels[test_ix], test_preds, False)
    train_f1 = calc_f1(labels[train_ix], train_preds, False)

    print("Loss: ", train_loss_val)

    print("Train - Micro - F1", train_f1)
    print("Valid - Micro - F1", valid_f1)
    print("Test - Micro - F1", test_f1)
    print()

