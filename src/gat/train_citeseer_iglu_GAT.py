#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np


# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[5]:


from utils import *


# In[6]:


import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


# In[7]:


tf.reset_default_graph()


# In[8]:


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask =  load_data("citeseer")


# In[9]:


pad = preprocess_adj(adj)


# In[10]:


adj_full = sp.coo_matrix((pad[1], (pad[0][:, 0], pad[0][:, 1])), pad[2])


# In[11]:


full_laplacian = adj_full


# In[12]:


X = features.todense().astype(np.float32)


# In[13]:


num_layers = 2
num_heads = [8, 1]


# In[14]:


W = {layer:{head:None for head in range(num_heads[layer])} for layer in range(num_layers)}
A = {layer:{head:None for head in range(num_heads[layer])} for layer in range(num_layers)}
B = {layer:{head:None for head in range(num_heads[layer])} for layer in range(num_layers)}
XW = {layer:{head:None for head in range(num_heads[layer])} for layer in range(num_layers)}
att_mats = {layer:{head:None for head in range(num_heads[layer])} for layer in range(num_layers)}


# In[15]:


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


# In[16]:


def dropout(x, keep_prob, seed=123):
    if isinstance(x, tf.SparseTensor):
        values = x.values 
        values = tf.nn.dropout(values, keep_prob)
        res = tf.SparseTensor(x.indices, values, x.dense_shape)
    else:
        res = tf.nn.dropout(x, keep_prob)
    return res


# In[17]:


tf.reset_default_graph()


# In[20]:


alpha = tf.get_variable(name="Alpha", shape=[3327,64], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
X1_var = tf.get_variable(name="X1", shape=[3327,64], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)


# In[21]:


with tf.variable_scope("Layer1"):
    for attn_head in range(num_heads[0]):
        W[0][attn_head] = glorot([X.shape[1], 8], name = f"Head_{attn_head}")
        A[0][attn_head] = glorot([8*2, 1], name = f"Attn_Head_{attn_head}")
        B[0][attn_head] = tf.get_variable(shape=[1, ], name=f"Bias_{attn_head}",dtype=tf.float32, 
                                         initializer=tf.zeros_initializer())
with tf.variable_scope("Layer2"):
    for attn_head in range(num_heads[1]):
        W[1][attn_head] = glorot([8*8, y_train.shape[1]], name = f"Head_{attn_head}")
        A[1][attn_head] = glorot([2*y_train.shape[1], 1], name = f"Attn_Head_{attn_head}")
        B[1][attn_head] = tf.get_variable(shape=[1, ], name=f"Bias_{attn_head}",dtype=tf.float32, 
                                         initializer=tf.zeros_initializer())


# In[22]:


drop_ph = tf.placeholder(tf.float32)


# In[23]:


X_in = dropout(X, 1 - drop_ph)


# In[24]:


for layer in range(1):
    for head in range(num_heads[layer]):
        XW[layer][head] = dropout(tf.matmul(X, W[layer][head]), 1 - drop_ph)


# In[25]:


B1 = tf.get_variable(name="Bias1", shape=[64, ], dtype=tf.float32)
B2= tf.get_variable(name="Bias2", shape=[y_train.shape[1], ], dtype=tf.float32)


# In[26]:


coo_mat = full_laplacian.tocoo()
indices = np.concatenate([coo_mat.row.reshape(-1, 1), coo_mat.col.reshape(-1, 1)], axis = 1)


# In[27]:


for layer in range(0, 1):
    for head in range(num_heads[layer]):
        features = XW[layer][head]
        pairwise_features = tf.concat([tf.gather(features,indices[:,0]),tf.gather(features,indices[:,1])],axis=1)
        attention_coefficients = tf.matmul(pairwise_features, A[layer][head]) 
        attention_coefficients = tf.nn.leaky_relu(attention_coefficients, alpha=0.2) + B[layer][head]
        attention_coefficients = tf.reshape(attention_coefficients, (-1,))
        attention_matrix = tf.SparseTensor(indices=indices,values=attention_coefficients,dense_shape=full_laplacian.shape)
        attention_matrix = tf.sparse_reorder(attention_matrix)
        attention_matrix = tf.sparse_softmax(attention_matrix)
        att_mats[layer][head] = attention_matrix


# In[28]:


AXW = tf.concat([tf.sparse_tensor_dense_matmul(dropout(att_mats[0][head], 1-drop_ph), XW[0][head])for head in att_mats[0]], axis = 1) + B1
X1_to_assign = tf.nn.elu(AXW)
X1_to_assign = dropout(X1_to_assign, keep_prob= 1 - drop_ph)
X1_assign_op = X1_var.assign(X1_to_assign)


# In[29]:


for layer in range(1, 2):
    for head in range(num_heads[layer]):
        XW[layer][head] = tf.matmul(X1_var, W[layer][head])


# In[30]:


for layer in range(1, 2):
    for head in range(num_heads[layer]):
        features = XW[layer][head]
        pairwise_features = tf.concat([tf.gather(features,indices[:,0]),tf.gather(features,indices[:,1])],axis=1)
        attention_coefficients = tf.matmul(pairwise_features, A[layer][head])
        attention_coefficients = tf.nn.leaky_relu(attention_coefficients, alpha=0.2)  + B[layer][head]
        attention_coefficients = tf.reshape(attention_coefficients, (-1,))
        attention_matrix = tf.SparseTensor(indices=indices,values=attention_coefficients,dense_shape=full_laplacian.shape)
        attention_matrix = tf.sparse_reorder(attention_matrix)
        attention_matrix = tf.sparse_softmax(attention_matrix)
        att_mats[layer][head] = attention_matrix       
############################################################################################

ind_preds = [tf.sparse_tensor_dense_matmul(dropout(att_mats[1][head], 1 - drop_ph), XW[1][head]) for head in att_mats[1]]
for ix, elem in enumerate(ind_preds):
    if ix == 0:
        final_preds = elem
    else:
        final_preds+=elem
preds = final_preds + B2


# In[31]:


wdecay = 5e-2


# In[32]:


weight_params = []
for v in tf.trainable_variables():
    if 'Bias' not in v.name:
        weight_params.append(v)
reg_loss = wdecay *  tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])


# In[33]:


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


# In[34]:


label_ph = tf.placeholder(tf.float32, shape=[None, None])
mask_ph = tf.placeholder(tf.float32, shape=[None])


# In[35]:


loss = masked_softmax_cross_entropy(preds, label_ph, mask_ph)


# In[36]:


alpha_assign_op = alpha.assign(tf.gradients(loss, X1_var)[0])


# In[37]:


l1_vars = []
for var in tf.trainable_variables():
    if 'Layer1' in var.name:
        l1_vars.append(var)
    if 'Bias1' in var.name:
        l1_vars.append(var)
l2_vars = list(set(tf.trainable_variables()) - set(l1_vars))


# In[38]:


var_vals = []
for v in l1_vars:
    if 'Bias' not in v.name:
        var_vals.append(v)


# In[39]:


grad_w = tf.gradients(loss, l2_vars)
grad_R = tf.gradients(tf.multiply(alpha, X1_to_assign), l1_vars)


# In[40]:


reg_term_w = tf.gradients(reg_loss, l2_vars)
reg_term_R = tf.gradients(reg_loss, l1_vars)


for ix, val in enumerate(reg_term_w):
    if val is not None:
        grad_w[ix]+=val

for ix, val in enumerate(reg_term_R):
    if val is not None:
        grad_R[ix]+=val


# In[46]:


lr_ph = tf.placeholder(tf.float32)


# In[47]:


optimizer = {'w':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_w, l2_vars))),
            'R':tf.train.AdamOptimizer(learning_rate=lr_ph).apply_gradients(list(zip(grad_R, l1_vars)))}


# In[48]:


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# In[49]:


from sklearn.metrics import accuracy_score


# In[56]:


for epoch in range(400):

    if epoch == 0:
        sess.run(X1_assign_op, {drop_ph:0})

    sess.run(alpha_assign_op, {drop_ph:0, label_ph:y_train, mask_ph:train_mask})
    sess.run(optimizer['R'], {lr_ph:1e-3, drop_ph:0.7})
    sess.run(X1_assign_op, {drop_ph:0})
    _, lv = sess.run([optimizer['w'], loss], {lr_ph:1e-3, drop_ph:0.7, label_ph:y_train, mask_ph:train_mask})
    full_preds = sess.run(preds, {drop_ph:0})
    tr_preds, va_preds, te_preds = full_preds[train_mask], full_preds[val_mask], full_preds[test_mask]

    tr_acc = accuracy_score(y_train.argmax(1)[train_mask], tr_preds.argmax(1))
    valid_acc = accuracy_score(y_val.argmax(1)[val_mask], va_preds.argmax(1))
    test_acc = accuracy_score(y_test.argmax(1)[test_mask], te_preds.argmax(1))
    print(f"Epoch: {epoch} Train: {tr_acc}    Valid: {valid_acc}  Test: {test_acc}")
    print("Loss:", lv)

