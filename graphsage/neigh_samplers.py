from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import numpy as np
import pdb
"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs

        # retrieve matrix of [numofids, degree(128)]
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        # shuffling along degree axis 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        # pick [numofids, num_samples]
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists


class MLNeighborSampler(Layer):

    def __init__(self, adj_info, features, **kwargs):
        super(MLNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
        self.batch_size = FLAGS.max_degree
        self.node_dim = features.shape[1]
        self.reuse = False 

    def _call(self, inputs):
       
        #import pdb
        #pdb.set_trace()
        ids, num_samples = inputs

        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        vert_num = ids.shape[0]
        neig_num = self.adj_info.shape[1]


        
        # build model 
        # l = W*x1
        # l = relu(l*x2^t)
        with tf.variable_scope("MLsampler"):

            #bias = tf.Variable(tf.zeros([1]), trainable=False, name='bias')

            #vert = tf.gather(self.features, indices=ids, axis=0)
            #neig = tf.gather(self.features, indices=tf.reshape(adj_lists, [-1]), axis=0)
            
            v_f = tf.nn.embedding_lookup(self.features, ids)
            n_f = tf.nn.embedding_lookup(self.features, tf.reshape(adj_lists, [-1]))
           

            n_f = tf.reshape(n_f, shape=[-1, neig_num, self.node_dim])

            l = tf.layers.dense(v_f, self.node_dim, activation=None, trainable=False, reuse=self.reuse, name='dense')
            #l = tf.nn.relu(l, name='relu')
            
            #l = tf.layers.dense(l, self.node_dim, activation=None, trainable=False, reuse=self.reuse, name='dense2')
            #l = tf.matmul(tf.expand_dims(l, axis=1), neig, transpose_b=True, name='matmul')
            #l = tf.nn.relu(l, name='relu') 
            
            l = tf.expand_dims(l, axis=1)
            l = tf.matmul(l, n_f, transpose_b=True, name='matmul') 
           
            #l = tf.nn.bias_add(l, tf.tile(bias, [neig_num]))
            out = tf.nn.relu(l, name='relu')
            out = tf.squeeze(out)
            #out = tf.squeeze(l)

            # replace zeros to large number
            #idxz = tf.where(!tf.equal(out,0))

            condition = tf.equal(out, 0.0)
            case_true = tf.multiply(tf.ones(out.shape, tf.float32), 9999)
            case_false = out
            out = tf.where(condition, case_true, case_false)
    
            # sort (sort top k of negative estimated loss)
            out, idx_y = tf.nn.top_k(-out, num_samples)
            idx_y = tf.cast(idx_y, tf.int32)
           
            #import pdb
            #pdb.set_trace()
            x_ = np.zeros([vert_num, num_samples])
            for j in range(vert_num):
                x_[j,:] = j*np.ones([1, num_samples])
            
            idx_x = tf.Variable(x_, trainable=False, dtype=tf.int32)
            adj_ids = tf.nn.embedding_lookup(self.adj_info, ids)
            adj_lists = tf.gather_nd(adj_ids, tf.stack([tf.reshape(idx_x,[-1]), tf.reshape(idx_y,[-1])], axis=1))
            
            adj_lists = tf.reshape(adj_lists, [vert_num, num_samples])

            self.reuse = True


        return adj_lists 




