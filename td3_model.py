import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from net import *
from memory import *
from reward import *

def rcnn(X, initial_state, k):
    cnn1 = tf.keras.layers.Conv1D(12, k, padding="causal")(X)
    cnn1 = tf.contrib.layers.layer_norm(cnn1)
    cnn1 = tf.nn.relu(cnn1)
    cnn2 = tf.keras.layers.Conv1D(32, k, padding="causal")(cnn1)
    cnn2 = tf.contrib.layers.layer_norm(cnn2)
    cnn2 = tf.nn.relu(cnn2)
    cnn3 = tf.keras.layers.Conv1D(64, k, padding="causal")(cnn2)
    cnn3 = tf.contrib.layers.layer_norm(cnn3)
    cnn3 = tf.nn.relu(cnn3)

    cnn4 = tf.keras.layers.Conv1D((12+32+64), 1, padding="causal")(X)
    cnn4 = tf.contrib.layers.layer_norm(cnn4)
    cnn4 = tf.nn.relu(cnn4)

    feed = tf.keras.layers.Concatenate()([cnn1, cnn2, cnn3])
    feed = tf.keras.layers.Add()([feed, cnn4])
    feed = tf.nn.relu(feed)
    # feed = tf.keras.layers.MaxPool1D()(feed)

    return feed

def cnn_net(x,init_state):
    cnn1 = tf.keras.layers.Conv1D(64, 2, padding="causal")(x)
    cnn1 = tf.contrib.layers.layer_norm(cnn1)
    cnn1 = tf.nn.relu(cnn1)
    cnn1 = tf.keras.layers.Conv1D(128, 2, padding="causal")(cnn1)
    cnn1 = tf.contrib.layers.layer_norm(cnn1)
    cnn1 = tf.nn.relu(cnn1)
    cnn1 = tf.keras.layers.Conv1D(256, 2, padding="causal")(cnn1)
    cnn1 = tf.contrib.layers.layer_norm(cnn1)
    cnn1 = tf.nn.relu(cnn1)

    cnn2 = tf.keras.layers.Conv1D(128, 4, padding="causal")(x)
    cnn2 = tf.contrib.layers.layer_norm(cnn2)
    cnn2 = tf.nn.relu(cnn2)
    cnn2 = tf.keras.layers.Conv1D(256, 4, padding="causal")(cnn2)
    cnn2 = tf.contrib.layers.layer_norm(cnn2)
    cnn2 = tf.nn.relu(cnn2)

    cnn3 = tf.keras.layers.Conv1D(256, 8, padding="causal")(x)
    cnn3 = tf.contrib.layers.layer_norm(cnn3)
    cnn3 = tf.nn.relu(cnn3)

    concat = tf.keras.layers.Concatenate()([cnn1,cnn2,cnn3])
    cnn = tf.keras.layers.Conv1D(int(x.shape[-1]), 3, padding="causal")(concat)
    cnn = tf.contrib.layers.layer_norm(cnn)
    cnn = tf.nn.relu(cnn)

    add = tf.keras.layers.Add()([cnn,x])
    x = tf.nn.relu(add)
    
    # x = tf.keras.layers.GlobalMaxPool1D()(x)

    return x

def cnn(x,init_state):
    x = rcnn(x,init_state,2)
    x = rcnn(x,init_state,2)
    x = rcnn(x,init_state,2)
    x = rcnn(x,init_state,2)
    x = rcnn(x,init_state,2)
    x = tf.keras.layers.Flatten()(x)
    return x

def cnn2(x,init_state):
    x = cnn_net(x, init_state)
    x = cnn_net(x, init_state)
    x = cnn_net(x, init_state)
    x = tf.keras.layers.Flatten()(x)
    return x

def atenttion(x,output_size,activation=None):
    dense = NoisyDenseFG
    shape = int(x.shape[-1])
    atenttion = mlp(dense, x, shape, activ_fn=tf.nn.softmax, layer_norm=False)
    mul = tf.keras.layers.Multiply()([x, atenttion])

    atenttion = mlp(dense, x, shape*2, activ_fn=None, layer_norm=False)

    x = mlp(dense, atenttion, output_size, activ_fn=None, layer_norm=False)

    return x

def atenttion2(x, output_size, activation=None):
    dense = NoisyDenseFG
    shape = int(x.shape[-1])
    atenttion = mlp(dense, x, shape, activ_fn=tf.nn.softmax, layer_norm=False)
    mul = tf.keras.layers.Multiply()([x, atenttion])

    atenttion = mlp(dense, x, shape*2, activ_fn=None, layer_norm=False)

    tensor_action, tensor_validation = tf.split(atenttion, 2, 1)
    feed_action = dense(output_size)(tensor_action)
    feed_validation = dense(1)(tensor_validation)

    x = feed_validation + tf.subtract(feed_action,
                                           tf.reduce_mean(feed_action, axis=1, keep_dims=True))

    return x

class Actor_Critic():
    def __init__(self,layer_norm=False,noise=True):
        self.layer_norm = layer_norm
        self.noise = noise
        self.conv_net = cnn2

    def actor(self,obs,initial_state,output_size,name="actor"):
        with tf.variable_scope(name):
            feed = self.conv_net(obs,initial_state)
            dense = NoisyDenseFG if self.noise == True else tf.keras.layers.Dense
            self.logits = atenttion2(feed,output_size)
            self.logits = tf.tanh(self.logits)
            return self.logits
    
    def critic(self, obs, initial_state, action, reward,name="critic"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            feed = self.conv_net(obs,initial_state)
            dense = NoisyDenseFG if self.noise == True else tf.keras.layers.Dense

            qf_h = tf.keras.layers.Concatenate()([feed, action])

            self.qf1 = atenttion(qf_h, 1)
            self.qf2 = atenttion(qf_h, 1)

            return self.qf1, self.qf2


# class Actor_Critic():
#     def __init__(self,layer_norm=False,noise=False):
#         self.layer_norm = layer_norm
#         self.noise = noise
#         self.conv_net = rcnn

#     def actor(self,obs,initial_state,output_size,name="actor"):
#         with tf.variable_scope(name):
#             obs = tf.keras.layers.Flatten()(obs)
#             feed_actor = tf.keras.layers.Dense(128, activation=tf.nn.relu)(obs)
#             tensor_action, tensor_validation = tf.split(feed_actor, 2, 1)
#             feed_action = tf.layers.dense(tensor_action, output_size)
#             feed_validation = tf.layers.dense(tensor_validation, 1)
#             self.logits = feed_validation + tf.subtract(feed_action,
#                                                         tf.reduce_mean(feed_action, axis=1, keep_dims=True))
#             self.logits = tf.tanh(self.logits)
#             return self.logits
    
#     def critic(self, obs, initial_state, action,name="critic"):
#         with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#             obs = tf.keras.layers.Flatten()(obs)
#             feed = tf.keras.layers.Dense(128, activation=tf.nn.relu)(obs)
#             dense = NoisyDenseFG if self.noise == True else tf.keras.layers.Dense

#             qf_h = tf.keras.layers.Concatenate()([feed, action])
#             qf1_h = mlp(dense, qf_h, 128, layer_norm=self.layer_norm)
#             qf1_h = mlp(dense, qf1_h, 128, layer_norm=self.layer_norm)
#             self.qf1 = dense(1, name="qf1")(qf1_h)
#             self.qf1 = tf.identity(self.qf1, "qf1")
#             qf2_h = mlp(dense, qf_h, 128, layer_norm=self.layer_norm)
#             qf2_h = mlp(dense, qf2_h, 128, layer_norm=self.layer_norm)
#             self.qf2 = dense(1, name="qf2")(qf2_h)
#             self.qf2 = tf.identity(self.qf2, "qf2")

#             return self.qf1, self.qf2
