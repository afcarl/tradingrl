import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from net import *
from memory import *
from reward import *


def rcnn(X, initial_state):
    cnn1 = tf.keras.layers.Conv1D(32, 3, padding="causal")(X)
    cnn1 = tf.contrib.layers.layer_norm(cnn1)
    cnn1 = tf.nn.relu(cnn1)
    cnn2 = tf.keras.layers.Conv1D(64, 3, padding="causal")(cnn1)
    cnn2 = tf.contrib.layers.layer_norm(cnn2)
    cnn2 = tf.nn.relu(cnn2)
    cnn3 = tf.keras.layers.Conv1D(128, 3, padding="causal")(cnn2)
    cnn3 = tf.contrib.layers.layer_norm(cnn3)
    cnn3 = tf.nn.relu(cnn3)

    cnn4 = tf.keras.layers.Conv1D(32+64+128, 1, padding="same")(X)
    # cnn4 = tf.contrib.layers.layer_norm(cnn4)
    cnn4 = tf.nn.relu(cnn4)

    feed = tf.keras.layers.Concatenate()([cnn1, cnn2, cnn3])
    feed = tf.keras.layers.Add()([feed, cnn4])
    feed = tf.keras.layers.Conv1D(128, 3, padding="causal")(feed)
    feed = tf.keras.layers.Flatten()(feed)
    # feed = tf.keras.layers.LSTM(128, return_state=False, name="lstm")(feed)

    return feed


class Actor_Critic():
    def __init__(self,layer_norm=False,noise=False):
        self.layer_norm = layer_norm
        self.noise = noise
        self.conv_net = rcnn

    def actor(self,obs,initial_state,output_size,name="actor"):
        with tf.variable_scope(name):
            feed = self.conv_net(obs,initial_state)
            dense = NoisyDenseFG if self.noise == True else tf.keras.layers.Dense

            tensor_action, tensor_validation = tf.split(feed, 2, 1)
            feed_action = tf.layers.dense(tensor_action, output_size)
            feed_validation = tf.layers.dense(tensor_validation, 1)
            self.logits = feed_validation + tf.subtract(feed_action,
                                                        tf.reduce_mean(feed_action, axis=1, keep_dims=True))
            # self.logits = tf.tanh(self.logits)
            return self.logits
    
    def critic(self, obs, initial_state, action, reward,name="critic"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            feed = self.conv_net(obs,initial_state)
            dense = NoisyDenseFG if self.noise == True else tf.keras.layers.Dense

            qf_h = tf.keras.layers.Concatenate()([feed, action, reward])
            qf_h = tf.keras.layers.Dense(128,activation="relu")(qf_h)

            attention_qf1 = tf.keras.layers.Dense(128,activation="softmax")(qf_h)
            mul_qf1 = tf.keras.layers.Multiply()([qf_h,attention_qf1])
            attention_qf1 = tf.keras.layers.Dense(128)(mul_qf1)
            self.qf1 = dense(1, name="qf1")(attention_qf1)

            attention_qf2 = tf.keras.layers.Dense(128,activation="softmax")(qf_h)
            mul_qf2 = tf.keras.layers.Multiply()([qf_h,attention_qf2])
            attention_qf2 = tf.keras.layers.Dense(128)(mul_qf2)
            self.qf2 = dense(1, name="qf1")(attention_qf2)

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
