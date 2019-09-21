import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from net import *
from memory import *
from reward import *


def rcnn(X, initial_state):
    cnn1 = tf.keras.layers.Conv1D(128, 2, padding="same")(X)
    cnn1 = tf.contrib.layers.layer_norm(cnn1)
    cnn1 = tf.nn.relu(cnn1)
    cnn2 = tf.keras.layers.Conv1D(128, 4, padding="same")(X)
    cnn2 = tf.contrib.layers.layer_norm(cnn2)
    cnn2 = tf.nn.relu(cnn2)
    cnn3 = tf.keras.layers.Conv1D(128, 8, padding="same")(X)
    cnn3 = tf.contrib.layers.layer_norm(cnn3)
    cnn3 = tf.nn.relu(cnn3)

    cnn4 = tf.keras.layers.Conv1D(128*3, 1, padding="same")(X)
    cnn4 = tf.contrib.layers.layer_norm(cnn4)
    cnn4 = tf.nn.relu(cnn4)

    feed = tf.keras.layers.Concatenate()([cnn1, cnn2, cnn3])
    feed = tf.keras.layers.Add()([feed, cnn4])
    feed = tf.keras.layers.LSTM(128, return_state=False, name="lstm")(feed)

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

            tensor_action = mlp(dense, feed, 512, layer_norm=self.layer_norm)
            feed_action = dense(output_size)(tensor_action)
            tensor_validation = mlp(dense, feed, 512, layer_norm=self.layer_norm)
            feed_validation = dense(1)(tensor_validation)
            mu_ = feed_validation + tf.subtract(feed_action, tf.reduce_mean(feed_action, axis=1, keep_dims=True))
            self.logits = tf.tanh(mu_,name="logits")

            return self.logits
    
    def critic(self, obs, initial_state, action,name="critic"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            feed = self.conv_net(obs,initial_state)
            dense = NoisyDenseFG if self.noise == True else tf.keras.layers.Dense

            qf_h = tf.keras.layers.Concatenate()([feed, action])
            qf1_h = mlp(dense, qf_h, 128, layer_norm=self.layer_norm)
            qf1_h = mlp(dense, qf1_h, 128, layer_norm=self.layer_norm)
            self.qf1 = dense(1, name="qf1")(qf1_h)
            self.qf1 = tf.identity(self.qf1, "qf1")
            qf2_h = mlp(dense, qf_h, 128, layer_norm=self.layer_norm)
            qf2_h = mlp(dense, qf2_h, 128, layer_norm=self.layer_norm)
            self.qf2 = dense(1, name="qf2")(qf2_h)
            self.qf2 = tf.identity(self.qf2, "qf2")

            return self.qf1, self.qf2

