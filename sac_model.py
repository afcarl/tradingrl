import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from net import *

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)


def gaussian_entropy(log_std):
    return tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1, name="entropy")


def gaussian_likelihood(input_, mu_, log_std):
    pre_sum = -0.5 * (((input_ - mu_) / (tf.exp(log_std) + EPS))
                      ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def clip_but_pass_gradient(input_, lower=-1., upper=1.):
    clip_up = tf.cast(input_ > upper, tf.float32)
    clip_low = tf.cast(input_ < lower, tf.float32)
    return input_ + tf.stop_gradient((upper - input_) * clip_up + (lower - input_) * clip_low)


def apply_squashing_func(mu_, pi_, logp_pi):
    # Squash the output
    deterministic_policy = tf.nn.softmax(mu_)
    policy = tf.nn.softmax(pi_, name="logits")
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    # logp_pi -= tf.reduce_sum(tf.log(1 - policy ** 2 + EPS),
                            #  axis=1, name="logp_pi")
    return deterministic_policy, policy, logp_pi

def rcnn(X, initial_state):
    cnn1 = tf.keras.layers.Conv1D(12, 2, padding="causal", activation=tf.nn.relu)(X)
    cnn2 = tf.keras.layers.Conv1D(12, 4, padding="causal", activation=tf.nn.relu)(X)
    cnn3 = tf.keras.layers.Conv1D(12, 8, padding="causal", activation=tf.nn.relu)(X)

    feed = tf.keras.layers.Concatenate()([cnn1,cnn2,cnn3])
    # initial_state=initial_state
    feed = tf.keras.layers.GRU(128, return_state=False,name="gru")(feed)
    
    # last_state = feed[-1]
    # last_state = tf.identity(last_state,"last_state")
    # feed = feed[0]

    return feed, cnn3

def cnn(X, initial_state):
    cnn1 = tf.keras.layers.Conv1D(12,2,padding="causal",name="cnn1")(X)
    cnn1 = tf.contrib.layers.layer_norm(cnn1)
    cnn1 = tf.nn.relu(cnn1)
    cnn2 = tf.keras.layers.Conv1D(24,2,padding="causal")(X)
    cnn2 = tf.contrib.layers.layer_norm(cnn2)
    cnn2 = tf.nn.relu(cnn2)
    cnn3 = tf.keras.layers.Conv1D(36,2,padding="causal")(X)
    cnn3 = tf.contrib.layers.layer_norm(cnn3)
    cnn3 = tf.nn.relu(cnn3)
    cnn4 = tf.keras.layers.Conv1D(48,2,padding="causal")(X)
    cnn4 = tf.contrib.layers.layer_norm(cnn4)
    cnn4 = tf.nn.relu(cnn4)
    cnn5 = tf.keras.layers.Conv1D(60,2,padding="causal")(X)
    cnn5 = tf.contrib.layers.layer_norm(cnn5)
    cnn5 = tf.nn.relu(cnn5)
    feed = tf.keras.layers.concatenate([cnn1, cnn2, cnn3, cnn4, cnn5])
    feed = tf.keras.layers.Conv1D(512,2,padding="causal")(feed)
    feed = tf.contrib.layers.layer_norm(feed)
    feed = tf.nn.relu(feed)
    feed = tf.keras.layers.GRU(512, return_state=False,name="gru")(feed)
    feed = tf.contrib.layers.layer_norm(feed)

    return feed

class Actor_Critic():
    def __init__(self,layer_norm=True,noise=True):
        self.layer_norm = layer_norm
        self.noise = noise
        self.conv_net = rcnn

    def actor(self,obs,initial_state,output_size,name):
        LOG_STD_MAX = 2
        LOG_STD_MIN = -20
        with tf.variable_scope(name):
            feed, self.last_state = self.conv_net(obs,initial_state)
            dense = NoisyDenseFG if self.noise == True else tf.keras.layers.Dense

            tensor_action = mlp(dense, feed, 512, layer_norm=self.layer_norm)
            feed_action = dense(output_size)(tensor_action)
            tensor_validation = mlp(dense, feed, 512, layer_norm=self.layer_norm)
            feed_validation = dense(1)(tensor_validation)
            mu_ = feed_validation + tf.subtract(feed_action, tf.reduce_mean(feed_action, axis=1, keep_dims=True))
            log_std = mlp(dense, feed, 512, layer_norm=self.layer_norm)
            log_std = dense(output_size)(log_std)

            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

            self.std = std = tf.exp(log_std)
            # Reparameterization trick
            pi_ = mu_ + 0.2 * (tf.random_normal(tf.shape(mu_)) * std)
            logp_pi = gaussian_likelihood(pi_, mu_, log_std)
            self.entropy = gaussian_entropy(log_std)
            # MISSING: reg params for log and mu
            # Apply squashing and account for it in the probabilty
            deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        return deterministic_policy, policy, logp_pi, self.entropy, self.last_state

    def critic(self, obs, initial_state, action=None, create_vf=True, create_qf=True,name="critic"):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            self.qf1, self.qf2, self.value_fn = None,None,None
            feed,_ = self.conv_net(obs,initial_state)
            dense = NoisyDenseFG if self.noise == True else tf.keras.layers.Dense

            if create_vf:
                # vf_h = tf.keras.layers.Concatenate()([feed,total_reward])
                vf_h = mlp(dense, feed, 512, layer_norm=self.layer_norm)
                vf_h = mlp(dense, vf_h, 512, layer_norm=self.layer_norm)
                self.value_fn = dense(1, name="value_fn")(vf_h)
                self.value_fn = tf.identity(self.value_fn, "value_fn")

            if create_qf:
                qf_h = tf.keras.layers.Concatenate()([feed, action])
                qf1_h = mlp(dense, qf_h, 512, layer_norm=self.layer_norm)
                qf1_h = mlp(dense, qf1_h, 512, layer_norm=self.layer_norm)
                self.qf1 = dense(1, name="qf1")(qf1_h)
                self.qf1 = tf.identity(self.qf1, "qf1")
                qf2_h = mlp(dense, qf_h, 512, layer_norm=self.layer_norm)
                qf2_h = mlp(dense, qf2_h, 512, layer_norm=self.layer_norm)
                self.qf2 = dense(1, name="qf2")(qf2_h)
                self.qf2 = tf.identity(self.qf2, "qf2")

        return self.qf1, self.qf2, self.value_fn
