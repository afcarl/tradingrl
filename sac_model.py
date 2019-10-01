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
    deterministic_policy = tf.tanh(mu_)
    policy = tf.tanh(pi_, name="logits")
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    # logp_pi -= tf.reduce_sum(tf.log(1 - policy ** 2 + EPS),
                            #  axis=1, name="logp_pi")
    return deterministic_policy, policy, logp_pi

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

def atenttion(dense,x,output_size,activation=None):
    shape = int(x.shape[-1])
    atenttion = mlp(dense, x, shape, activ_fn=tf.nn.softmax, layer_norm=False)
    mul = tf.keras.layers.Multiply()([x, atenttion])

    atenttion = mlp(dense, x, shape*2, activ_fn=None, layer_norm=False)

    x = mlp(dense, atenttion, output_size, activ_fn=None, layer_norm=False)

    return x

def atenttion2(dense, x, output_size, activation=None):
    shape = int(x.shape[-1])
    atenttion = mlp(dense, x, shape, activ_fn=tf.nn.softmax, layer_norm=False)
    mul = tf.keras.layers.Multiply()([x, atenttion])

    atenttion = mlp(dense, x, shape*2, activ_fn=None, layer_norm=False)

    tensor_action, tensor_validation = tf.split(atenttion, 2, 1)
    feed_action = dense(output_size)(tensor_action)
    feed_validation = dense(1)(tensor_validation)

    x = feed_validation + tf.subtract(feed_action,
                                           tf.reduce_mean(feed_action, axis=1, keep_dims=True))
    x2 = dense(output_size)(atenttion)

    return x,x2

class Actor_Critic():
    def __init__(self,layer_norm=True,noise=False):
        self.layer_norm = layer_norm
        self.noise = noise
        self.conv_net = cnn2

    def actor(self,obs,initial_state,output_size,name):
        LOG_STD_MAX = 2
        LOG_STD_MIN = -20
        with tf.variable_scope(name):
            feed = self.conv_net(obs,initial_state)
            dense = NoisyDenseFG if self.noise == True else tf.keras.layers.Dense

            mu_, log_std = atenttion2(dense, feed, output_size)

            # mu_ = tf.clip_by_value(mu_, LOG_STD_MIN, LOG_STD_MAX)
            # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

            self.std = std = tf.exp(log_std)
            # Reparameterization trick
            pi_ = mu_ +  (tf.random_normal(tf.shape(mu_)) * std)
            logp_pi = gaussian_likelihood(pi_, mu_, log_std)
            self.entropy = gaussian_entropy(log_std)
            # MISSING: reg params for log and mu
            # Apply squashing and account for it in the probabilty
            deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        return deterministic_policy, policy, logp_pi, self.entropy

    def critic(self, obs, initial_state, action=None, create_vf=True, create_qf=True,name="critic"):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            self.qf1, self.qf2, self.value_fn = None,None,None
            feed = self.conv_net(obs,initial_state)
            dense = NoisyDenseFG if self.noise == True else tf.keras.layers.Dense

            if create_vf:
                # vf_h = tf.keras.layers.Concatenate()([feed,total_reward])
                self.value_fn = atenttion(dense, feed,1)

            if create_qf:
                qf_h = tf.keras.layers.Concatenate()([feed,action])
                self.qf1 = atenttion(dense, qf_h, 1)
                self.qf2 = atenttion(dense, qf_h, 1)

        return self.qf1, self.qf2, self.value_fn
