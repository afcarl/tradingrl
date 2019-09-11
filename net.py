
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops.init_ops import Constant




######################################################################################################################################################

# 参照:https://github.com/spring01/drlbox/blob/master/drlbox/layer/noisy_dense.py
class NoisyDense(tf.keras.layers.Dense):

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        kernel_shape = [input_shape[-1].value, self.units]
        kernel_quiet = self.add_variable('kernel_quiet',
                                         shape=kernel_shape,
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint,
                                         dtype=self.dtype,
                                         trainable=True)
        scale_init = Constant(value=(0.5 / np.sqrt(kernel_shape[0])))
        kernel_noise_scale = self.add_variable('kernel_noise_scale',
                                               shape=kernel_shape,
                                               initializer=scale_init,
                                               dtype=self.dtype,
                                               trainable=True)
        kernel_noise = self.make_kernel_noise(shape=kernel_shape)
        self.kernel = kernel_quiet + kernel_noise_scale * kernel_noise
        if self.use_bias:
            bias_shape = [self.units,]
            bias_quiet = self.add_variable('bias_quiet',
                                           shape=bias_shape,
                                           initializer=self.bias_initializer,
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint,
                                           dtype=self.dtype,
                                           trainable=True)
            bias_noise_scale = self.add_variable(name='bias_noise_scale',
                                                 shape=bias_shape,
                                                 initializer=scale_init,
                                                 dtype=self.dtype,
                                                 trainable=True)
            bias_noise = self.make_bias_noise(shape=bias_shape)
            self.bias = bias_quiet + bias_noise_scale * bias_noise
        else:
            self.bias = None
        self.built = True

    def make_kernel_noise(self, shape):
        raise NotImplementedError

    def make_bias_noise(self, shape):
        raise NotImplementedError


'''
Noisy dense layer with independent Gaussian noise
distributed  A3Cに用いる。
'''
class NoisyDenseIG(NoisyDense):

    def make_kernel_noise(self, shape):
        noise = tf.random_normal(shape, dtype=self.dtype)
        kernel_noise = tf.Variable(noise, trainable=False, dtype=self.dtype)
        self.noise_list = [kernel_noise]
        return kernel_noise

    def make_bias_noise(self, shape):
        noise = tf.random_normal(shape, dtype=self.dtype)
        bias_noise = tf.Variable(noise, trainable=False, dtype=self.dtype)
        self.noise_list.append(bias_noise)
        return bias_noise


'''
Noisy dense layer with factorized Gaussian noise
DQN and Duelling に用いる
'''
class NoisyDenseFG(NoisyDense):

    def make_kernel_noise(self, shape):
        kernel_noise_input = self.make_fg_noise(shape=[shape[0]])
        kernel_noise_output = self.make_fg_noise(shape=[shape[1]])
        self.noise_list = [kernel_noise_input, kernel_noise_output]
        kernel_noise = kernel_noise_input[:, tf.newaxis] * kernel_noise_output
        return kernel_noise

    def make_bias_noise(self, shape):
        return self.noise_list[1] # kernel_noise_output

    def make_fg_noise(self, shape):
        noise = tf.random_normal(shape, dtype=self.dtype)
        trans_noise = tf.sign(noise) * tf.sqrt(tf.abs(noise))
        return tf.Variable(trans_noise, trainable=False, dtype=self.dtype)

'''
We now turn to explicit instances of the noise distributions for linear layers in a noisy network.
We explore two options: Independent Gaussian noise, which uses an independent Gaussian noise  entry per weight and Factorised Gaussian noise, 
which uses an independent noise per each output  and another independent noise per each input.
The main reason to use factorised Gaussian noise is  to reduce the compute time of random number generation in our algorithms.
This computational  overhead is especially prohibitive in the case of single-thread agents such as DQN and Duelling.
For  this reason we use factorised noise for DQN and Duelling and independent noise for the distributed  A3C,
for which the compute time is not a major concern.

from https://arxiv.org/pdf/1706.10295.pdf
'''

######################################################################################################################################################

def exploration(prediction,output_size,e = 0.2,tau=1.0):
  prediction = prediction.astype("float64")

#   np.random.seed(1)
  if np.random.rand() < e:
    clip=(-25.0, 25.0)
    exp_values = np.exp(np.clip(prediction / tau, clip[0], clip[1]))
    probs = exp_values / np.sum(exp_values)
    action = np.random.choice(range(output_size), p=probs)
  else:
    action = np.argmax(prediction)

  return action

######################################################################################################################################################

def mlp(dense,input_ph, layers, activ_fn=tf.nn.relu, layer_norm=False):
    output = input_ph
    output = dense(layers)(input_ph)
    if layer_norm:
      output = tf.contrib.layers.layer_norm(output)
    output = activ_fn(output)
    return output

######################################################################################################################################################


def get_trainable_vars(name):
    """
    returns the trainable variables
    :param name: (str) the scope
    :return: ([TensorFlow Variable])
    """
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

def get_vars(scope):
    """
    Alias for get_trainable_vars
    :param scope: (str)
    :return: [tf Variable]
    """
    return get_trainable_vars(scope)

######################################################################################################################################################

def initialize_uninitialized_vars(sess):
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
    
class ICM():
    def __init__(self,input_size,output_size,lr,scope="icm"):
        self.window_size = input_size[1]
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, input_size,name="state")
            self.next_state = tf.placeholder(tf.float32, input_size,name="next_state")
            self.action = tf.placeholder(tf.int32, (None, 1))

            state_f = self.conv(self.state,"state")
            next_state_f = self.conv(self.next_state,"next_state")

            g = tf.concat([state_f, next_state_f], 1)
            g = tf.keras.layers.Dense(256,activation=tf.nn.elu)(g)
            action_hat = tf.keras.layers.Dense(output_size,activation=tf.nn.softmax)(g)

            action_one_hot = tf.one_hot(self.action, output_size, dtype=tf.float32)
            f = tf.keras.layers.Concatenate()([action_hat, tf.dtypes.cast(self.action,tf.float32)])
            f = tf.keras.layers.Dense(256,activation=tf.nn.elu)(f)
            predicted_state = tf.keras.layers.Dense(self.window_size)(f)

            inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=action_hat, labels=action_one_hot), name='inverse_loss')
            forward_loss = 0.5 * tf.reduce_mean(
                tf.square(tf.subtract(predicted_state, next_state_f)), name='forward_loss')
            self.ri = 2 * (forward_loss * 2)
            icm_loss = 20.0 * (0.8 * inverse_loss + 0.2 * forward_loss)

        icm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model/"+scope)
        icm_gradients, _ = tf.clip_by_global_norm(tf.gradients(icm_loss, icm_vars), 40.0)
        self.optimizer = tf.train.AdamOptimizer(lr).apply_gradients(zip(icm_gradients, icm_vars))

    def conv(self,X,scope):
        with tf.variable_scope(scope):
            cnn1 = tf.keras.layers.Conv1D(12,2,padding="causal",activation=tf.nn.relu)(X)
            cnn1 = tf.keras.layers.AlphaDropout(0.3,seed=2)(cnn1)
            cnn2 = tf.keras.layers.Conv1D(24,2,padding="causal",activation=tf.nn.relu)(X)
            cnn2 = tf.keras.layers.AlphaDropout(0.3,seed=2)(cnn2)
            cnn3 = tf.keras.layers.Conv1D(36,2,padding="causal",activation=tf.nn.relu)(X)
            cnn3 = tf.keras.layers.AlphaDropout(0.3,seed=2)(cnn3)
            cnn4 = tf.keras.layers.Conv1D(48,2,padding="causal",activation=tf.nn.relu)(X)
            cnn4 = tf.keras.layers.AlphaDropout(0.3,seed=2)(cnn4)
            cnn5 = tf.keras.layers.Conv1D(60,2,padding="causal",activation=tf.nn.relu)(X)
            cnn5 = tf.keras.layers.AlphaDropout(0.3,seed=2)(cnn5)
            feed = tf.keras.layers.concatenate([cnn1, cnn2, cnn3, cnn4, cnn5])
            feed = tf.keras.layers.Conv1D(512,2,padding="causal",activation=tf.nn.relu)(feed)
            feed = tf.keras.layers.GRU(self.window_size,name="gru")(feed)

            return feed

class ICM2():
    def __init__(self,input_size,output_size,lr,scope="icm"):
        self.window_size = input_size[1]
        self.flatten = input_size[-1]
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, input_size,name="state")
            self.next_state = tf.placeholder(tf.float32, input_size,name="next_state")
            self.action = tf.placeholder(tf.int32, (None, 1))

            state_f = self.conv(self.state,"state")
            next_state_f = self.conv(self.next_state,"state")

            g = tf.concat([state_f, next_state_f], 1)
            g = tf.keras.layers.Dense(256,activation=tf.nn.elu)(g)
            action_hat = tf.keras.layers.Dense(output_size,activation=tf.nn.softmax)(g)

            action_one_hot = tf.one_hot(self.action, output_size, dtype=tf.float32)
            f = tf.keras.layers.Concatenate()([action_hat, tf.dtypes.cast(self.action,tf.float32)])
            f = tf.keras.layers.Dense(256,activation=tf.nn.elu)(f)
            predicted_state = tf.keras.layers.Dense(self.window_size)(f)

            inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=action_hat, labels=action_one_hot), name='inverse_loss')
            forward_loss = 0.5 * tf.reduce_mean(
                tf.square(tf.subtract(predicted_state, next_state_f)), name='forward_loss')
            self.ri = (forward_loss * 2)
            icm_loss = 20.0 * (0.8 * inverse_loss + 0.2 * forward_loss)

        icm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model/"+scope)
        icm_gradients, _ = tf.clip_by_global_norm(tf.gradients(icm_loss, icm_vars), 40.0)
        self.optimizer = tf.train.AdamOptimizer(lr).apply_gradients(zip(icm_gradients, icm_vars))

    def conv(self,X,scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            flatten = tf.keras.layers.Flatten()(X)
            x = tf.keras.layers.Dense(self.window_size*self.flatten,activation=tf.nn.softmax)(flatten)
            mul_x = tf.keras.layers.Multiply()([flatten,x])
            x = tf.keras.layers.Dense(236,activation=tf.nn.elu)(mul_x)
            x = tf.keras.layers.Dense(self.window_size)(x)
            
            return x

# actionを加える？
class RND():
    def __init__(self,input_size,output_size,lr,name="rnd"):
        self.window_size = input_size[1]
        self.flatten_size = input_size[-1]
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, input_size, name="state")
            self.action = tf.placeholder(tf.float32, (None, output_size))
            feed,outputs = self.conv(self.state,"target")

            with tf.variable_scope("predictor"):
                x = tf.keras.layers.Dense(256,activation=tf.nn.elu)(feed)
                x = tf.keras.layers.Dense(self.window_size)(x)
                x2 = tf.keras.layers.Dense(output_size,activation=tf.nn.softmax)(x)
                x = tf.keras.layers.Concatenate()([x,x2])

            loss = 0.5 * tf.reduce_mean((outputs - x) ** 2)
            self.ri = loss
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(tf.reduce_mean(loss),
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model/"+name+"/predictor"))

    def conv(self,X,scope):
        with tf.variable_scope(scope):
            flatten = tf.keras.layers.Flatten()(X)
            x = tf.keras.layers.Dense(
                self.window_size * self.flatten_size, activation=tf.nn.softmax)(flatten)
            mul_x = tf.keras.layers.Multiply()([flatten,x])
            x = tf.keras.layers.Dense(236,activation=tf.nn.elu)(mul_x)
            x = tf.keras.layers.Dense(self.window_size)(x)
            outputs = tf.keras.layers.Concatenate()([x,self.action])
            
            return x,outputs

class RND2():
    def __init__(self,input_size,output_size,lr,scope="rnd2"):
        self.window_size = input_size[1]
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, input_size,name="state")
            self.next_state = tf.placeholder(tf.float32, input_size,name="next_state")
            self.action = tf.placeholder(tf.int32, (None, 1))

            with tf.variable_scope("target"):
                next_state_f = self.conv(self.next_state,"next_state")

            with tf.variable_scope("predicter"):
                state_f = self.conv(self.state, "state")
                g = tf.concat([state_f, next_state_f], 1)
                g = tf.keras.layers.Dense(256,activation=tf.nn.elu)(g)
                action_hat = tf.keras.layers.Dense(output_size,activation=tf.nn.softmax)(g)

                action_one_hot = tf.one_hot(self.action, output_size, dtype=tf.float32)
                f = tf.keras.layers.Concatenate()([action_hat, tf.dtypes.cast(self.action,tf.float32)])
                f = tf.keras.layers.Dense(256,activation=tf.nn.elu)(f)
                predicted_state = tf.keras.layers.Dense(self.window_size)(f)

            inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=action_hat, labels=action_one_hot), name='inverse_loss')
            forward_loss = 0.5 * tf.reduce_mean(
                tf.square(tf.subtract(predicted_state, next_state_f)), name='forward_loss')
            self.ri = 2 * (forward_loss * 2)
            icm_loss = 20.0 * (0.8 * inverse_loss + 0.2 * forward_loss)

        icm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope + "/predicter")
        icm_gradients, _ = tf.clip_by_global_norm(tf.gradients(icm_loss, icm_vars), 40.0)
        self.optimizer = tf.train.AdamOptimizer(lr).apply_gradients(zip(icm_gradients, icm_vars))

    def conv(self,X,name):
        with tf.variable_scope(name):
            cnn1 = tf.keras.layers.Conv1D(12,2,padding="causal",activation=tf.nn.relu)(X)
            cnn1 = tf.keras.layers.AlphaDropout(0.3,seed=2)(cnn1)
            cnn2 = tf.keras.layers.Conv1D(24,2,padding="causal",activation=tf.nn.relu)(X)
            cnn2 = tf.keras.layers.AlphaDropout(0.3,seed=2)(cnn2)
            cnn3 = tf.keras.layers.Conv1D(36,2,padding="causal",activation=tf.nn.relu)(X)
            cnn3 = tf.keras.layers.AlphaDropout(0.3,seed=2)(cnn3)
            cnn4 = tf.keras.layers.Conv1D(48,2,padding="causal",activation=tf.nn.relu)(X)
            cnn4 = tf.keras.layers.AlphaDropout(0.3,seed=2)(cnn4)
            cnn5 = tf.keras.layers.Conv1D(60,2,padding="causal",activation=tf.nn.relu)(X)
            cnn5 = tf.keras.layers.AlphaDropout(0.3,seed=2)(cnn5)
            feed = tf.keras.layers.concatenate([cnn1, cnn2, cnn3, cnn4, cnn5])
            feed = tf.keras.layers.Conv1D(512,2,padding="causal",activation=tf.nn.relu)(feed)
            feed = tf.keras.layers.GRU(self.window_size,name="gru")(feed)

            return feed


