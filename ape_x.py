from numba import njit
import tensorflow as tf
from IPython.display import clear_output
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from net import *
from memory import *
# from reward_ditect import *
from reward import *
from collections import deque
import random
from sac_model import *
import time
import logging
import shutil
import ta

####################################################################################################################################

class Actor:
    LEARNING_RATE = 1e-3
    GAMMA = 0.99

    def __init__(self, sess, path, window_size, num, STEP_SIZE, OUTPUT_SIZE, saver_path=None, restore=False, norm=False, ent_coef='auto', target_entropy='auto'):
        self.sess = sess
        self.path = path
        self.window_size = window_size
        self.num = num
        self.STEP_SIZE = STEP_SIZE
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.saver_path = saver_path
        self.rewards = reward3
        
        self.preproc()
        self.state_size = (None, self.window_size, self.df.shape[-1])
        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        tf.get_logger().setLevel(logging.ERROR)

        with tf.device('/cpu:0'):
            with tf.variable_scope("input"):
              self.policy_tf = Actor_Critic(norm)
              self.target_policy = Actor_Critic(norm)

              self.state = tf.placeholder(tf.float32, self.state_size)
              self.new_state = tf.placeholder(tf.float32, self.state_size)
              self.initial_state = tf.placeholder(tf.float32,(None,128))
              self.action = tf.placeholder(tf.float32,(None,self.OUTPUT_SIZE))
              self.reward = tf.placeholder(tf.float32,(None,1))
              self.done = tf.placeholder(tf.float32, (None, 1))

            with tf.variable_scope("model", reuse=False):
            #   self.deterministic_policy, self.policy_out, logp_pi, self.entropy, self.last_state
              self.deterministic_action, self.policy_out, logp_pi, self.entropy = self.policy_tf.actor(self.state,self.initial_state,self.OUTPUT_SIZE,"actor")
              qf1, qf2, value_fn = self.policy_tf.critic(self.state, self.initial_state, self.action, create_vf=True, create_qf=True,name="critic")
              qf1_pi, qf2_pi, _ = self.policy_tf.critic(self.state, self.initial_state, self.deterministic_action, create_vf=False, create_qf=True, name="critic")

            if self.target_entropy == 'auto':
                # automatically set target entropy if needed
                self.target_entropy = -np.prod(self.OUTPUT_SIZE).astype(np.float32)
            else:
                # Force conversion
                # this will also throw an error for unexpected string
                self.target_entropy = float(self.target_entropy)

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                # Default initial value of ent_coef when learned
                init_value = 1.0
                if '_' in self.ent_coef:
                    init_value = float(self.ent_coef.split('_')[1])
                    assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                    initializer=np.log(init_value).astype(np.float32))
                self.ent_coef = tf.exp(self.log_ent_coef)
            else:
                # Force conversion to float
                # this will throw an error if a malformed string (different from 'auto')
                # is passed
                self.ent_coef = float(self.ent_coef)

            with tf.variable_scope("target", reuse=False):
              _,_,target_vf = self.target_policy.critic(self.new_state,self.initial_state,self.action,create_qf=True, create_vf=True)
            with tf.variable_scope("loss"):
                min_qf_pi = tf.minimum(qf1_pi, qf2_pi)
                q_backup = tf.stop_gradient(self.reward + self.GAMMA * (1.0 - self.done) * target_vf)
                qf1_loss = tf.abs(0.5 * tf.reduce_mean((q_backup - qf1) ** 2))
                qf2_loss = tf.abs(0.5 * tf.reduce_mean((q_backup - qf2) ** 2))

                self.absolute_errors = tf.abs(q_backup - qf1)

                ent_coef_loss = -tf.reduce_mean(self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))

                policy_kl_loss = 0.5 * tf.reduce_mean((self.ent_coef * logp_pi - qf1_pi) ** 2)
                v_backup = tf.reduce_mean(min_qf_pi - self.ent_coef * logp_pi)
                value_loss = (0.5 * tf.reduce_mean((value_fn - v_backup) ** 2))
                self.values_losses = qf1_loss + qf2_loss + value_loss
                self.policy_loss = policy_kl_loss

            self.actor_optimizer = tf.train.AdamOptimizer(0.0001,name="actor_optimizer").minimize(self.policy_loss, var_list=get_vars('model/actor/'))
            self.vf_optimizer = tf.train.AdamOptimizer(0.0001,name="vf_optimizer").minimize(self.values_losses,var_list=get_vars("model/critic/"))
            self.entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE,name="entropy_optimizer").minimize(ent_coef_loss,var_list=self.log_ent_coef)
        
        self.saver = tf.train.Saver(tf.global_variables())
        self.saver_path = saver_path

        if restore == True:
            self.saver.restore(self.sess, self.saver_path)
            # self.sess.run(tf.initializers.variables(get_vars("model/actor/")))
        else:
          self.sess.run(tf.global_variables_initializer())

    def nstep(self,r):
        running_add = 0.0
        for t in range(len(r)):
            running_add += self.GAMMA * r[t]

        return running_add

    def discount_rewards(self, r, running_add):
        running_add = running_add * self.GAMMA + r
        return running_add

    def preproc(self):
        self.dat = df = pd.read_csv(self.path)
        s = np.asanyarray(ta.stoch(df["High"],df["Low"],df["Close"],14)).reshape((-1, 1)) - np.asanyarray(ta.stoch_signal(df["High"],df["Low"],df["Close"],14)).reshape((-1, 1))
        m = np.asanyarray(ta.macd(df["Close"])).reshape((-1, 1)) - np.asanyarray(ta.macd_signal(df["Close"])).reshape((-1, 1))
        trend3 = np.asanyarray(self.dat[["Close"]]) - np.asanyarray(ta.ema(self.dat["Close"],20)).reshape((-1, 1))
        cross1 = np.asanyarray(ta.ema(self.dat["Close"],20)).reshape((-1, 1)) - np.asanyarray(ta.ema(self.dat["Close"],5)).reshape((-1, 1))
        y = np.asanyarray(self.dat[["Open"]])
        x = np.concatenate([s,m], 1)

        gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x, y, self.window_size)
        self.x = []
        self.y = []
        for i in gen:
            self.x.extend(i[0].tolist())
            self.y.extend(i[1].tolist())
        self.x = np.asanyarray(self.x)#.reshape((-1, self.window_size, x.shape[-1]))
        self.y = np.asanyarray(self.y)

        self.df = self.x[-self.STEP_SIZE::]
        self.trend = self.y[-self.STEP_SIZE::]

    def _select_action(self, state, next_state=None):
        # self.policy_out
        out = self.deterministic_action if self.num == 0 else self.policy_out
        prediction = self.sess.run(out, feed_dict={self.state: [state], self.initial_state: self.init_value})[0]
        action = np.argmax(prediction)
        self.pred = prediction

        return action

    def _memorize(self, state, action, reward, new_state, dead, o):
        self.MEMORIES.append((state, action, reward, new_state, dead, o))

    def _construct(self,replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        init_values = np.array([a[-1] for a in replay])
        actions = np.array([a[1] for a in replay]).reshape((-1, self.OUTPUT_SIZE))
        rewards = np.array([a[2] for a in replay]).reshape((-1, 1))
        done = np.array([a[-2] for a in replay]).reshape((-1, 1))

        absolute_errors = self.sess.run(self.absolute_errors,
                                feed_dict={self.state:states,self.new_state: new_states, self.action: actions, self.done:done,
                                           self.reward: rewards,self.initial_state:init_values})
        return absolute_errors

    def prob(self,history):
        prob = np.asanyarray(history)
        a = np.mean(prob == 0)
        b = np.mean(prob == 1)
        c = 1 - (a + b)
        prob = [a,b,c]
        return prob

    def run(self,count, queues, spread, pip_cost, los_cut, day_pip,iterations=1000000, n=10,step=100):
        spread = spread / pip_cost
        self.rand = np.random.RandomState()
        lc = los_cut / pip_cost
        for i in range(iterations):
            if (i - 1) % step == 0:
                h = self.rand.randint(self.x.shape[0]-(self.STEP_SIZE+1))
                self.df = self.x[h:h+self.STEP_SIZE]
                self.trend = self.y[h:h+self.STEP_SIZE]
            done = 0.0
            position = 3
            pip = []
            provisional_pip = []
            running_add = 0.0
            total_pip = 0.0
            old_reword = 0.0
            states = []
            h_a = []
            h_r = []
            h_p = []
            self.init_value = np.zeros(128).reshape((1,-1))
            old = np.asanyarray(0)
            self.history = []
            self.MEMORIES = deque()
            for t in  range(0, len(self.trend)-1):
                action = self._select_action(self.df[t])
                h_a.append(self.pred)
                self.history.append(action)
                
                states,provisional_pip,position,total_pip = self.rewards(self.trend[t],pip,provisional_pip,action,position,states,pip_cost,spread,total_pip)
                h_p.append(position)
                reward =  total_pip - old_reword
                old_reword = total_pip
                h_r.append(reward)

                running_add = self.discount_rewards(reward,running_add)
                if t == len(self.trend)-1:
                    done = 1.0
                self._memorize(self.df[t], h_a[t], running_add*10, self.df[t+1], done, self.init_value[0])
            # for t in range(0, len(self.trend)-1):
            #     tau = t - n + 1
            #     if tau >= 0:
            #       rewards = self.nstep(h_r[tau+1:tau+n])
            #       self._memorize(self.df[tau], h_a[tau], rewards*10, self.df[t+1], done, self.init_value[0])

            batch_size = len(self.MEMORIES) #if self.num == 0 else int(len(self.MEMORIES) / 2)
            replay = random.sample(self.MEMORIES, batch_size)
            ae = np.asanyarray(self._construct(replay)).reshape((1,-1))
            queues.put((replay,ae))

            if i % count == 0 and self.num == 0:
                self.pip = np.asanyarray(provisional_pip) * pip_cost
                self.pip = [p if p >= -los_cut else -los_cut for p in self.pip]
                self.total_pip = np.sum(self.pip)
                mean_pip = self.total_pip / (t + 1)
                trade_accuracy = np.mean(np.asanyarray(self.pip) > 0)
                self.trade = trade_accuracy
                mean_pip *= day_pip
                prob = self.prob(self.history)
                position_prob = self.prob(h_p)

                print('action probability = ', prob)
                print("buy = ", position_prob[1], " sell = ", position_prob[-1])
                print('trade accuracy = ', trade_accuracy)
                print('epoch: %d, total rewards: %f, mean rewards: %f' % (i + 1, float(self.total_pip), float(mean_pip)))
            try:
                self.saver.restore(self.sess, self.saver_path)
            except:
                pass
                # import traceback
                # traceback.print_exc()
####################################################################################################################################
import time
from functools import lru_cache

class Leaner:
    LEARNING_RATE = 1e-3
    GAMMA = 0.99

    def __init__(self, sess, path, window_size, OUTPUT_SIZE, MEMORY_SIZE, device='/device:GPU:0', saver_path=None, restore=False, norm=False, ent_coef='auto', target_entropy='auto'):
        self.sess = sess
        self.path = path
        self.window_size = window_size
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.saver_path = saver_path
        self.rewards = reward2
        self.MEMORY_SIZE = MEMORY_SIZE
        self.memory = Memory(self.MEMORY_SIZE)

        self.preproc()
        self.state_size = (None, self.window_size, self.df.shape[-1])

        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        # tf.reset_default_graph()
        tf.get_logger().setLevel(logging.ERROR)
        with tf.device(device):
            with tf.variable_scope("input"):
              self.policy_tf = Actor_Critic(norm)
              self.target_policy = Actor_Critic(norm)

              self.state = tf.placeholder(tf.float32, self.state_size)
              self.new_state = tf.placeholder(tf.float32, self.state_size)
              self.initial_state = tf.placeholder(tf.float32,(None,128))
              self.action = tf.placeholder(tf.float32,(None,self.OUTPUT_SIZE))
              self.reward = tf.placeholder(tf.float32,(None,1))
              self.done = tf.placeholder(tf.float32, (None, 1))

            with tf.variable_scope("model", reuse=False):
            #   self.deterministic_policy, self.policy_out, logp_pi, self.entropy, self.last_state
              self.deterministic_action, self.policy_out, logp_pi, self.entropy = self.policy_tf.actor(self.state,self.initial_state,self.OUTPUT_SIZE,"actor")
              qf1, qf2, value_fn = self.policy_tf.critic(self.state, self.initial_state, self.action, create_vf=True, create_qf=True,name="critic")
              qf1_pi, qf2_pi, _ = self.policy_tf.critic(self.state, self.initial_state, self.deterministic_action, create_vf=False, create_qf=True, name="critic")

            if self.target_entropy == 'auto':
                # automatically set target entropy if needed
                self.target_entropy = -np.prod(self.OUTPUT_SIZE).astype(np.float32)
            else:
                # Force conversion
                # this will also throw an error for unexpected string
                self.target_entropy = float(self.target_entropy)

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                # Default initial value of ent_coef when learned
                init_value = 1.0
                if '_' in self.ent_coef:
                    init_value = float(self.ent_coef.split('_')[1])
                    assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                    initializer=np.log(init_value).astype(np.float32))
                self.ent_coef = tf.exp(self.log_ent_coef)
            else:
                # Force conversion to float
                # this will throw an error if a malformed string (different from 'auto')
                # is passed
                self.ent_coef = float(self.ent_coef)

            with tf.variable_scope("target", reuse=False):
              _,_,target_vf = self.target_policy.critic(self.new_state,self.initial_state,self.action,create_qf=True, create_vf=True)
            with tf.variable_scope("loss"):
                min_qf_pi = tf.minimum(qf1_pi, qf2_pi)
                q_backup = tf.stop_gradient(self.reward + self.GAMMA * (1.0 - self.done) * target_vf)
                qf1_loss = tf.abs(0.5 * tf.reduce_mean((q_backup - qf1) ** 2))
                qf2_loss = tf.abs(0.5 * tf.reduce_mean((q_backup - qf2) ** 2))

                self.absolute_errors = tf.abs(q_backup - qf1)

                ent_coef_loss = -tf.reduce_mean(self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))

                policy_kl_loss = 0.5 * tf.reduce_mean((self.ent_coef * logp_pi) - qf1_pi ** 2)
                v_backup = tf.reduce_mean(min_qf_pi - self.ent_coef * logp_pi)
                value_loss = (0.5 * tf.reduce_mean((value_fn - v_backup) ** 2))
                self.values_losses = qf1_loss + qf2_loss + value_loss
                self.policy_loss = policy_kl_loss

            self.actor_optimizer = tf.train.AdamOptimizer(0.001,name="actor_optimizer").minimize(self.policy_loss, var_list=get_vars('model/actor/'))
            self.vf_optimizer = tf.train.AdamOptimizer(0.001,name="vf_optimizer").minimize(self.values_losses,var_list=get_vars("model/critic/"))
            self.entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE,name="entropy_optimizer").minimize(ent_coef_loss,var_list=self.log_ent_coef)

        source_params = get_vars("model/critic")
        target_params = get_vars("target/critic")

        self.target_update = tf.group([tf.assign(v_targ, 0.995*v_targ + (1-0.995)*v_main)
                                  for v_main, v_targ in zip(get_vars('model/critic'), get_vars('target/critic'))])

        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('model/critic'), get_vars('target/critic'))])

        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=2 )
        self.saver_path = saver_path

        if restore == True:
          self.saver.restore(self.sess, self.saver_path)
        else:
          self.sess.run(tf.global_variables_initializer())
          self.sess.run(target_init)

    def preproc(self):
        self.dat = df = pd.read_csv(self.path)
        s = np.asanyarray(ta.stoch(df["High"],df["Low"],df["Close"],14)).reshape((-1, 1)) - np.asanyarray(ta.stoch_signal(df["High"],df["Low"],df["Close"],14)).reshape((-1, 1))
        m = np.asanyarray(ta.macd(df["Close"])).reshape((-1, 1)) - np.asanyarray(ta.macd_signal(df["Close"])).reshape((-1, 1))
        trend3 = np.asanyarray(self.dat[["Close"]]) - np.asanyarray(ta.ema(self.dat["Close"],20)).reshape((-1, 1))
        cross1 = np.asanyarray(ta.ema(self.dat["Close"],8)).reshape((-1, 1)) - np.asanyarray(ta.ema(self.dat["Close"],5)).reshape((-1, 1))
        y = np.asanyarray(self.dat[["Open"]])
        x = np.concatenate([s,m], 1)

        gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x, y, self.window_size)
        self.x = []
        self.y = []
        for i in gen:
            self.x.extend(i[0].tolist())
            self.y.extend(i[1].tolist())
        self.x = np.asanyarray(self.x)#.reshape((-1, self.window_size, x.shape[-1]))
        self.y = np.asanyarray(self.y)

        self.df = self.x
        self.trend = self.y

    def _construct_memories_and_train(self,i):
        tree_idx, replay = self.memory.sample(self.size)

        states = np.array([a[0][0] for a in replay])
        new_states = np.array([a[0][3] for a in replay])
        init_values = np.array([a[0][-1] for a in replay])
        actions = np.array([a[0][1] for a in replay]).reshape((-1, self.OUTPUT_SIZE))
        rewards = np.array([a[0][2] for a in replay]).reshape((-1, 1))
        done = np.array([a[0][4] for a in replay]).reshape((-1, 1))

        step_ops = [self.absolute_errors, self.vf_optimizer, self.entropy_optimizer]
        absolute_errors,_,_ = self.sess.run(step_ops, feed_dict={self.state: states, self.new_state: new_states, self.done: done,
                                                                        self.action: actions, self.reward: rewards, self.initial_state: init_values})
        if i % 3 == 0:
            _ = self.sess.run(self.actor_optimizer, feed_dict={self.state: states, self.new_state: new_states, self.done: done,
                                                            self.action: actions, self.reward: rewards, self.initial_state: init_values})
            self.sess.run(self.target_update)
        self.memory.batch_update(tree_idx, absolute_errors)

    @lru_cache(1028)
    def leaner(self, queues, iterations=1000000000):
        a = True
        while a:
            if not queues.empty():
                replay, ae = queues.get()
                for r in range(len(replay)):
                    exp = replay[r]
                    self.memory.store(exp, ae[0, r])
                a = False
        self.size = 320
        for i in range(iterations):
            for s in range(100):
                # start = time.time()
                try:
                    self._construct_memories_and_train(s)
                    # elapsed_time = time.time() - start
                    # print(elapsed_time, i, 3)
                except:
                    import traceback
                    traceback.print_exc()

            _ = self.saver.save(self.sess, self.saver_path,write_meta_graph=False)
            if (i+1) % 40 == 0:
                _ = shutil.copy("/content/" + self.saver_path + ".data-00000-of-00001","/content/drive/My Drive/model")
                _ = shutil.copy("/content/" + self.saver_path + ".index","/content/drive/My Drive/model")
                _ = shutil.copy("/content/checkpoint","/content/drive/My Drive/model")
            elif (i+1) % 20 == 0:
                _ = self.saver.save(self.sess, self.saver_path,write_meta_graph=False)
                _ = shutil.copy("/content/" + self.saver_path + ".data-00000-of-00001","/content/drive/My Drive")
                _ = shutil.copy("/content/" + self.saver_path + ".index","/content/drive/My Drive")
                _ = shutil.copy("/content/checkpoint","/content/drive/My Drive")
            replay,ae = queues.get()
            for r in range(len(replay)):
                exp = replay[r]
                self.memory.store(exp, ae[0, r])
            
                    

