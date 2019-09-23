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
from td3_model import *
import time
import logging
import shutil
import ta

class Actor:
    GAMMA = 0.99
    def __init__(self, sess, path, window_size, num, STEP_SIZE, OUTPUT_SIZE, saver_path=None, restore=False, noise=False, norm=False):
        self.sess = sess
        self.path = path
        self.window_size = window_size
        self.num = num
        self.step_size = STEP_SIZE
        self.output_size = OUTPUT_SIZE
        self.saver_path = saver_path
        self.rewards = reward2
        
        self.preproc()
        self.state_size = (None, self.window_size, self.df.shape[-1])

        target_noise = 0.2
        noise_clip = 0.5

        tf.get_logger().setLevel(logging.ERROR)
        with tf.variable_scope("input"):
            self.policy_tf = Actor_Critic(norm, noise)
            self.target_policy = Actor_Critic(norm, noise)

            self.state = tf.placeholder(tf.float32, self.state_size)
            self.new_state = tf.placeholder(tf.float32, self.state_size)
            self.initial_state = tf.placeholder(tf.float32, (None, 128))
            self.action = tf.placeholder(tf.float32, (None, self.output_size))
            self.reward = tf.placeholder(tf.float32, (None, 1))
            self.done = tf.placeholder(tf.float32, (None, 1))

        with tf.variable_scope("model", reuse=False):
            self.policy_out = self.policy_tf.actor(self.state,self.initial_state,self.output_size)
            qf1, qf2 = self.policy_tf.critic(self.state, self.initial_state,self.action, self.reward)
            qf1_pi, qf2_pi = self.policy_tf.critic(self.state, self.initial_state,self.policy_out, self.reward)

        with tf.variable_scope("target", reuse=False):
            policy_out = self.target_policy.actor(self.new_state,self.initial_state,self.output_size)
            action_noise = tf.random_normal(tf.shape(policy_out), stddev=target_noise)
            action_noise = tf.clip_by_value(action_noise, -noise_clip, noise_clip)
            noisy_action = tf.clip_by_value(policy_out + action_noise, -1, 1)
            target_qf1,target_qf2 = self.target_policy.critic(self.new_state, self.initial_state,noisy_action,self.reward)

        with tf.variable_scope("loss"):
            min_qf = tf.minimum(target_qf1, target_qf2)
            backup = tf.stop_gradient(self.reward + self.GAMMA * (1.0 - self.done) * min_qf)
            min_qf = tf.minimum(qf1_pi, qf2_pi)
            pi_loss = -tf.reduce_mean(min_qf)
            q1_loss = tf.reduce_mean((qf1-backup)**2)
            q2_loss = tf.reduce_mean((qf2-backup)**2)
            q_loss = q1_loss + q2_loss

            self.absolute_errors = tf.abs(backup - qf1)

        train_pi_op = tf.train.AdamOptimizer(1e-2).minimize(pi_loss,var_list=get_vars("model/actor"))
        train_q_op  = tf.train.AdamOptimizer(1e-4).minimize(q_loss,var_list=get_vars("model/critic"))

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

        if restore == True:
          self.saver.restore(self.sess, self.saver_path)
        else:
          self.sess.run(tf.global_variables_initializer())

    def preproc(self):
        self.dat = df = pd.read_csv(self.path)
        m = np.asanyarray(ta.macd(df["Close"],6,12)).reshape((-1, 1)) - np.asanyarray(ta.macd_signal(df["Close"],6,12,3)).reshape((-1, 1))
        s = np.asanyarray(ta.stoch(self.dat["High"], self.dat["Low"], self.dat["Close"])).reshape((-1, 1)) - np.asanyarray(ta.stoch_signal(self.dat["High"], self.dat["Low"], self.dat["Close"])).reshape((-1, 1))
        ema = np.asanyarray(ta.ema(self.dat["Close"],4)).reshape((-1, 1)) - np.asanyarray(ta.ema(self.dat["Close"], 2)).reshape((-1, 1))
        trend = np.asanyarray(self.dat[["Close"]]) - np.asanyarray(ta.ema(self.dat["Close"],10)).reshape((-1, 1))
        y = np.asanyarray(self.dat[["Open"]])
        x = np.concatenate([s,ema,trend], 1)

        gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x, y, self.window_size)
        self.x = []
        self.y = []
        for i in gen:
            self.x.extend(i[0].tolist())
            self.y.extend(i[1].tolist())
        self.x = np.asanyarray(self.x)#.reshape((-1, self.window_size, x.shape[-1]))
        self.y = np.asanyarray(self.y)

        self.df = self.x[-self.step_size::]
        self.trend = self.y[-self.step_size::]

    def _select_action(self, state, next_state=None):
        prediction = self.sess.run(self.policy_out, feed_dict={self.state: [state], self.initial_state: self.init_value})

        noise = 0.2 if self.num != 0 else 0.1
        if noise != 0.:
            # prediction = np.clip(prediction, -1, 1)
            prediction += noise * self.rand.randn(self.output_size)
            prediction = np.clip(prediction, -1, 1)
        action = np.argmax(prediction)

        self.pred = prediction
        return action

    def _memorize(self, state, action, reward, new_state, dead, o):
        self.MEMORIES.append((state, action, reward, new_state, dead, o))

    def _construct(self,replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        init_values = np.array([a[-1] for a in replay])
        actions = np.array([a[1] for a in replay]).reshape((-1, self.output_size))
        rewards = np.array([a[2] for a in replay]).reshape((-1, 1))
        done = np.array([a[4] for a in replay]).reshape((-1, 1))

        absolute_errors = self.sess.run(self.absolute_errors,
                                feed_dict={self.state:states,self.new_state: new_states, self.action: actions,self.done:done,
                                           self.reward: rewards,self.initial_state:init_values})
        return absolute_errors

    def prob(self):
              prob = np.asanyarray(self.history)
              a = np.mean(prob == 0)
              b = np.mean(prob == 1)
              c = 1 - (a + b)
              prob = [a,b,c]
              return prob

    def nstep(self, r):
        running_add = 0.0
        for t in range(len(r)):
            running_add += self.GAMMA * r[t]
        return running_add

    def discount_rewards(self, r, running_add):
            running_add = running_add * self.GAMMA + r
            return running_add
            
    def run(self, queues, spread, pip_cost, los_cut, day_pip,iterations=1000000, n=4):
        spread = spread / pip_cost
        self.rand = np.random.RandomState()
        lc = los_cut / pip_cost
        for i in range(iterations):
            if (i - 1) % 1 == 0:
                h = self.rand.randint(self.x.shape[0]-(self.step_size+1))
                self.df = self.x[h:h+self.step_size]
                self.trend = self.y[h:h+self.step_size]
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
            self.init_value = np.zeros(128).reshape((1,-1))
            old = np.asanyarray(0)
            self.history = []
            self.MEMORIES = deque()
            for t in  range(0, len(self.trend)-1):
                action = self._select_action(self.df[t])
                # h_i.append(self.init_value[0])
                h_a.append(self.pred)
                self.history.append(action)
                
                states,provisional_pip,position,total_pip = self.rewards(self.trend[t],pip,provisional_pip,action,position,states,pip_cost,spread,total_pip,lc)

                reward =  total_pip - old_reword
                h_r.append(reward)
                old_reword = total_pip

                # running_add = self.discount_rewards(reward,running_add)
                # if t == self.step_size-1:
                #     done = 1.0
                # self._memorize(self.df[t], h_a[t], running_add*100, self.df[t+1], done, self.init_value[0])

            for t in range(0, len(self.trend)-1):
                tau = t - n + 1
                if tau >= 0:
                  rewards = self.nstep(h_r[tau+1:tau+n])
                  self._memorize(self.df[tau], h_a[tau],rewards*10, self.df[t+1], done, self.init_value[0])

            batch_size = int(len(self.MEMORIES) / 2)
            replay = random.sample(self.MEMORIES, batch_size)
            ae = np.asanyarray(self._construct(replay)).reshape((1,-1))
            queues.put((replay,ae))
            # (i + 1) % 10 == 0 and
            if (i + 1) % 5 == 0 and self.num == 0:
                self.pip = np.asanyarray(provisional_pip) * pip_cost
                self.pip = [p if p >= -los_cut else -los_cut for p in self.pip]
                self.total_pip = np.sum(self.pip)
                mean_pip = self.total_pip / (t + 1)
                trade_accuracy = np.mean(np.asanyarray(self.pip) > 0)
                self.trade = trade_accuracy
                mean_pip *= day_pip
                prob = self.prob()

                print('action probability = ', prob)
                print('trade accuracy = ', trade_accuracy)
                print('epoch: %d, total rewards: %f, mean rewards: %f' % (i + 1, float(self.total_pip), float(mean_pip)))
            try:
                self.saver.restore(self.sess, self.saver_path+"1")
            except:
                pass

#####################################################################################################################################

class Leaner:
    GAMMA = 0.99
    def __init__(self, sess, path, window_size, OUTPUT_SIZE, MEMORY_SIZE, device='/device:GPU:0', saver_path=None, restore=False, noise=False, norm=False):
        self.sess = sess
        self.path = path
        self.window_size = window_size
        self.output_size = OUTPUT_SIZE
        self.saver_path = saver_path
        self.rewards = reward2
        self.MEMORY_SIZE = MEMORY_SIZE
        self.memory = Memory(self.MEMORY_SIZE)
        
        self.preproc()
        self.state_size = (None, self.window_size, self.df.shape[-1])

        target_noise = 0.2
        noise_clip = 0.5

        tf.get_logger().setLevel(logging.ERROR)
        with tf.device(device):
            with tf.variable_scope("input"):
                self.policy_tf = Actor_Critic(norm, noise)
                self.target_policy = Actor_Critic(norm, noise)

                self.state = tf.placeholder(tf.float32, self.state_size)
                self.new_state = tf.placeholder(tf.float32, self.state_size)
                self.initial_state = tf.placeholder(tf.float32, (None, 128))
                self.action = tf.placeholder(tf.float32, (None, self.output_size))
                self.reward = tf.placeholder(tf.float32, (None, 1))
                self.done = tf.placeholder(tf.float32, (None, 1))

            with tf.variable_scope("model", reuse=False):
                self.policy_out = self.policy_tf.actor(self.state,self.initial_state,self.output_size)
                qf1, qf2 = self.policy_tf.critic(self.state, self.initial_state,self.action, self.reward)
                qf1_pi, qf2_pi = self.policy_tf.critic(self.state, self.initial_state,self.policy_out, self.reward)

            with tf.variable_scope("target", reuse=False):
                policy_out = self.target_policy.actor(self.new_state,self.initial_state,self.output_size)
                action_noise = tf.random_normal(tf.shape(policy_out), stddev=target_noise)
                action_noise = tf.clip_by_value(action_noise, -noise_clip, noise_clip)
                noisy_action = tf.clip_by_value(policy_out + action_noise, -1, 1)
                target_qf1,target_qf2 = self.target_policy.critic(self.new_state, self.initial_state,noisy_action,self.reward)

            with tf.variable_scope("loss"):
                min_qf = tf.minimum(target_qf1, target_qf2)
                backup = tf.stop_gradient(self.reward + self.GAMMA * (1.0 - self.done) * min_qf)
                min_qf = tf.minimum(qf1_pi, qf2_pi)
                self.pi_loss = -tf.reduce_mean(self.reward + self.GAMMA * (1.0 - self.done) * min_qf)
                q1_loss = 0.5 * tf.reduce_mean((qf1-backup)**2)
                q2_loss = 0.5 * tf.reduce_mean((qf2-backup)**2)
                q_loss = q1_loss + q2_loss

                self.absolute_errors = tf.abs(backup - qf1)

            self.train_pi_op = tf.train.AdamOptimizer(1e-3).minimize(self.pi_loss,var_list=get_vars("model/actor"))
            self.train_q_op  = tf.train.AdamOptimizer(1e-3).minimize(q_loss,var_list=get_vars("model/critic"))

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

        self.target_update = tf.group([tf.assign(v_targ, 0.995*v_targ + (1-0.995)*v_main)
                                       for v_main, v_targ in zip(get_vars('model'), get_vars('target'))])

        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('model'), get_vars('target'))])

        if restore == True:
          self.saver.restore(self.sess, self.saver_path)
        else:
          self.sess.run(tf.global_variables_initializer())
          self.sess.run(target_init)

    def preproc(self):
        self.dat = df = pd.read_csv(self.path)
        m = np.asanyarray(ta.macd(df["Close"],6,12)).reshape((-1, 1)) - np.asanyarray(ta.macd_signal(df["Close"],6,12,3)).reshape((-1, 1))
        s = np.asanyarray(ta.stoch(self.dat["High"], self.dat["Low"], self.dat["Close"])).reshape((-1, 1)) - np.asanyarray(ta.stoch_signal(self.dat["High"], self.dat["Low"], self.dat["Close"])).reshape((-1, 1))
        ema = np.asanyarray(ta.ema(self.dat["Close"],4)).reshape((-1, 1)) - np.asanyarray(ta.ema(self.dat["Close"], 2)).reshape((-1, 1))
        trend = np.asanyarray(self.dat[["Close"]]) - np.asanyarray(ta.ema(self.dat["Close"],10)).reshape((-1, 1))
        # dlr = np.asanyarray(ta.daily_log_return(self.dat["Close"])).reshape((-1, 1))
        rsi = np.asanyarray(ta.rsi(self.dat["Close"])).reshape((-1, 1))
        y = np.asanyarray(self.dat[["Open"]])
        x = np.concatenate([s,ema,trend], 1)
        

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

    def _construct_memories_and_train(self, replay, i, index=None):
        replay = np.asanyarray(replay)

        states = np.array([a[0][0] for a in replay])
        new_states = np.array([a[0][3] for a in replay])
        init_values = np.array([a[0][-1] for a in replay])
        actions = np.array([a[0][1] for a in replay]).reshape((-1, self.output_size))
        rewards = np.array([a[0][2] for a in replay]).reshape((-1, 1))
        done = np.array([a[0][4] for a in replay]).reshape((-1, 1))

        step_ops = [self.absolute_errors, self.train_q_op]

        absolute_errors, _= self.sess.run(step_ops,feed_dict={self.state: states, self.new_state: new_states,self.done:done,
                                                                        self.action: actions, self.reward: rewards, self.initial_state: init_values})
        if i % 2 == 0:
            pi,_, _ = self.sess.run([self.pi_loss,self.train_pi_op, self.target_update], feed_dict={self.state: states, self.new_state: new_states, self.done: done,
                                                                                      self.action: actions, self.reward: rewards, self.initial_state: init_values})
            # print(pi)
        if index is None:
          self.memory.batch_update(self.tree_idx, absolute_errors)
        else:
          self.memory.batch_update(index, absolute_errors)
        
    def leaner(self, queues, iterations=1000000000):
        i = 0
        a = True
        while a:
            if not queues.empty():
                replay, ae = queues.get()
                for r in range(len(replay)):
                    exp = replay[r]
                    self.memory.store(exp, ae[0, r])
                a = False

        for _ in range(iterations):
            size = 32
            try:
                self.tree_idx, batch = self.memory.sample(size)
            except:
                self.memory = Memory(self.MEMORY_SIZE)
            try:
                cost = self._construct_memories_and_train(batch,i)
                i += 1
                # print(i)
                if i % 10 == 0:
                    saved_path = self.saver.save(self.sess, self.saver_path+"1",write_meta_graph=False)
                saved_path = self.saver.save(self.sess, self.saver_path,write_meta_graph=False)
                if (i + 1) % 5 == 0:
                    _ = shutil.copy("/content/" + self.saver_path + ".data-00000-of-00001","/content/drive/My Drive")
                    _ = shutil.copy("/content/" + self.saver_path + ".index","/content/drive/My Drive")
                    _ = shutil.copy("/content/checkpoint","/content/drive/My Drive")
                if (i + 1) % 15 == 0:
                    _ = shutil.copy("/content/" + self.saver_path + ".data-00000-of-00001","/content/drive/My Drive/model")
                    _ = shutil.copy("/content/" + self.saver_path + ".index","/content/drive/My Drive/model")
                    _ = shutil.copy("/content/checkpoint","/content/drive/My Drive/model")
            except:
                pass
                # import traceback
                # traceback.print_exc()

            if i % 2 == 0:
                    saved_path = self.saver.save(self.sess, self.saver_path+"1", write_meta_graph=False)

            if not queues.empty():
                replay,ae = queues.get()
                for r in range(len(replay)):
                    exp = replay[r]
                    self.memory.store(exp, ae[0, r])
