import numpy as np
import tensorflow as tf


class Actor:
    def __init__(self, n_state, n_l1, actions, sess, lr=0.01):
        self.n_state = n_state
        self.n_l1 = n_l1
        self.actions = actions
        self.n_action = len(self.n_action)
        self.lr = lr

        self.build_net()
        self.td_error = tf.placeholder(tf.float32, shape=[None, 1])
        self.input_act = tf.placeholder(tf.int32, shape=[None, self.n_action])
        log_pro = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out_act, labels=self.input_act)
        self.loss = tf.reduce_mean(log_pro * self.td_error)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess = sess

    def build_net(self):
        self.state = tf.placeholder(tf.float32, shape=[None, self.n_state])

        w_init = tf.random_normal_initializer(mean=0., stddev=0.03)
        b_init = tf.constant_initializer(0.01)
        self.l1 = tf.layers.dense(self.state, self.n_l1, activation=tf.nn.relu, kernel_initializer=w_init,
                                  bias_initializer=b_init)
        self.out = tf.layers.dense(self.l1, self.n_action, activation=None, kernel_initializer=w_init,
                                   bias_initializer=b_init)
        self.out_act = tf.nn.softmax(self.out)

    def conver_state(self, state):
        row = (int(state / 10) - 2) / 2
        col = (state % 10 - 2) / 2
        return [row, col]

    def choose_action(self, state, other_state, islern=True):
        if ((np.random.uniform() > self.epsilon) and islern):
            return np.random.choice(self.actions)
        else:
            s = np.hstack((self.conver_state(state), self.conver_state(other_state)))
            s = s.reshape((1, self.n_state))
            action_val = self.sess.run(self.act_pro, feed_dict={self.states: s})
            # print("act_pro")
            # print(action_val)
            # action_index = np.random.choice(range(action_val.shape[1]), p=action_val.ravel())
            action_index = np.argmax(action_val[0])
            action = self.actions[action_index]
            print("state=" + str(state) + "; other state=" + str(other_state) + ";action=" + str(action))
        return action

    def lern(self, state, act, td):
        _, loss = self.sess.run([self.train_step, self.loss],
                                feed_dict={self.state: state, self.input_act: act, self.td_error: td})


class Critic:
    def __init__(self, n_state, n_l1, sess, lr=0.01, reward_decay=0.9):
        self.n_state = n_state
        self.n_l1 = n_l1
        self.lr = lr
        self.gamma = reward_decay
        self.build_net()
        self.reward = tf.placeholder(tf.float32, shape=[None, 1])
        self.state = tf.placeholder(tf.float32, shape=[None, self.n_state])
        self.next_state = tf.placeholder(tf.float32, shape=[None, self.n_state])
        self.q_eval = self.build_net(self.state)
        self.q_next = self.build_net(self.next_state)
        self.td_error = self.reward + self.gamma * self.q_next - self.q_eval
        self.loss = tf.square(self.td_error)
        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.sess = sess

    def build_net(self, state):
        w_init = tf.random_normal_initializer(mean=0., stddev=0.03)
        b_init = tf.constant_initializer(0.01)
        l1 = tf.layers.dense(self.state, self.n_l1, activation=tf.nn.relu, kernel_initializer=w_init,
                             bias_initializer=b_init)
        q = tf.layers.dense(self.l1, self.n_action, activation=None, kernel_initializer=w_init,
                            bias_initializer=b_init)
        return q

    def lern(self, state, next_state, reward):
        _, loss = self.sess.run([self.train_step, self.loss],
                                feed_dict={self.state: state, self.next_state: next_state, self.reward: reward})


class RL_brain:
    def __init__(self, n_state, n_l1, actions, lr=0.01, reward_decay=0.9):
        self.n_state = n_state
        self.n_l1 = n_l1
        self.actions = actions
        self.n_action = len(self.n_action)
        self.lr = lr

        self.sess = tf.Session()
        self.actor = Actor(n_state, n_l1, actions, self.sess)
        self.critic = Critic(n_state, n_l1, self.sess, lr, reward_decay)
        self.sess.run(tf.global_variables_initializer())

