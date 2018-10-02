import numpy as np
import tensorflow as tf


class Policy:
    def __init__(self, actions, n_state=4, lr=0.01, reward_decay=0.95, n_l1=40):
        self.epsilon = 0.9
        self.actions = actions
        self.n_state = n_state
        self.n_actions = len(self.actions)
        self.n_l1 = n_l1
        self.lr = lr
        self.gamma = reward_decay

        self.reward_v, self.action_v, self.state_v = [], [], []

        self.build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def build_net(self):
        self.states = tf.placeholder(tf.float32, [None, self.n_state], name="state")
        self.act = tf.placeholder(tf.int32, [None, self.n_actions], name="actions_num")
        self.tf_vt = tf.placeholder(tf.float32, [None, 1], name="actions_value")
        with tf.variable_scope("l1"):
            l1 = tf.layers.dense(self.states, self.n_l1, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.03),
                                 bias_initializer=tf.constant_initializer(0.01), name="l1")
        with tf.variable_scope("out"):
            out = tf.layers.dense(l1, self.n_actions,
                                  kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.03),
                                  bias_initializer=tf.constant_initializer(0.01), name="out")
        self.act_pro = tf.nn.softmax(out, name="act_pro")
        with tf.name_scope("loss"):
            log_pro = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.act_pro, labels=self.act)
            self.loss = tf.reduce_mean(log_pro * self.tf_vt)
        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def conver_state(self, state):
        row = (int(state / 10) - 2) / 2
        col = (state % 10 - 2) / 2
        return [row, col]

    def store_transition(self, state, action, reward, other_state):
        s = self.conver_state(state) + self.conver_state(other_state)
        self.state_v.append(s)
        action_index = self.actions.index(action)
        action_v = np.zeros(len(self.actions), dtype=np.float32)
        action_v[action_index] = 1
        self.action_v.append(action_v)
        self.reward_v.append(reward)

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

    def clear_storage(self):
        self.reward_v, self.action_v, self.state_v = [], [], []

    def lern(self):
        s_v = np.array(self.state_v)
        a_v = np.array(self.action_v)
        # r_v = np.vstack(self.reward_v)
        r_v = self._discount_and_norm_rewards()
        # print("s_v")
        # print(s_v)
        # print("a_v")
        # print(a_v)
        # print("r_v")
        # print(r_v)
        _, cost = self.sess.run([self.train_step, self.loss],
                                feed_dict={self.states: s_v, self.act: a_v,
                                           self.tf_vt: r_v})
        self.clear_storage()
        print("loss=" + str(cost))
        # print(q_target)
        self.cost_his.append(cost)

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.reward_v)
        running_add = 0
        for t in reversed(range(0, len(self.reward_v))):
            running_add = running_add * self.gamma + self.reward_v[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs = discounted_ep_rs.astype(np.float32)
        mean = np.mean(discounted_ep_rs)
        # print("mean=" + str(mean))
        discounted_ep_rs -= mean
        std = np.std(discounted_ep_rs)
        # print("std=" + str(std))
        if (std != 0.):
            discounted_ep_rs /= std
        return np.vstack(discounted_ep_rs)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
