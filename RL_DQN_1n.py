import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, actions, n_state=4, lr=0.01, epsilon=0.9, batch_size=200,
                 reward_decay=0.9, n_l1=200):
        self.actions = actions
        self.n_state = n_state
        self.n_l1 = n_l1
        self.lr = lr
        self.epsilon = epsilon
        self.memory_size = batch_size
        self.batch_size = batch_size
        self.gamma = reward_decay

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_state * 2 + 2))
        self.build_net()
        self.action_v = tf.placeholder(tf.float32, shape=[None, len(self.actions)])
        # self.q_target = tf.placeholder(tf.float32, shape=[None, len(self.actions)])
        self.q_target = tf.placeholder(tf.float32, shape=[None])

        with tf.variable_scope('loss'):
            # self.loss = tf.reduce_mean(
            #    tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.q_target, logits=self.q_eval))
            # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            q_action = tf.reduce_sum(tf.multiply(self.q_eval, self.action_v), reduction_indices=1)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, q_action))
        with tf.variable_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def build_net(self):
        self.states = tf.placeholder(tf.float32, shape=[None, self.n_state])

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.003), tf.constant_initializer(0.001)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            l1 = tf.layers.dense(self.states, self.n_l1, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='eval_l1')
            self.q_eval = tf.layers.dense(l1, len(self.actions), kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='eval_q')

    def close(self):
        self.sess.close()

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
            action_val = self.sess.run(self.q_eval, feed_dict={self.states: s})
            # print(s)
            print(action_val[0])
            action = self.actions[np.argmax(action_val[0])]
            print("state=" + str(state) + "; other state=" + str(other_state) + ";action=" + str(action))
            return action

    def store_transition(self, state, other_state, next_s, next_other_s, action, reward):
        if (not hasattr(self, "memory_counter")):
            self.memory_counter = 0
        action_index = self.actions.index(action)
        transition = np.hstack((self.conver_state(state), self.conver_state(other_state), [action_index, reward],
                                self.conver_state(next_s), self.conver_state(next_other_s)))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def lern(self):

        if (self.memory_counter < self.memory_size):
            return

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        lern_v = self.memory[batch_index, :]
        states = lern_v[:, :self.n_state]
        next_states = lern_v[:, -self.n_state:]

        q_next = self.sess.run(self.q_eval, feed_dict={self.states: next_states})
        # q_eval = self.sess.run(self.q_eval, feed_dict={self.states: states})

        q_target = np.zeros(self.batch_size)
        eval_act_index = lern_v[:, self.n_state].astype(int)
        action_v = np.zeros((self.batch_size, len(self.actions)), dtype=np.int32)
        # print("eval_act_index")
        # print(eval_act_index)
        eval_reward = lern_v[:, self.n_state + 1]

        # print("eval_reward")
        # print(eval_reward)

        selected_q_next = np.max(q_next, axis=1)
        for index in batch_index:
            action_v[index][eval_act_index[index]] = 1
            if (eval_reward[index] == 0):
                q_target[index] = eval_reward[index] + self.gamma * selected_q_next[index]
                # q_target[index] = eval_reward[index] + self.gamma * np.max(q_next[index])
            else:
                q_target[index] = eval_reward[index]

            # print("max")
            # print(np.max(q_next, axis=1))
        _, cost = self.sess.run([self.train_step, self.loss],
                                feed_dict={self.states: states, self.q_target: q_target,
                                           self.action_v: action_v})

        print("loss=" + str(cost))
        # print(q_target)
        self.cost_his.append(cost)
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


if (__name__ == "__main__"):
    Actions = ["up", "right", "down", "left"]
    states = [11, 12, 13, 21, 22, 23, 31, 33]
    rewards = [-1, 0, 1]
    rl = DQN(Actions, memory_size=200, batch_size=5)
    for i in range(5):
        state = np.random.choice(states, size=4)
        action = np.random.choice(Actions)
        # print(action)
        reward = np.random.choice(rewards)
        rl.store_transition(state[0], state[1], state[2], state[3], action, reward)
    rl.lern()
