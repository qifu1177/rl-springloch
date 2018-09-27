import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, actions, n_state=4, lr=0.01, epsilon=0.9, memory_size=2000, batch_size=200,
                 reward_decay=0.9, replace_target_iter=200):
        self.actions = actions
        self.n_state = n_state
        self.n_l1 = 400
        self.lr = lr
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, self.n_state * 2 + 2))
        self.build_net()

        self.q_target = tf.placeholder(tf.float32, shape=[None, len(self.actions)])

        with tf.variable_scope('loss'):
            # self.loss = tf.reduce_mean(
            #    tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.q_target, logits=self.q_eval))
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net_params')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net_params')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def build_layers(self, state, c_names, w_initializer, b_initializer):
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.n_state, self.n_l1], initializer=w_initializer, collections=c_names)
            b1 = tf.get_variable('b1', [1, self.n_l1], initializer=b_initializer, collections=c_names)
            l1 = tf.nn.relu(tf.matmul(state, w1) + b1)

        with tf.variable_scope('Q'):
            w2 = tf.get_variable('w2', [self.n_l1, len(self.actions)], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, len(self.actions)], initializer=b_initializer, collections=c_names)
            out = tf.matmul(l1, w2) + b2

        return out

    def build_net(self):
        self.states = tf.placeholder(tf.float32, shape=[None, self.n_state])
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.003), tf.constant_initializer(0.001)
        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_eval = self.build_layers(self.states, c_names, w_initializer, b_initializer)

        # ------------------ build target_net ------------------
        self.next_states = tf.placeholder(tf.float32, shape=[None, self.n_state])
        with tf.variable_scope('target_net'):
            c_names = ["target_net_params", tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = self.build_layers(self.next_states, c_names, w_initializer, b_initializer)

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
        if (self.learn_step_counter % self.replace_target_iter == 0):
            self.sess.run(self.target_replace_op)

        if (self.memory_counter > self.memory_size):
            indexs = np.random.choice(self.memory_size, self.batch_size)
        else:
            indexs = np.random.choice(self.memory_counter, self.batch_size)

        lern_v = self.memory[indexs, :]
        states = lern_v[:, :self.n_state]
        next_states = lern_v[:, -self.n_state:]

        q_eval, q_next = self.sess.run([self.q_eval, self.q_next],
                                       feed_dict={self.states: states, self.next_states: next_states})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = lern_v[:, self.n_state].astype(int)
        # print("eval_act_index")
        # print(eval_act_index)
        eval_reward = lern_v[:, self.n_state + 1]

        # print("eval_reward")
        # print(eval_reward)

        selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = eval_reward + self.gamma * selected_q_next

        # print("q_target")
        # print(q_target)
        _, cost = self.sess.run([self.train_step, self.loss],
                                feed_dict={self.states: states, self.q_target: q_target})

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
