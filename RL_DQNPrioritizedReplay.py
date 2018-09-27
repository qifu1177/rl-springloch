import numpy as np
import tensorflow as tf


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


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

        self.memory = Memory(capacity=memory_size)
        self.build_net()

        self.q_target = tf.placeholder(tf.float32, shape=[None, len(self.actions)])
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('loss'):
            # self.loss = tf.reduce_mean(
            #    tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.q_target, logits=self.q_eval))
            self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)  # for updating Sumtree
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
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

        action_index = self.actions.index(action)
        transition = np.hstack((self.conver_state(state), self.conver_state(other_state), [action_index, reward],
                                self.conver_state(next_s), self.conver_state(next_other_s)))
        self.memory.store(transition)  # have high priority for newly arrived transition

    def lern(self):
        if (self.learn_step_counter % self.replace_target_iter == 0):
            self.sess.run(self.target_replace_op)

        tree_idx, lern_v, ISWeights = self.memory.sample(self.batch_size)
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
        _, abs_errors, cost = self.sess.run([self.train_step, self.abs_errors, self.loss],
                                            feed_dict={self.states: states, self.q_target: q_target,
                                                       self.ISWeights: ISWeights})
        self.memory.batch_update(tree_idx, abs_errors)

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
