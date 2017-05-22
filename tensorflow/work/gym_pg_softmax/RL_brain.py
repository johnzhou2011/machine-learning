import numpy as np
import tensorflow as tf


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        self.saver = tf.train.Saver()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        self.layer1 = tf.layers.dense(
            inputs=self.tf_obs,
            units=self.n_features*2,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        # fc2
        self.layer2 = tf.layers.dense(
            inputs=self.layer1,
            units=self.n_features * 2,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # fc4
        self.all_act = tf.layers.dense(
            inputs=self.layer2,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc4'
        )

        self.all_act_prob = tf.nn.softmax(self.all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act,labels=self.tf_acts)  # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            self.loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        # layer1 = self.sess.run(self.layer1, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # print('layer1:',layer1)
        # layer2 = self.sess.run(self.layer2, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # print('layer2:', layer2)
        # layer3 = self.sess.run(self.layer3, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # print('layer3:', layer3)
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        #print(prob_weights)
        # if np.random.rand() < 0.9:
        #     action = np.random.choice(range(prob_weights.shape[1]),
        #                               p=prob_weights.ravel())  # select action w.r.t the actions prob
        # else:
        #     action = np.random.choice(range(prob_weights.shape[1]))
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def save(self, step):
        self.saver.save(self.sess, "models/sw1", global_step=step)

    def restore(self, path):
        self.saver.restore(self.sess, path)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # print('ep_rs',discounted_ep_rs_norm)
        # train on episode
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })
        loss = self.sess.run(self.loss, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        print('loss:', loss)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        if self.ep_rs.__len__() == 1:
            return [-10]
        else:
            # print('befor ep_rs:', self.ep_rs)
            discounted_ep_rs = np.zeros_like(self.ep_rs).astype(float)
            running_add = 0
            for t in reversed(range(0, len(self.ep_rs))):
                running_add = running_add * self.gamma + self.ep_rs[t]
                discounted_ep_rs[t] = running_add

            # print('after ep_rs:', discounted_ep_rs)
            # normalize episode rewards
            discounted_ep_rs -= np.mean(discounted_ep_rs)
            discounted_ep_rs /= np.std(discounted_ep_rs)

            return discounted_ep_rs
