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
        self.episode = 1
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
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer1 = tf.layers.dense(
            inputs=self.tf_obs,
            units=self.n_features * 2,
            activation=tf.nn.relu,
            trainable=True,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        dropout = tf.layers.dropout(layer1, rate=0.1)
        fc1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc1')
        tf.summary.histogram('kernel', fc1_vars[0])
        tf.summary.histogram('bias', fc1_vars[1])
        tf.summary.histogram('act', layer1)

        # fc2
        layer2 = tf.layers.dense(
            inputs=dropout,
            units=self.n_features * 2,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        fc2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc2')
        tf.summary.histogram('kernel', fc2_vars[0])
        tf.summary.histogram('bias', fc2_vars[1])
        tf.summary.histogram('act', layer2)

        # fc3
        layer3 = tf.layers.dense(
            inputs=layer2,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc3'
        )
        fc3_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc3')
        tf.summary.histogram('kernel', fc3_vars[0])
        tf.summary.histogram('bias', fc3_vars[1])
        tf.summary.histogram('act', layer3)

        self.all_act_prob = tf.nn.softmax(layer3, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer3,
                                                                          labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            tf.summary.scalar('loss', loss)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss, tf.contrib.framework.get_global_step())

    def save(self, step):
        self.saver.save(self.sess, "models/sw1", global_step=step)

    def restore(self, path):
        self.saver.restore(self.sess, path)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: [observation]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.writer.add_summary(self.sess.run(self.merged, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        }), self.episode)
        self.episode += 1

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs).astype(float)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs).astype(int)
        return discounted_ep_rs
