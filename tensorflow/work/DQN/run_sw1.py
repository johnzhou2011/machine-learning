import gym
from RL_brain import DQNPrioritizedReplay
import tensorflow as tf
import gym_sw1

env = gym.make('SW1-NORMAL-ATTACK-v0')
# env.env.init_params('10.20.64.162',5000
env.env.init_params('10.20.64.116', 5000)

MEMORY_SIZE = 10000

n_actions = env.action_space.n
n_features = env.observation_space.shape[0]

sess = tf.Session()
with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=n_actions, n_features=n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(30000):

        if i_episode % 1 == 0:
            RL.save(i_episode)

        observation = env.reset()
        while True:
            # env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > 1000:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1

train(RL_prio)