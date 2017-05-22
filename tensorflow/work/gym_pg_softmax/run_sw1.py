import gym
import gym_sw1
import numpy as np
from RL_brain import PolicyGradient

env = gym.make('SW1-NORMAL-ATTACK-v0')
# env.env.init_params('10.20.64.162',5000
env.env.init_params('10.20.64.116', 5000)

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    output_graph=True,
)

# RL.restore('models/sw1-50')

for i_episode in range(30000):

    observation = env.reset()

    if i_episode % 1 == 0:
        RL.save(i_episode)
    print('episode: ', i_episode)

    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        print(observation_)
        RL.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)
            print("episode:", i_episode, "  reward:", int(ep_rs_sum))
            vt = RL.learn()
            break
        observation = observation_
