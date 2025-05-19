from ddpg.agent import Agent
import gymnasium as gym
import numpy as np

env = gym.make()        # Put the gymnasium environment name or custom-built environment
agent = Agent(alpha=0.000025, beta=0.00025, input_dims="Put the input dimension here e.g. [[x1 x2] [y1 y2] [z1 z2]] --> [3, 2]", tau=0.001, gamma=0.99,
               n_actions="Put the action dimensions here e.g. [x1 x2 x3] --> 3", layer1_size=400, layer2_size=300, batch_size=64)

agent.load_models()
np.random.seed(0)

# Set this boolean to True when training and False when testing
# Noise will not be added to action in testing mode
is_training = True

num_episode = 1000
score_history = []
for i in range(1, num_episode+1):
    observation, info = env.reset()
    terminated = False
    truncated = False
    score = 0
    while (not terminated and not truncated):
        action = agent.choose_action(observation, is_training, (num_episode-i)/num_episode)
        new_state, reward, terminated, truncated, info = env.step(action)
        if (not truncated):
            agent.remember(observation, action, reward, new_state, int(terminated))
        agent.learn()
        score += reward
        observation = new_state
    score_history.append(score)

    if (i % 25 == 0):
       agent.save_models()

    print("Episode", i, 'score %.2f')

