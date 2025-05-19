# Runnable demo of main.py using the Lunar Lander Environment
from ddpg.agent import Agent
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt 

env = gym.make('LunarLanderContinuous-v3')
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, gamma=0.99,
               n_actions=2, layer1_size=400, layer2_size=300, batch_size=64)

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

    print('episode', i, 'score %.2f' % score, 'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

# Plot the training result
filename = 'training_result.png' 
N = len(score_history)
running_avg = np.empty(N)
for t in range(N):
    running_avg[t] = np.mean(score_history[max(0, t-100):(t+1)])
x = [i for i in range(N)]
plt.ylabel('Score')       
plt.xlabel('Game')                     
plt.plot(x, running_avg)
plt.savefig(filename)
