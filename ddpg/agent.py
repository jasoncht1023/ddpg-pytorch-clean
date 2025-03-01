import torch as T
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .ou_action_noise import OUActionNoise
from .replay_buffer import ReplayBuffer
from .actor_network import ActorNetwork
from .critic_network import CriticNetwork
import os

# alpha and beta are the learning rate for actor and critic network, gamma is the discount factor for future reward
# tau is the soft "update rate" of the target networks parameters
class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma, n_actions, layer1_size, layer2_size, batch_size=64, max_size=1000000):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.model_dir = "trained_model"

        self.actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, input_dims=input_dims,
                                  fc1_dims=layer1_size, fc2_dims=layer2_size, name="actor", chkpt_dir=self.model_dir)

        self.critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=layer1_size, fc2_dims=layer2_size, name="critic", chkpt_dir=self.model_dir)

        self.target_actor = ActorNetwork(learning_rate=alpha, n_actions=n_actions, input_dims=input_dims,
                                         fc1_dims=layer1_size, fc2_dims=layer2_size, name="target_actor", chkpt_dir=self.model_dir)
        
        self.target_critic = CriticNetwork(learning_rate=beta, n_actions=n_actions, input_dims=input_dims,
                                           fc1_dims=layer1_size, fc2_dims=layer2_size, name="target_critic", chkpt_dir=self.model_dir)

        self.noise = OUActionNoise(mu=np.zeros(n_actions), sigma=0.2, theta=0.15)

        self.n_actions = n_actions

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, is_training):    
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)

        # Epsilon-greedy exploration
        if (is_training == True):
            epsilon = np.random.rand()
            if (self.memory.mem_cntr < 100000):
                if (epsilon < 0.5):
                    mu += T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            elif (self.memory.mem_cntr < 300000):
                if (epsilon < 0.25):
                    mu += T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            else:
                if (epsilon < 0.1):
                    mu += T.tensor(self.noise(), dtype=T.float).to(self.actor.device)     

        self.actor.train()
        return mu.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if (self.memory.mem_cntr < self.batch_size):
            return

        states, action, reward, new_states, done = self.memory.sample_buffer(self.batch_size)
        
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        states = T.tensor(states, dtype=T.float).to(self.critic.device)

        target_actions = self.target_actor.forward(new_states)
        target_critic_value = self.target_critic.forward(new_states, target_actions)
        critic_value = self.critic.forward(states, action)
        
        target = []
        for i in range(self.batch_size):
            target.append(reward[i] + self.gamma * (1 - done[i]) * target_critic_value[i])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(critic_value, target)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(states)
        actor_loss = -self.critic.forward(states, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters(tau=self.tau)

    def update_network_parameters(self, tau):
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] =  tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_dict[name].clone()
            
        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        if not os.path.isdir(self.model_dir): 
            os.makedirs(self.model_dir)

        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
