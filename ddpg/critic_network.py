import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

# Critic / Q-value Network / Q
# Evaluate state/action pairs
class CriticNetwork(nn.Module):
    def __init__(self, learning_rate, n_actions, input_dims, fc1_dims, fc2_dims, name, chkpt_dir="trained_model"):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,name+'_ddpg')

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[1])                     # Square root of the fan-in
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(fc1_dims)          

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[1])                     # Square root of the fan-in
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.action_value = nn.Linear(n_actions, fc2_dims)
        f3 = 0.0003
        self.q = nn.Linear(fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = T.add(state_value, action_value)
        state_action_value = F.relu(state_action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file): 
            print("... loading checkpoint ...")
            self.load_state_dict(T.load(self.checkpoint_file))    