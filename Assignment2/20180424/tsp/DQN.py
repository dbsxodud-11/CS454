import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from GNN import *

def update_model(source, target, tau) :
    for source_param, target_param in zip(source.parameters(), target.parameters()) :
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

class ReplayMemory :

    def __init__(self, capacity) :
        self.memory = deque(maxlen = capacity)

    def __len__(self) :
        return len(self.memory)

    def push(self, transition) :
        self.memory.append(transition)

    def sample(self, batch_size) :

        transitions = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        state = torch.cat(state)
        action = torch.tensor(action, dtype=torch.int64).reshape(-1, 1)
        reward = torch.tensor(action, dtype=torch.float32).reshape(-1, 1)
        next_state = torch.cat(next_state)
        done = torch.tensor(done, dtype=torch.float32).reshape(-1, 1)

        return state, action, reward, next_state, done

class MLP(nn.Module) :

    def __init__(self, input_dim, output_dim, hidden_dim) :
        super(MLP, self).__init__()

        input_dims = [input_dim] + hidden_dim
        output_dims = hidden_dim + [output_dim]
        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(input_dims, output_dims) :
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.ReLU())

    def forward(self, x) :

        for layer in self.layers :
            x = layer(x)

        return x

class DQNAgent(nn.Module) :

    def __init__(self, replay_memory, main_network, target_network, batch_size) :
        super(DQNAgent, self).__init__()

        self.replay_memory = replay_memory
        self.main_network = main_network
        self.target_network = target_network
        update_model(self.main_network, self.target_network, tau = 1.0)

        self.batch_size = batch_size
        self.gamma = 0.99

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr = 0.001)
    
    def forward(self, x) :

        x = self.main_network(x)
        return x

    def push(self, transition) :

        self.replay_memory.push(transition)

    def train_start(self) :

        return len(self.replay_memory) >= self.batch_size

    def train(self) :

        state, action, reward, next_state, done = self.replay_memory.sample(self.batch_size)
        current_q_values = self.main_network(state).gather(1, action)
        #print(current_q_values)
        next_q_values = torch.max(self.target_network(next_state), 1)[0].reshape(-1, 1).detach()
        target = reward + self.gamma * next_q_values * (1 - done)
        #print(target)
        mse_loss = self.criterion(target, current_q_values)

        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()
        
        return mse_loss.item()

    def update_target(self) :

        update_model(self.main_network, self.target_network, tau=1.0)

    def get_state(self, graph, start) :

        self.gnn = GraphNeuralNetwork(num_layer = 3,
                                node_input_dim = 14, node_output_dim = 12,
                                edge_input_dim = 5, edge_output_dim = 12)
        
        graph = self.gnn(graph)

        return graph.ndata["h"][start].reshape(1, -1)