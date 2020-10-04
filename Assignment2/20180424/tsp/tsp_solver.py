# Traveling Salesman Problem
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import numpy as np
import torch
from GNN import *
from DQN import *
#from TspEnv import *;
from TspEnv_v4 import *;

#Visualization
def visualization(problem) :

    graph = problem.get_graph()
    node_coords = problem.as_name_dict().get("node_coords")
    plt.figure(figsize = (19.2, 14.4))
    nx.draw_networkx_nodes(graph, pos = node_coords, node_size = 10, node_color = [0.7, 0.7, 0.7, 0.5], width = 0.5, edgelist = None)
    plt.show()

if __name__ == "__main__" :

    #DQN Algorithm
    max_episode = 100

    input_dim = 14
    output_dim = 12

    replay_memory = ReplayMemory(5000)
    main_network = MLP(input_dim, output_dim, hidden_dim = [64 for _ in range(3)])
    target_network = MLP(input_dim, output_dim, hidden_dim = [64 for _ in range(3)])

    epsilon = 0.9
    epsilon_decay = 0.05
    epsilon_min = 0.05
    batch_size = 64

    agent = DQNAgent(replay_memory, main_network, target_network, batch_size)

    loss_list = []
    reward_list = []
    trajectory_list = []
    best_performance_episode = -1
    best_performance = None
    steps = 1
    target_update = 5

    env = TspEnv("pr107.tsp")
    #env = TspEnv("rl1304.tsp")
    #env = TspEnv("rl11849.tsp")

    for episode in range(max_episode) :
        
        trajectory_epi = []
        graph, start = env.reset()
        state = agent.get_state(graph)
        #print(state, start)
        done = False
        loss_epi = []
        reward_epi = []
        trajectory_epi.append(start)

        while not done :

            if random.random() < epsilon :
                action = random.randint(0, 11)
            else :
                action = torch.argmax(agent(state)).item()

            next_graph, reward, done, next_start = env.step(action, start)

            trajectory_epi.append(next_start)

            if done :
                break

            next_state = agent.get_state(next_graph)
            #print(next_state, next_start)
            reward_epi.append(reward)
            transition = [state, action, reward, next_state, done]
            agent.push(transition)

            if agent.train_start() :
                loss = agent.train()
                loss_epi.append(loss)

            if steps % target_update == 0 :
                agent.update_target()
            #print(steps)
            steps += 1
            
            state = next_state
            start = next_start

        epsilon -= epsilon_decay
        if epsilon <= epsilon_min :
            epsilon = epsilon_min
        
        total_reward = sum(reward_epi)
        if best_performance_episode == -1 :
            best_performance_episode = episode
            best_performance = -total_reward
        else :
            if best_performance > -total_reward :
                best_performance_episode = episode
                best_performance = -total_reward

        reward_list.append(total_reward)
        if agent.train_start() :
            loss_list.append(sum(loss_epi)/len(loss_epi))

        trajectory_epi.append(trajectory_epi[0])
        trajectory_list.append(trajectory_epi)

        print(episode+1, reward_list[-1])

    plt.plot(loss_list)
    plt.show()
    plt.close("all")

    plt.plot(reward_list)
    plt.show()
    plt.close("all")

    print("Best Episode: " + str(best_performance_episode))
    print("Best Performance: " + str(best_performance))
    print("Trajectory: " + str(trajectory_list[best_performance_episode]))