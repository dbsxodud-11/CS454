# Traveling Salesman Problem
import tsplib95
import matplotlib.pyplot as plt
import numpy as np
import torch
from DDPG import *
import math
import random
import csv

def reverse(state, action_1, action_2, node_position) :

    original_performance = get_performance(state, node_position)
    state = list(map(int, state))
    #print(state)
    index_1 = min(action_1, action_2)
    index_2 = max(action_1, action_2)
    #print(index_1, index_2)
    if index_1 == index_2 :
        reward = 0
        state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
    else :
        shuffle = state[index_1:index_2+1]
        random.shuffle(shuffle)
        state = state[:index_1] + shuffle + state[index_2+1:]
        state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
        next_performance = get_performance(state.squeeze().tolist(), node_position)
        reward = original_performance - next_performance

    return state, reward

def get_performance(state, node_position) :

    state = list(map(int, state))
    performance = 0.0

    for i in range(len(state)-1) :
        performance += math.sqrt((node_position.get(state[i])[0] - node_position.get(state[i+1])[0])**2 + (node_position.get(state[i])[1] - node_position.get(state[i+1])[1])**2)

    return performance


if __name__ == "__main__" :
    
    #Faster RL algorithm to solve TSP proble - swap nodes to get a better solution
    max_episode = 500
    max_epi_step = 100

    #DDPG but use e-greedy action selection(not temporally correlated)
    epsilon = 0.9
    epsilon_decay = 0.05
    epsilon_min = 0.05
    batch_size = 64

    #Load a problem
    problem = tsplib95.load("pr107.tsp")
    node_position = problem.as_name_dict().get("node_coords")
    numberOfNodes = len(node_position)
    
    agent = DDPGAgent(state_dim = numberOfNodes+1, action_dim = 2, action_min = 1, action_max = numberOfNodes-1)

    actor_loss_list = []
    critic_loss_list = []
    reward_list = []
    performance_list = []
    best_performance = 100000000
    best_trajectory = list(np.random.permutation(np.arange(1,numberOfNodes+1)))
    best_trajectory += [best_trajectory[0]]

    for episode in range(max_episode) :

        state = torch.tensor(best_trajectory, dtype = torch.float32).reshape(1, -1)
        original_performance = get_performance(state.squeeze().tolist(), node_position)

        actor_loss_epi_list = []
        critic_loss_epi_list = []
        reward_epi_list = []
        
        done = False
        step = 0

        while not done :

            if random.random() < epsilon :
                action_1, action_2 = [random.randint(1, numberOfNodes-1), random.randint(1, numberOfNodes-1)]
            else :
                action_1, action_2 = agent.get_action(state)

            next_state, reward = reverse(state.squeeze().tolist(), action_1, action_2, node_position)
            reward_epi_list.append(reward)
            #print(reward)
            if reward > 0 :
                #print(reward)
                best_performance = original_performance - reward
                #print(original_performance, reward, best_performance)
                best_trajectory = next_state.squeeze().tolist()
                state = next_state

            if step == max_epi_step : done = True
            transition = [state, action_1, action_2, reward, next_state, done]
            agent.push(transition)

            step += 1

            if agent.train_start() :
                critic_loss, actor_loss = agent.train()
                critic_loss_epi_list.append(critic_loss)
                actor_loss_epi_list.append(actor_loss)

        reward_list.append(sum(reward_epi_list))
        if agent.train_start() :
            critic_loss_list.append(sum(critic_loss_epi_list) / len(critic_loss_epi_list))
            actor_loss_list.append(sum(actor_loss_epi_list) / len(actor_loss_epi_list))

        epsilon -= epsilon_decay
        if epsilon <= epsilon_min :
            epsilon = epsilon_min

        #print(episode+1, reward_list[-1], best_performance)
        performance_list.append(best_performance)

    print("Best Performance: " + str(best_performance))
    best_trajectory = list(map(int, best_trajectory))
    
    with open("solution.csv", "w", newline="") as file :
        writer = csv.writer(file)
        for city in best_trajectory[:-1] :
            writer.writerow([city])
    
    #print("Best Trajectory: " + str(best_trajectory))





    
    