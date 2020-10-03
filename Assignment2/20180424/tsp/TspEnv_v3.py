import torch
import numpy as np
import tsplib95
import random
import dgl
import networkx as nx
from GNN import *;
import math

class TspEnv :

    def __init__(self, tsp_name) :

        #Load a Problem
        self.tsp_name = tsp_name
        self.problem = tsplib95.load(self.tsp_name)

        #Get Node Information
        self.node_position = self.problem.as_name_dict().get("node_coords")
        self.numberOfNodes = len(self.node_position)

        #Get Distance Information
        self.distances = dict()
        self.statistics = dict() #information about distribution of distances to adhacent nodes

        for i in range(self.numberOfNodes) :
            self.distances[i+1] = dict()
            self.statistics[i+1] = dict()
            for j in range(12) :
                self.statistics[i+1][j] = 0

        self.max_distance = 0
        for i in range(self.numberOfNodes) :
            for j in range(self.numberOfNodes) :
                
                if i >= j : continue
                else :
                    distance = math.sqrt((self.node_position[i+1][0] - self.node_postion[j+1][0])**2 + (self.node_position[i+1][1] - self.node_position[j+1][1])**2)
                    self.distances[i+1][j+1] = distance
                    self.distances[j+1][i+1] = distance
                    if self.max_distance < distance : self.max_distance = distance
                    distance_class = int((distance * (10 ** -3)))
                    self.statistics[i+1][min(distance_class, 11)] = self.statistics[i+1].get(min(distance_class, 11)) + 1
                    self.statistics[j+1][min(distance_class, 11)] = self.statistics[i+1].get(min(distance_class, 11)) + 1

    def reset(self) :

        self.problem = tsplib95.load(self.tsp_name)
        self.node_position = self.problem.as_name_dict().get("node_coords")

        #Choose a random start point
        self.start = random.randint(1, self.numberOfNodes)
        self.start_position = self.node_position.get(self.start)

        #Initial Graph
        self.graph = self.get_graph(start)

        return self.graph, self.start
    
    def get_graph(self, start) :

        #Construct a graph from the given starting point
        self.nodes = [start]
        self.node_from = []
        self.node_to = []
        self.node_features = [self.node_position.get(start) + [self.statistics.get(start).get(i) for i in range(12)]]
        
        #Connect adjacent nodes with edges 3 times
        for _ in range(12) :
            adjacent_nodeID = random.sample(list(self.node_position.keys()), 1)[0]
            while adjacent_nodeID == start :
                adjacent_nodeID = random.sample(list(self.node_position.keys()), 1)[0]
            self.nodes.append(adjacent_nodeID)
            self.node_from.append(adjacent_nodeID)
            self.node_to.append(start)
            self.node_features.append(self.node_position.get(adjacent_nodeID) + [self.statistics.get(adjacent_nodeID).get(i) for i in range(12)])

        first_step = len(self.nodes)
        for nodeID in self.nodes[1:first_step] :   
            for _ in range(12) :
                adjacent_nodeID = random.sample(list(self.node_position.keys()), 1)[0]
                while adjacent_nodeID == nodeID :
                    adjacent_nodeID = random.sample(list(self.node_position.keys()), 1)[0]
                self.nodes.append(adjacent_nodeID)
                self.node_from.append(adjacent_nodeID)
                self.node_to.append(nodeID)
                self.node_features.append(self.node_position.get(adjacent_nodeID) + [self.statistics.get(adjacent_nodeID).get(i) for i in range(12)])    

        second_step = len(self.nodes)
        for nodeID in self.nodes[first_step:second_step] :
            for _ in range(12) :
                adjacent_nodeID = random.sample(list(self.node_position.keys()), 1)[0]
                while adjacent_nodeID == nodeID :
                    adjacent_nodeID = random.sample(list(self.node_position.keys()), 1)[0]
                self.nodes.append(adjacent_nodeID)
                self.node_from.append(adjacent_nodeID)
                self.node_to.append(nodeID)
                self.node_features.append(self.node_position.get(adjacent_nodeID) + [self.statistics.get(adjacent_nodeID).get(i) for i in range(12)])

        #DGLGraph node starts from 0
        self.node_from.append(0)
        self.node_to.append(0)

        graph = dgl.DGLGraph((self.node_from, self.node_to))
        graph.ndata["h"] = np.float32(self.node_features)
        graph.edata["h"] = np.float32(np.random.randn(graph.number_of_edges(), 5))

        return graph

    def step(self, action, start) :

        #Execute action and get a reward and next state
        position = self.node_position.pop(start+1)

        done = self.get_done() :
        if done :
            for end, end_position in self.node_position.items() :
                reward = self.get_reward(position, end_position)
                reward = self.get_reward(end_position, self.start_position)
            return self.graph, reward, done, end

        next_start = sorted(self.graph.in_edges(start)[1].tolist())[action]
        next_position = self.node_position.get(next_start)

        reward = self.get_reward(position, next_position)

        #Construct a new graph
        self.graph.clear()
        self.graph = self.get_graph(next_start)

        return self.graph, reward, done, next_start

    def get_done(self) :
        return len(self.node_position) == 1

    def get_reward(positon, next_position) :
        return -math.sqrt((position[0] - next_position[0])**2 + (position[1] - next_position[1])**2)
