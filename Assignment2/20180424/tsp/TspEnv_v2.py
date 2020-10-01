import torch
import numpy as np
import tsplib95
import random
import dgl
import networkx as nx
from GNN import *;
import math

class TspEnv:

    def __init__(self, tsp_name) :

        #Load a Problem
        self.tsp_name = tsp_name
        self.problem = tsplib95.load(self.tsp_name)
        
        #Get Node Information
        self.node_position = self.problem.as_name_dict().get("node_coords")
        self.numberOfNodes = len(self.node_position)

        #Get Distance Information
        self.distances = dict()
        self.statistics = dict()
        for i in range(self.numberOfNodes) :
            self.distances[i+1] = dict()
            self.statistics[i+1] = dict()
            for j in range(12) :
                self.statistics[i+1][j] = 0

        for i in range(self.numberOfNodes) :
            for j in range(self.numberOfNodes) :
                if i >= j : continue
                else :
                    distance = math.sqrt((self.node_position[i+1][0] - self.node_position[j+1][0])**2 + (self.node_position[i+1][1] - self.node_position[j+1][1])**2)
                    self.distances[i+1][j+1] = distance
                    self.distances[j+1][i+1] = distance
                    distance_class = int(distance * (10 ** -3))
                    self.statistics[i+1][min(distance_class, 11)] = self.statistics[i+1].get(min(distance_class, 11))+1
                    self.statistics[j+1][min(distance_class, 11)] = self.statistics[j+1].get(min(distance_class, 11))+1

    def reset(self) :

        self.problem = tsplib95.load(self.tsp_name)
        self.node_position = self.problem.as_name_dict().get("node_coords")

        #Construct a Graph
        self.node_from = []
        self.node_to = []

        self.node_features = []

        for nodeID, position in self.node_position.items() :

            adjacent_nodes = []
            node_feature = position + [0 for _ in range(12)]

            for _ in range(12) :
                adjacent_nodeID = random.sample(list(self.node_position.keys()), 1)[0]
                while adjacent_nodeID == nodeID :
                    adjacent_nodeID = random.sample(list(self.node_position.keys()), 1)[0]

                adjacent_nodes.append(adjacent_nodeID)
            adjacent_nodes = sorted(adjacent_nodes)

            for i in range(12) :
                self.node_from.append(nodeID-1)
                self.node_to.append(adjacent_nodes[i]-1)

            for i in range(12) :
                node_feature[i+2] = self.statistics[nodeID].get(i)

            self.node_features.append(node_feature)

        self.graph = dgl.DGLGraph((self.node_from, self.node_to))
        
        self.graph.ndata["h"] = np.float32(self.node_features)
        self.graph.edata["h"] = np.float32(np.random.randn(len(self.node_position)*12, 5))

        #Randomly choose a starting point
        self.start = random.randint(0, self.numberOfNodes-1)
        self.start_position = self.node_position.get(self.start+1)

        return self.graph, self.start      

    def step(self, action, start) :
        
        position = self.node_position.pop(start+1)
        
        done = self.get_done()
        if done :
            reward = -math.sqrt((position[0] - self.start_position[0])**2 + (position[1] - self.start_position[1])**2)
            return self.graph, reward, done, self.start
        #Execute an action
        #print(self.graph.out_edges(start)[1].tolist())
        next_start = sorted(self.graph.out_edges(start)[1].tolist())[action]
        #print(self.graph.out_edges(next_start)[1].tolist())
        next_position = self.node_position.get(next_start+1)

        #Get a Reward
        reward = -math.sqrt((position[0] - next_position[0])**2 + (position[1] - next_position[1])**2)

        #Remove edges
        nodes_in, _, edges_id_in = self.graph.in_edges(start, "all")
        self.graph.remove_edges(edges_id_in)
        _, nodes_out, edges_id_out = self.graph.out_edges(start, "all")
        self.graph.remove_edges(edges_id_out)
    
        #randomly construct new edges
        for node_lost_edge in nodes_in.tolist() :
            new_adjacent_node = random.sample(self.node_position.keys(), 1)[0]
            while new_adjacent_node == node_lost_edge+1 :
                new_adjacent_node = random.sample(self.node_position.keys(), 1)[0]
            self.graph.add_edge(node_lost_edge, new_adjacent_node-1)

        self.graph.ndata["h"] = np.float32(self.node_features)
        self.graph.edata["h"] = np.float32(np.random.randn(self.graph.number_of_edges(), 5))

        return self.graph, reward, done, next_start

    def get_done(self) :
        return len(self.node_position) == 1
        