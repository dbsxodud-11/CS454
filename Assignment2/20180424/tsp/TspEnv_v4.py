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
                    distance = math.sqrt((self.node_position[i+1][0] - self.node_position[j+1][0])**2 + (self.node_position[i+1][1] - self.node_position[j+1][1])**2)
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
        self.graph = self.get_graph(self.start)

        return self.graph, self.start

    def get_graph(self, start) :

        self.graph_nodeID = dict()
        self.graph_nodeID_inverse = dict()
        self.node_from = []
        self.node_to = []
        self.nodes = [start]

        self.graph_nodeID[start] = 0
        
        # First Layer
        for _ in range(12) :
            adjacent_nodeID = random.sample(list(self.node_position), 1)[0]
            
            while adjacent_nodeID == start :
                adjacent_nodeID = random.sample(list(self.node_position), 1)[0]

            if adjacent_nodeID not in self.graph_nodeID.keys() :
                self.graph_nodeID[adjacent_nodeID] = len(self.graph_nodeID)
                self.graph_nodeID_inverse[self.graph_nodeID.get(adjacent_nodeID)] = adjacent_nodeID
            else :
                if len(self.node_position) > 12 :
                    while adjacent_nodeID in self.graph_nodeID.keys() :
                        adjacent_nodeID = random.sample(list(self.node_position), 1)[0]
                    self.graph_nodeID[adjacent_nodeID] = len(self.graph_nodeID)
                    self.graph_nodeID_inverse[self.graph_nodeID.get(adjacent_nodeID)] = adjacent_nodeID

            self.node_from.append(self.graph_nodeID.get(adjacent_nodeID))
            self.node_to.append(self.graph_nodeID.get(start))
            
            if adjacent_nodeID not in self.nodes :
                self.nodes.append(adjacent_nodeID)

        # # Second Layer
        # for nodeID in self.nodes[1:] :
        #     for _ in range(12) :
        #         adjacent_nodeID = random.sample(list(self.node_position), 1)[0]
                
        #         while adjacent_nodeID == nodeID :
        #             adjacent_nodeID = random.sample(list(self.node_position), 1)[0]
                
        #         if adjacent_nodeID not in self.graph_nodeID.keys() :
        #             self.graph_nodeID[adjacent_nodeID] = len(self.graph_nodeID)
        #             self.graph_nodeID_inverse[self.graph_nodeID.get(adjacent_nodeID)] = adjacent_nodeID
                
        #         self.node_from.append(self.graph_nodeID.get(adjacent_nodeID))
        #         self.node_to.append(self.graph_nodeID.get(nodeID))
                
        #         if adjacent_nodeID not in self.nodes :
        #             self.nodes.append(adjacent_nodeID)

        self.graph = dgl.DGLGraph((self.node_from, self.node_to))
        self.graph.ndata["h"] = np.float32(np.random.randn(self.graph.number_of_nodes(), 14))
        for nodeID, graph_nodeID in self.graph_nodeID.items() :
            self.graph.nodes[graph_nodeID].data["h"] = torch.tensor(self.node_position.get(nodeID) + [self.statistics.get(nodeID).get(i) for i in range(12)], dtype=torch.float32).reshape(1, -1)
        self.graph.edata["h"] = np.float32(np.random.randn(self.graph.number_of_edges(), 5))

        return self.graph

    def step(self, action, start) :

        position = self.node_position.pop(start)

        done = self.get_done() 
        if done :
            for end, end_position in self.node_position.items() :
                reward = self.get_reward(position, end_position)
                reward += self.get_reward(end_position, self.start_position)
            return self.graph, reward, done, end

        next_candidates = self.graph.in_edges(0)[0].tolist()
        if len(next_candidates) >= 12 :
            next_start = next_candidates[action]
        else :
            next_start = next_candidates[min(len(next_candidates)-1, action)]

        real_next_start = self.graph_nodeID_inverse.get(next_start)
        next_position = self.node_position.get(self.graph_nodeID_inverse.get(next_start))

        reward = self.get_reward(position, next_position)

        #Graph Modification
        nodes_in, _, edges_in = self.graph.in_edges(self.graph_nodeID.get(start), "all")
        self.graph.remove_edges(edges_in)

        # for node_in in list(nodes_in) :
        #     if node_in == next_start : continue
        #     sub_nodes_in, _, sub_edges_in = self.graph.in_edges(node_in, "all")
        #     self.graph.remove_edges(sub_edges_in)

        nodes_to_be_removed = []
        for node in self.graph.nodes().tolist() :
            if node != next_start :
                nodes_to_be_removed.append(node)
        self.graph.remove_nodes(nodes_to_be_removed)

        self.graph_nodeID[next_start] = 0
        self.graph_nodeID_inverse[0] = next_start

        # for i, node in enumerate(self.graph.in_edges(0)[0].tolist()) :
        #     self.graph_nodeID[self.graph_nodeID_inverse.get(node)] = i+1
        #     self.graph_nodeID_inverse[i+1] = node

        # self.graph_nodeID_inverse = {value: key for key, value in self.graph_nodeID.items() if value <= len(self.graph.in_edges(0)[0].tolist())}
        # self.graph_nodeID = {key : value for key, value in self.graph_nodeID.items() if value <= len(self.graph.in_edges(0)[0].tolist())}

        addition_nodes = []
        addition_edges_from = []
        addition_edges_to = []

        for _ in range(12) :
            adjacent_nodeID = random.sample(list(self.node_position), 1)[0]
            
            while adjacent_nodeID == start :
                adjacent_nodeID = random.sample(list(self.node_position), 1)[0]

            if adjacent_nodeID not in self.graph_nodeID.keys() :
                self.graph_nodeID[adjacent_nodeID] = len(self.graph_nodeID)
                self.graph_nodeID_inverse[self.graph_nodeID.get(adjacent_nodeID)] = adjacent_nodeID
                addition_nodes.append(self.graph_nodeID.get(adjacent_nodeID))
            else :
                if len(self.node_position) > 12 :
                    while adjacent_nodeID in self.graph_nodeID.keys() :
                        adjacent_nodeID = random.sample(list(self.node_position), 1)[0]
                    self.graph_nodeID[adjacent_nodeID] = len(self.graph_nodeID)
                    self.graph_nodeID_inverse[self.graph_nodeID.get(adjacent_nodeID)] = adjacent_nodeID
                    addition_nodes.append(self.graph_nodeID.get(adjacent_nodeID))

            addition_edges_from.append(self.graph_nodeID.get(adjacent_nodeID))
            addition_edges_to.append(0)
        
        self.graph.add_nodes(len(addition_nodes))
        print(self.graph.nodes())
        print(self.graph_nodeID_inverse)
        for additional_node in addition_nodes :    
            self.graph.nodes[additional_node].data["h"] = torch.tensor(self.node_position.get(self.graph_nodeID_inverse.get(additional_node)) + [self.statistics.get(self.graph_nodeID_inverse.get(additional_node)).get(i) for i in range(12)], dtype=torch.float32).reshape(1, -1)
        self.graph.add_edges(addition_edges_from, addition_edges_to)
        self.graph.edata["h"] = np.float32(np.random.randn(self.graph.number_of_edges(), 5))

        return self.graph, reward, done, real_next_start

    def get_done(self) :
        return len(self.node_position) == 1

    def get_reward(self, position, next_position) :
        return -math.sqrt((position[0] - next_position[0])**2 + (position[1] - next_position[1])**2)