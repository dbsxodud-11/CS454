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
        self.problem = tsplib95.load(tsp_name)
        
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
            if i % 100 == 0 : print(str(i / 11849 * 100) + "%")
            for j in range(self.numberOfNodes) :
                if i == j : continue
                else :
                    distance = math.sqrt((self.node_position[i+1][0] - self.node_position[j+1][0])**2 + (self.node_position[i+1][1] - self.node_position[j+1][1])**2)
                    self.distances[i+1][j+1] = distance
                    distance_class = int(distance * (10 ** -3))
                    self.statistics[i+1][min(distance_class, 11)] = self.statistics[i+1].get(min(distance_class, 11))+1

    def reset(self) :

        start = random.randint(0, self.numberOfNodes-1)
        return self.get_graph(start), start

    def get_graph(self, start) :

        if len(self.node_position) == 1 : return dgl.DGLGraph()
        #Create a Graph
        node_from = []
        node_to = []
        
        node_features = []
        self.edges = dict()
        oldnodeID = 0
        for nodeID, position in self.node_position.items() :
            
            if oldnodeID+1 != nodeID :
                while oldnodeID+1 != nodeID :
                    node_features.append([0 for _ in range(14)])
                    oldnodeID += 1
            oldnodeID = nodeID

            adjacent_nodes = []
            node_feature = position + [0 for _ in range(12)]

            for _ in range(8) :
                adjacent_nodeID = random.sample(list(self.node_position.keys()), 1)[0]
                while adjacent_nodeID == nodeID :
                    adjacent_nodeID = random.sample(list(self.node_position.keys()), 1)[0]
                
                adjacent_nodes.append(adjacent_nodeID)
            adjacent_nodes = sorted(adjacent_nodes)

            for i in range(8) :
                node_from.append(nodeID-1)
                node_to.append(adjacent_nodes[i]-1)
            self.edges[nodeID] = adjacent_nodes

            for i in range(12) :
                node_feature[i+2] = self.statistics[nodeID].get(i)

            #print(node_feature)
            node_features.append(node_feature)

        #Just Because Nodes must start from zero
        if 1 not in self.edges :
            node_from.append(0)
            node_to.append(0)

        self.graph = dgl.DGLGraph((node_from, node_to))
        self.graph.ndata["h"] = np.float32(node_features)
        if 1 not in self.edges :
            self.graph.edata["h"] = np.float32(np.random.randn(len(self.node_position)*8+1, 5))
        else :
            self.graph.edata["h"] = np.float32(np.random.randn(len(self.node_position)*8, 5))
        
        return self.graph

    def step(self, action, start) :
        #Remove start node from node position
        position = self.node_position.pop(start+1)

        #Move one step forward
        next_start = self.edges.get(start+1)[action]
        next_position = self.node_position.get(next_start)

        #Get next state, reward, done, next_start
        next_graph = self.get_graph(next_start-1)
        reward = -math.sqrt((position[0] - next_position[0])**2 + (position[1] - next_position[1])**2)
        done = self.get_done()

        return next_state, reward, done, next_start-1

    def get_done(self) :
        return len(self.node_position) == 1

        