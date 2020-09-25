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

    def reset(self) :

        start = random.randint(0, self.numberOfNodes-1)
        return self.get_state(start), start

    def get_state(self, start) :

        if len(self.node_position) == 1 : return torch.zeros(12)
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

            min_distance = 1000000
            max_distance = 0
            for other_nodeID, other_position in self.node_position.items() :
                if nodeID == other_nodeID : continue
                distance = math.sqrt((position[0] - other_position[0])**2 + (position[1] - other_position[1])**2)
                if distance < 1000 : node_feature[2] += 1
                elif (distance < 2000) & (distance >= 1000) : node_feature[3] += 1
                elif (distance < 3000) & (distance >= 2000) : node_feature[4] += 1
                elif (distance < 4000) & (distance >= 3000) : node_feature[5] += 1
                elif (distance < 5000) & (distance >= 4000) : node_feature[6] += 1
                elif (distance < 6000) & (distance >= 5000) : node_feature[7] += 1
                elif (distance < 7000) & (distance >= 6000) : node_feature[8] += 1
                elif (distance < 8000) & (distance >= 7000) : node_feature[9] += 1
                elif (distance < 9000) & (distance >= 10000) : node_feature[10] += 1
                else : node_feature[11] += 1

                if distance < min_distance :
                    node_feature[12] = distance
                    min_distance = distance
                if distance > max_distance :
                    node_feature[13] = distance
                    max_distance = distance 

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

        #Communicate with other nodes with Graph Neural Network
        self.gnn = GraphNeuralNetwork(num_layer = 5,
                                    node_input_dim = 14, node_output_dim = 16,
                                    edge_input_dim = 5, edge_output_dim = 16)

        self.graph = self.gnn(self.graph)
        
        return self.graph.ndata["h"][start].reshape(1, -1)

    def step(self, action, start) :
        #Remove start node from node position
        position = self.node_position.pop(start+1)

        #Move one step forward
        next_start = self.edges.get(start+1)[action]
        next_position = self.node_position.get(next_start)

        #Get next state, reward, done, next_start
        next_state = self.get_state(next_start-1)
        reward = -math.sqrt((position[0] - next_position[0])**2 + (position[1] - next_position[1])**2)
        done = self.get_done()

        return next_state, reward, done, next_start-1

    def get_done(self) :
        return len(self.node_position) == 1

        