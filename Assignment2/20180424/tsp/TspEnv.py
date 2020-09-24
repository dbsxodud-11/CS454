import torch
import numpy as np
import tsplib95
import random
import dgl
import networkx as nx
from GNN import *;

class TspEnv :

    def __init__(self, tsp_name) :

        self.problem = tsplib95.load(tsp_name)
        #Create a Graph
        node_position = self.problem.as_name_dict().get("node_coords")
        edges, distances, node_position = self.construct_edges(node_position)
        
        self.edges = edges
        self.distances = distances
        self.node_position = node_position

        node_from = []
        node_to = []

        for key, value in self.node_position.items() :
            for i in range(8) :
                node_from.append(key-1)
                node_to.append(value[i+2]-1)
            
        self.graph = dgl.DGLGraph((node_from, node_to))
        self.graph.ndata["h"] = np.float32(list(node_position.values()))
        self.graph.edata["h"] = np.float32(np.random.randn(len(self.node_position) * 8, 5))

        self.visited = dict()

    def reset(self) :

        start = random.randint(0, 107)
        #Use a graph neural network
        self.gnn = GraphNeuralNetwork(num_layer = 3,
                                node_input_dim = 18, node_output_dim = 12,
                                edge_input_dim = 5, edge_output_dim = 12)

        self.graph = self.gnn(self.graph)

        self.visited[start] = 1

        return self.graph.ndata["h"][start], start+1

    def step(self, action, start) :

        destination = self.edges.get(start)[action]
        distance = self.distances.get(start)[action]
        self.visited[destination] = 1

        done = self.get_done()

        return self.graph.ndata["h"][destination-1], -distance, done, destination

    def get_done(self) :

        return len(self.visited) == 107

    def construct_edges(self, node_position) :
    
        distance_list = []
        edges = dict()
        distances = dict()

        for number, position in node_position.items() :
            #connect the node with 5 -> 8 adjacent nodes
            distance = []
            for adjacent_number, adjacent_position in node_position.items() :
                if adjacent_number == number : continue
                distance.append((adjacent_number, tsplib95.distances.euclidean(position, adjacent_position)))
            distance_list.append(sorted(distance, key = lambda x: x[1]))

        for i in range(len(distance_list)) :
            minimum_distances = distance_list[i]
            return 3
            coord_x, coord_y = node_position.get(i+1)
            node_position[i+1] = [coord_x, coord_y, minimum_distances[0][0], minimum_distances[1][0], minimum_distances[2][0], minimum_distances[3][0], minimum_distances[4][0], minimum_distances[5][0], minimum_distances[6][0], minimum_distances[7][0],
                                minimum_distances[0][1], minimum_distances[1][1], minimum_distances[2][1], minimum_distances[3][1], minimum_distances[4][1], minimum_distances[5][1], minimum_distances[6][1], minimum_distances[7][1]]
            edges[i+1] = [minimum_distances[0][0], minimum_distances[1][0], minimum_distances[2][0], minimum_distances[3][0], minimum_distances[4][0], minimum_distances[5][0], minimum_distances[6][0], minimum_distances[7][0]]
            distances[i+1] = [minimum_distances[0][1], minimum_distances[1][1], minimum_distances[2][1], minimum_distances[3][1], minimum_distances[4][1], minimum_distances[5][1], minimum_distances[6][1], minimum_distances[7][1]]

        return edges, distances, node_position

    def visualization(self) :

        graph = self.problem.get_graph()
        node_coords = problem.as_name_dict().get("node_coords")
        plt.figure(figsize = (19.2, 14.4))
        nx.draw_networkx_nodes(graph, pos = node_coords, node_size = 10, node_color = [0.7, 0.7, 0.7, 0.5], width = 0.5, edgelist = None)
        plt.show()
