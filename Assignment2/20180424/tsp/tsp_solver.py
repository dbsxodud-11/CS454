# Traveling Salesman Problem
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import numpy as np
import torch
from GNN import *
from DQN import *

# 1. Load a problem
def get_problem() :

    #problem = tsplib95.load("rl11849.tsp")
    problem = tsplib95.load("pr107.tsp")
    #print(problem.as_name_dict().get("node_coords"))
    return problem

# 2. Visualization
def visualization(problem) :

    graph = problem.get_graph()
    node_coords = problem.as_name_dict().get("node_coords")
    plt.figure(figsize = (19.2, 14.4))
    nx.draw_networkx_nodes(graph, pos = node_coords, node_size = 10, node_color = [0.7, 0.7, 0.7, 0.5], width = 0.5, edgelist = None)
    plt.show()

def construct_edges(node_position) :
    
    distance_list = []
    for number, position in node_position.items() :
        #connect the node with 5 adjacent nodes
        distance = []
        for adjacent_number, adjacent_position in node_position.items() :
            distance.append((adjacent_number, tsplib95.distances.euclidean(position, adjacent_position)))
        distance_list.append(sorted(distance, key = lambda x: x[1]))

    for i in range(len(distance_list)) :
        minimum_distances = distance_list[i]
        coord_x, coord_y = node_position.get(i+1)
        node_position[i+1] = [coord_x, coord_y, minimum_distances[1][0]-1, minimum_distances[2][0]-1, minimum_distances[3][0]-1, minimum_distances[4][0]-1, minimum_distances[5][0]-1,
                            minimum_distances[1][1], minimum_distances[2][1], minimum_distances[3][1], minimum_distances[4][1], minimum_distances[5][1]]

    return node_position

if __name__ == "__main__" :

    problem = get_problem()
    node_position = problem.as_name_dict().get("node_coords")
    node_position = construct_edges(node_position)
    
    node_from = []
    node_to = []

    for key, value in node_position.items() :
        for i in range(5) :
            node_from.append(key-1)
            node_to.append(value[i+2])
        
    graph = dgl.DGLGraph((node_from, node_to))
    graph.ndata["h"] = np.float32(list(node_position.values()))
    print(graph.ndata)

    gnn = GraphNeuralNetwork(num_layer = 3,
                            node_input_dim = 12, node_output_dim = 5,
                            edge_input_dim = 5, edge_output_dim = 12)

    graph = gnn(graph)

    #DQN Algorithm
    max_episode = 300
