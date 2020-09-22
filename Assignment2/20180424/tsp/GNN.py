import numpy as np
import torch
import torch.nn as nn
import dgl

class MLP(nn.Module) :

    def __init__(self, input_dim, output_dim, hidden_dim) :
        super(MLP, self)._init__()

        input_dims = [input_dim] + hidden_dim
        output_dims = hidden_dim + [output_dim]

        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(input_dims, output_dims) :
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.ReLU())

    def forward(self, x) :

        for layer in self.layers:
            x = layer(x)

        return x

class GraphLayer(nn.Module) :

    def __init__(self, edge_model, node_model) :
        super(GraphLayer, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model

        self.message_func = dgl.function.copy_e("h", "m")
        self.aggregator = dgl.function.sum("m", "agg_m")

    def forward(self, g) :

        g.apply_edges(func = self.edge_update)

        g.pull(g.nodes(), message_func = self.message_func, reduce_func = self.aggregator)
        g.apply_nodes(func = self.node_update)

        g.ndata.pop("agg_m")

        return g

    def edge_update(self) :

        nf_from = edges.src["h"]
        nf_to = edges.dst["h"]
        ef = edges.data["h"]

        edge_input = toch.cat([nf_from, nf_to, ef], dim=-1)
        update_ef = self.edge_model(edge_input)

        return {"h" : update_ef}

    def node_update(self) :

        agg_m = nodes.data["agg_m"]
        nf = nodes.data["h"]

        node_input = torch.cat([agg_m, nf], dim = -1)
        update_nf = self.node_model(node_input)

        return {"h" : update_nf}

class GraphNeuralNetwork(nn.Module) :

    def __init__(self, num_layer,
                node_input_dim, node_output_dim,
                edge_input_dim, edge_output_dim,
                node_hidden_dim = 64, edge_hidden_dim = 64) :
        super(GraphNeuralNetwork, self).__init__()

        node_input_dims = [node_input_dim] + [node_hidden_dim for _ in range(num_layer-1)]
        node_output_dims = [node_hidden_dim for _ in range(num_layer-1)] + [node_output_dim]

        edge_input_dims = [edge_input_dim] + [edge_hidden_dim for _ in range(num_layer-1)]
        edge_output_dims = [edge_hiden_dim for _ in range(num_layer-1)] + [edge_output_dim]

        self.layers = nn.ModuleList()

        for node_in_dim, node_out_dim, edge_in_dim, edge_out_dim in zip(node_input_dims, node_output_dims, edge_input_dims, edge_output_dims) :

            edge_model = MLP(2*node_in_dim + edge_in_dim, edge_out_dim, hidden_dim = [64 for _ in range(3)])
            node_model = MLP(node_in_dim + edge_in_dim, edge_out_dim, hidden_dim = [64 for _ in range(3)])

            self.layers.append(GraphLayer(edge_model, node_model))

    def forward(self, g) :

        for layer in self.layers :
            g = layer(g)

        return g