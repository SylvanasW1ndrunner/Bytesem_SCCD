"""
Purpose: Defines hierarchical and flat GNN model architectures for smart contract clone detection.
Implements HierarchicalGNN with gated fusion and FlatGNN for ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool

class HierarchicalGNN(nn.Module):
    def __init__(self, node_feature_dim, gnn_hidden_dim, output_dim, num_gat_layers=3, heads=4):
        super(HierarchicalGNN, self).__init__()

        self.convs = nn.ModuleList()
        gat_output_dim = 0
        current_dim = node_feature_dim
        for i in range(num_gat_layers):
            self.convs.append(GATConv(current_dim, gnn_hidden_dim, heads=heads))
            current_dim = gnn_hidden_dim * heads
        gat_output_dim = current_dim

        self.func_agg_linear = nn.Linear(gat_output_dim, gat_output_dim)

        gate_input_dim = gat_output_dim * 2
        self.gating_network = nn.Sequential(
            nn.Linear(gate_input_dim, gat_output_dim),
            nn.Sigmoid()
        )

        self.final_linear = nn.Linear(gat_output_dim, output_dim)

    # TODO: Forward pass for hierarchical GNN with gated fusion of hierarchical and flat features
    def forward(self, data):
        x, edge_index, function_mapping, node_to_graph_batch = \
            data.x, data.edge_index, data.function_mapping, data.batch

        func_to_graph_batch = data.func_to_graph_batch if hasattr(data, 'func_to_graph_batch') else None

        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        block_embeddings = x

        valid_nodes_mask = function_mapping >= 0
        if hasattr(data, 'global_function_mapping') and torch.any(valid_nodes_mask):
            global_func_map = data.global_function_mapping
            func_embeds = global_add_pool(block_embeddings[valid_nodes_mask], global_func_map)
            func_embeds = F.elu(self.func_agg_linear(func_embeds))
            h_hier = global_mean_pool(func_embeds, func_to_graph_batch, size=data.num_graphs)
        else:
            h_hier = torch.zeros(data.num_graphs, self.func_agg_linear.out_features).to(x.device)

        h_flat = global_mean_pool(block_embeddings, node_to_graph_batch)

        h_combined = torch.cat([h_hier, h_flat], dim=1)

        gate = self.gating_network(h_combined)

        h_aggregated = gate * h_hier + (1 - gate) * h_flat

        final_embedding = self.final_linear(h_aggregated)
        final_embedding = F.normalize(final_embedding, p=2, dim=-1)

        return final_embedding

class FlatGNN(nn.Module):
    def __init__(self, node_feature_dim, gnn_hidden_dim, output_dim, num_gat_layers=3, heads=4):
        super(FlatGNN, self).__init__()

        self.convs = nn.ModuleList()
        current_dim = node_feature_dim
        for i in range(num_gat_layers):
            self.convs.append(GATConv(current_dim, gnn_hidden_dim, heads=heads))
            current_dim = gnn_hidden_dim * heads

        self.final_linear = nn.Linear(current_dim, output_dim)

    # TODO: Forward pass for flat GNN without hierarchical pooling for ablation studies
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = F.elu(conv(x, edge_index))

        graph_embedding = global_mean_pool(x, batch)

        final_embedding = self.final_linear(graph_embedding)
        final_embedding = F.normalize(final_embedding, p=2, dim=-1)

        return final_embedding