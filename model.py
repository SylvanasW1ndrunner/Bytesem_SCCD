
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool



class HierarchicalGNN(nn.Module):
    """
    一个融合了“分层”和“扁平”两种视角的、带门控机制的GNN模型。
    """
    def __init__(self, node_feature_dim, gnn_hidden_dim, output_dim,
                 num_gat_layers=3, heads=4):
        super(HierarchicalGNN, self).__init__()

        # ===============================
        # GAT Backbone (Block-level)
        # ===============================
        self.convs = nn.ModuleList()
        current_dim = node_feature_dim
        for _ in range(num_gat_layers):
            self.convs.append(GATConv(current_dim, gnn_hidden_dim, heads=heads))
            current_dim = gnn_hidden_dim * heads

        self.gat_output_dim = current_dim

        # ===============================
        # Hierarchical Path (Function-level)
        # ===============================
        self.func_agg_linear = nn.Linear(self.gat_output_dim, self.gat_output_dim)

        # Function-level attention
        self.func_att = nn.Sequential(
            nn.Linear(self.gat_output_dim, self.gat_output_dim // 2),
            nn.ReLU(),
            nn.Linear(self.gat_output_dim // 2, 1)
        )

        # Residual control
        self.hier_scale = nn.Parameter(torch.tensor(0.5))
        self.hier_proj = nn.Linear(self.gat_output_dim, self.gat_output_dim)

        # ===============================
        # Output Projection
        # ===============================
        self.final_linear = nn.Linear(self.gat_output_dim, output_dim)

    def forward(self, data):
        """
        data should contain:
          - x: node features
          - edge_index: CFG edges
          - batch: node -> graph mapping
          - function_mapping: node -> function (local, -1 if invalid)
          - global_function_mapping: node -> global function id
          - func_to_graph_batch: function -> graph id
        """

        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        function_mapping = data.function_mapping

        # ===============================
        # Block-level GNN
        # ===============================
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))

        block_embeddings = x   # [num_nodes, dim]

        # ===============================
        # Flat Path (Global CFG semantics)
        # ===============================
        h_flat = global_mean_pool(block_embeddings, batch)   # [num_graphs, dim]

        # ===============================
        # Hierarchical Path (Function-aware)
        # ===============================
        valid_nodes_mask = function_mapping >= 0
        has_hier_info = (
                hasattr(data, 'global_function_mapping') and
                hasattr(data, 'func_to_graph_batch')
        )

        if has_hier_info and torch.any(valid_nodes_mask):
            func_to_graph_batch = data.func_to_graph_batch
            # ---- Block -> Function aggregation ----
            func_embeds = global_add_pool(
                block_embeddings[valid_nodes_mask],
                data.global_function_mapping
            )  # [num_funcs, dim]

            func_embeds = F.elu(self.func_agg_linear(func_embeds))

            # ---- Function-level attention ----
            att_logits = self.func_att(func_embeds).squeeze(-1)   # [num_funcs]
            att_weights = torch.softmax(att_logits, dim=0)

            func_embeds = func_embeds * att_weights.unsqueeze(-1)

            # ---- Function -> Contract aggregation ----
            h_hier = global_add_pool(
                func_embeds,
                data.func_to_graph_batch,
                size=h_flat.size(0)
            )

        else:
            # fallback if function info is missing
            h_hier = torch.zeros_like(h_flat)

        # ===============================
        # Residual Fusion
        # ===============================
        h_final = h_flat + self.hier_scale * self.hier_proj(h_hier)

        # ===============================
        # Output
        # ===============================
        final_embedding = self.final_linear(h_final)
        final_embedding = F.normalize(final_embedding, p=2, dim=-1)


        if self.training:
            return final_embedding
        else:
            function_vectors_by_graph = [{} for _ in range(data.num_graphs)]
            if func_embeds.shape[0] == 0:
                return final_embedding, function_vectors_by_graph

            local_func_id_counter = [0] * data.num_graphs
            global_to_local_map = {}
            for global_id in range(len(func_embeds)):
                graph_idx = func_to_graph_batch[global_id].item()
                local_id = local_func_id_counter[graph_idx]
                global_to_local_map[global_id] = (graph_idx, local_id)
                local_func_id_counter[graph_idx] += 1


            for global_func_id, func_vec in enumerate(func_embeds):
                graph_idx, local_func_id = global_to_local_map[global_func_id]


                function_vectors_by_graph[graph_idx][local_func_id] = func_vec

            return final_embedding
class FlatGNN(nn.Module):
    """
    """
    def __init__(self, node_feature_dim, gnn_hidden_dim, output_dim, num_gat_layers=3, heads=4):
        super(FlatGNN, self).__init__()

        self.convs = nn.ModuleList()
        current_dim = node_feature_dim
        for i in range(num_gat_layers):
            self.convs.append(GATConv(current_dim, gnn_hidden_dim, heads=heads))
            current_dim = gnn_hidden_dim * heads

        self.final_linear = nn.Linear(current_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = F.elu(conv(x, edge_index))

        graph_embedding = global_mean_pool(x, batch)

        final_embedding = self.final_linear(graph_embedding)
        final_embedding = F.normalize(final_embedding, p=2, dim=-1)

        return final_embedding
