from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add

class GATConv(nn.Module):

    def __init__(self, emb_dim, heads=2, num_bond_type=5, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__()
        self.aggr = aggr
        self.num_bond_type = num_bond_type
        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding = nn.Linear(self.num_bond_type, heads * emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.reset_parameters()

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        norm = self.norm(edge_index, x.size(0), x.dtype)
        edge_embeddings = self.edge_embedding(edge_attr)

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x_i = torch.index_select(x, 0, edge_index_i)
        x_j = torch.index_select(x, 0, edge_index_j)

        # Message
        edge_attr = edge_embeddings.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i)
        messages = x_j * alpha.view(-1, self.heads, 1)

        # Aggregate messages to nodes
        out = scatter_add(messages, edge_index_i, dim=0, dim_size=x.size(0))

        # Update node features
        out = out.mean(dim=1)
        out = out + self.bias

        return out



# if __name__ == __main__:

    # attacher = GNNAttach(num_layer=2, num_atom_type=45, emb_dim=128)
    # gat = attacher.gnns[0]
    # x = node_feat_frags
    # x = attacher.node_embedding(x)
    # h_list = [x]
    # edge_index = edge_index_frags
    # edge_attr = edge_features