from torch import nn
import torch
from models.geom_block import GVLinear
from models.vanilla import MLP
from torch.nn import functional as F

class BondLinker(nn.Module):
    
    def __init__(self, hidden_channels=[256, 64], edge_channels=32, edge_dim = 4, node_type=8, frag_classes=125, out_class=4):
        super().__init__()
        self.node_type = node_type
        self.edge_channels = edge_channels
        self.edge_dim = edge_dim
        self.out_class = out_class

        self.sigmoid = nn.Sigmoid()
        self.fragment_embedding = nn.Embedding(frag_classes + 1, edge_channels)
        #self.edge_vec_map = EdgeExpansion(edge_channels)
        self.edge_sca_map = nn.Linear(1, edge_channels)
        self.focal_net = GVLinear(hidden_channels[0], hidden_channels[1],\
                                  edge_channels, edge_channels)
        
        self.valance_net = nn.Linear(edge_dim,edge_channels//2)
        self.next_atom_info_net = nn.Linear(node_type, edge_channels)
        self.bond_predictor = MLP(in_dim=6*edge_channels, out_dim=out_class, num_layers=1)

    def forward(self, focal_feat, focal_pos, next_site_pos,current_wid, next_motif_wid, next_bonded_atom_feature, \
                bonded_a_nei_edge_features, bonded_b_nei_edge_features):
        
        focal_info, _ = self.focal_net(focal_feat)
        bond_vec = next_site_pos - focal_pos
        bond_dist = torch.norm(bond_vec, dim=1).reshape(-1,1)

        bond_sca_info = self.edge_sca_map(bond_dist)
        current_frag_emb = self.fragment_embedding(current_wid)
        next_frag_emb = self.fragment_embedding(next_motif_wid)

        next_atom_info = self.next_atom_info_net(next_bonded_atom_feature)
        bond_2d_node_hidden = torch.cat([self.valance_net(bonded_a_nei_edge_features), self.valance_net(bonded_b_nei_edge_features)], dim=-1)
        bond_info = torch.cat([focal_info, bond_sca_info, current_frag_emb, next_frag_emb, next_atom_info, bond_2d_node_hidden], dim=1)
        
        bond_pred = self.bond_predictor(bond_info)

        return bond_pred
    


# from torch_geometric.nn import GATConv
# Previous version
# class GAT(torch.nn.Module):
#     '''
#     This GraphAttention Model is the node version, which is distinct from the gat
#     '''
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
#         self.conv2 = GATConv(8 * 8, out_channels, dropout=0.6)

#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)


# class BondLinker(nn.Module):
#     def __init__(self, in_sca, edge_dim=5, atom_class=19, out_class=4):
#         super().__init__()
#         self.bond_info_net = MLP(in_dim = edge_dim, out_dim=in_sca//2, num_layers=1) # in_sca//2=64
#         self.mol2dgnn = GAT(atom_class,in_sca//2)
#         self.bond_predictor = MLP(in_dim=2*in_sca, out_dim=out_class, num_layers=1)


#     def forward(self, context_next_node_feature, context_next_edge_index,
#         bond_query,a_bond_info,b_bond_info):

#         # use 2d GNN extract the chemical composition information, and then select the query bond
#         context_next_2d_hidden = self.mol2dgnn(context_next_node_feature, context_next_edge_index)
#         bond_2d_node_hidden = torch.cat([context_next_2d_hidden[bond_query[0]], context_next_2d_hidden[bond_query[1]]], dim=-1)
    

#         # # use the neighbor bonding information to strengthen the bond prediction
#         # associated_edges = search_neighbor(context_next_edge_index, bond_query)
#         # associated_a_edge_feature = context_next_edge_feature[associated_edges[0]]
#         # associated_b_edge_feature = context_next_edge_feature[associated_edges[1]]
#         # a_bond_info = torch.sum(associated_a_edge_feature, dim=0).unsqueeze(0)
#         # b_bond_info = torch.sum(associated_b_edge_feature, dim=0).unsqueeze(0)
#         bond_info_hidden = torch.cat([self.bond_info_net(a_bond_info), self.bond_info_net(b_bond_info)], dim=1)
#         bond_hidden = torch.concat([bond_2d_node_hidden, bond_info_hidden], axis=1)
#         bond_pred = self.bond_predictor(bond_hidden)
#         return bond_pred