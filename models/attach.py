# predict the 
from models.geom_block import GVPerceptronVN, GVLinear
from torch.nn import Module, Sequential
from models.gat import GATConv
from models.vanilla import MLP
from torch import nn
import torch
from torch.nn import functional as F

class AttachPoint(nn.Module):
    def __init__(self, gnn_num_layer=2, num_atom_type=45, in_sca=128, in_vec=32, frag_classes=125):
        super(AttachPoint, self).__init__()
        '''
        Basically, this model utilize the fragment information and all the possible attach points to predict the next attach point
        Input:
            node_feat_frags, edge_index_frags, edge_features: the 2D fragment information. (num_frag_nodes, 2d_feature_dim)
            next_site_hidden: the hidden state of the next attach point, which is the intermidiate output of the type prediction. [(1, sca_dim),(1,vec_dim)]
            current_site_hidden: the hidden state of the current attach point, which is the initial h_compose. (1, sca_dim)
        Output:
            (num_frag_nodes, 1)
        '''
        self.attacher = GNNAttach(num_layer=gnn_num_layer, num_atom_type=num_atom_type, emb_dim=in_sca)
        self.fragment_embedding = nn.Embedding(frag_classes + 1, in_sca)
        self.attach_site_map = GVPerceptronVN(in_scalar=in_sca, in_vector=in_vec, out_scalar=in_sca, out_vector=32)
        self.predicter_attach_hidden = MLP(in_dim=4*in_sca, out_dim=1, num_layers=1)
        self.focal_net = GVPerceptronVN(256, 64, in_sca, in_vec)
        
    def forward(self,
            node_feat_frags, edge_index_frags, edge_features,\
            current_wid, next_motif_wid,
            focal_info,
            node_batch_frags
            ):
        frag_node_2d = self.attacher(node_feat_frags, edge_index_frags, edge_features)
        num_nodes = frag_node_2d.shape[0]

        current_frag_emb = self.fragment_embedding(current_wid)
        next_frag_emd = self.fragment_embedding(next_motif_wid)
        focal_info_mapped = self.focal_net(focal_info)
        next_attach_hidden = torch.concat([frag_node_2d,\
            current_frag_emb[node_batch_frags],next_frag_emd[node_batch_frags],\
             focal_info_mapped[0][node_batch_frags]], axis=1)
        # h_compose[0][idx_focal]
        next_site_pred = self.predicter_attach_hidden(next_attach_hidden)
        return next_site_pred

def check_equality(tensor_list):
    # check whether the next fragment is equal
    if len(tensor_list) == 0:
        return False

    first_tensor = tensor_list[0]
    for tensor in tensor_list[1:]:
        if not torch.allclose(first_tensor, tensor):
            return False
    return True

class FrontierLayerVN(Module):
    def __init__(self, in_sca, in_vec, hidden_dim_sca, hidden_dim_vec):
        super().__init__()
        self.net = Sequential(
            GVPerceptronVN(in_sca, in_vec, hidden_dim_sca, hidden_dim_vec),
            GVLinear(hidden_dim_sca, hidden_dim_vec, 1, 1)
        )

    def forward(self, h_att, idx_ligans):
        h_att_ligand = [h_att[0][idx_ligans], h_att[1][idx_ligans]]
        pred = self.net(h_att_ligand)
        pred = pred[0]
        return pred

class GNNAttach(nn.Module):
    def __init__(self, num_layer, num_atom_type, emb_dim=128, JK = "last", drop_ratio = 0):
        super(GNNAttach, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # if self.num_layer < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_embedding = nn.Linear(num_atom_type, emb_dim)
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GATConv(emb_dim))

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
    
    def forward(self, x_frag, edge_index_frag, edge_attr_frag):

        x_frag = self.node_embedding(x_frag)
        h_list = [x_frag]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index_frag, edge_attr_frag)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        if self.JK == 'last':
            frag_node_feat2_2d = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            frag_node_feat2_2d = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]
            
        return frag_node_feat2_2d
    