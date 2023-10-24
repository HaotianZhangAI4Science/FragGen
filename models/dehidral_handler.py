from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import math
from torch import nn
from models.encoder.complex_encoder import AttentionInteractionBlockVN
from torch_scatter import scatter_add
from utils.geom_utils import rotate_batch_matrix_around_axis, batched_rotate_around_center
from models.vanilla import MLP

class DihedralHandler(nn.Module):
    def __init__(self, hidden_channels=[256, 64], interactions=None, edge_channels=32, num_edge_types=5, key_channels=128,num_interactions=2, return_pos=True):
        super().__init__()
        self.return_pos = return_pos
        self.interactions = nn.ModuleList()
        
        if interactions:
            self.interactions = interactions
        else:
            for _ in range(num_interactions):
                block = AttentionInteractionBlockVN(
                    hidden_channels=hidden_channels,
                    edge_channels=edge_channels,
                    num_edge_types=num_edge_types,
                    key_channels=key_channels
                )
                self.interactions.append(block)

    
        self.torsion_mlp = MLP(in_dim=hidden_channels[0]*3 , out_dim=1, num_layers=2)


    def forward(self, pl_node_attr, pl_pos, pl_edge_index, pl_edge_feature, a, b, ligand_idx, batch_mol, b_next=None, batch_b_next=None):
        # ligand_idx for selct the ligand in pl_pos and pl_node_attr
        # batch is the batch index for the ligand
        
        # structure interaction
        pl_edge_vector = pl_pos[pl_edge_index[0]] - pl_pos[pl_edge_index[1]]
        h = list(pl_node_attr)
        for interaction in self.interactions:
            delta_h = interaction(h, pl_edge_index, pl_edge_feature, pl_edge_vector)
            h[0] = h[0] + delta_h[0]
            h[1] = h[1] + delta_h[1]
    
        # compute the dihedral angle
        a_feat = h[0][a]
        b_feat = h[0][b]
        num_mol = a.shape[0]
        mol_feat = scatter_add(h[0][ligand_idx], batch_mol, dim=0, dim_size=num_mol)
        alpha_feat = torch.cat([a_feat, b_feat, mol_feat], dim=-1)
        alpha = self.torsion_mlp(alpha_feat)

        if self.return_pos:
            pos_updated = self.pos_update(alpha, pl_pos[a], pl_pos[b], pl_pos[b_next], batch_b_next)
            return alpha, pos_updated
        
        return alpha

    def pos_update(self, alpha, a_pos, b_pos, b_next_pos, batch_b_next):
        vec = a_pos - b_pos
        rotate_matrix = rotate_batch_matrix_around_axis(vec, alpha)
        predicted_pos = batched_rotate_around_center(rotate_matrix, a_pos, b_next_pos, batch_b_next)
        
        return predicted_pos


