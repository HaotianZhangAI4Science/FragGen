from torch import nn
from models.embed import AtomEmbedding, embed_compose
from models.encoder.complex_encoder import get_encoder
from models.attach import FrontierLayerVN
from models.subcavity import SubCavityPredictor
# from models.classifier import get_field_vn
from models.classifier import FragAtomTypeNet
from models.attach import AttachPoint, check_equality
from models.bond import BondLinker
from models.cartesian_handler import CartesianHandler
from torch.nn import functional as F
from torch_scatter import scatter_softmax
import torch
from models.dehidral_handler import DihedralHandler

class FragmentGeneration(nn.Module):
    
    def __init__(self, config,protein_atom_feature_dim, ligand_atom_feature_dim, frag_atom_feature_dim, num_edge_types=5, num_bond_types=4, num_classes=125, pos_pred_type='dihedral'):
        super().__init__()

        self.config = config
        # define the embedding net
        self.emb_dim = [config.hidden_channels, config.hidden_channels_vec]
        self.protein_atom_emb = AtomEmbedding(protein_atom_feature_dim, 1, *self.emb_dim)
        self.ligand_atom_emb = AtomEmbedding(ligand_atom_feature_dim, 1, *self.emb_dim) # the feature dim truncation is done in the AtomEmbedding, you can try self.ligand_atom_emb.in_scalar to check
        # define the interaction net
        self.encoder = get_encoder(config.encoder, num_edge_types=num_edge_types)  # @num_edge_types need to be changed
        # define the first attach net
        in_sca, in_vec = self.encoder.out_sca, self.encoder.out_vec
        self.frontier_pred = FrontierLayerVN(in_sca=in_sca, in_vec=in_vec, hidden_dim_sca=config.frontier.hidden_dim_sca, hidden_dim_vec=config.frontier.hidden_dim_vec)
        # define the next possible growing point net
        self.cavity_detector = SubCavityPredictor(in_sca=in_sca, in_vec=in_vec, num_filters=[config.subcavity.num_filters]*2, n_component=config.subcavity.n_component)
        # define the next atom/fragment net
        self.type_predictor = FragAtomTypeNet(in_sca=in_sca, in_vec=in_vec, hidden_sca=config.type.num_hidden_sca, hidden_vec=config.type.num_hidden_vec, \
                                                out_class=num_classes, edge_channels=config.type.edge_channels, cutoff=10.0)
        # define the next attach net
        self.attacher = AttachPoint(gnn_num_layer=2, num_atom_type=frag_atom_feature_dim, in_sca=self.type_predictor.out_hidden_sca, in_vec=self.type_predictor.out_hidden_vec, frag_classes=num_classes)
        # define the bond prediction net
        self.bonder = BondLinker([in_sca, in_vec],edge_channels=32, edge_dim=num_bond_types, node_type=8, frag_classes=num_classes, out_class=num_bond_types+1)
        # define the position predition net
        self.pos_pred_type = pos_pred_type
        if pos_pred_type == 'dihedral':
            self.pos_predictor = DihedralHandler(hidden_channels=self.emb_dim, edge_channels=32, num_edge_types=num_edge_types, key_channels=128, num_interactions=2, return_pos=False)
        elif pos_pred_type == 'cartesian':
            self.pos_predictor = CartesianHandler(dim_in=self.emb_dim[0], dim_tmp=self.emb_dim[0], edge_in=num_edge_types, edge_out=config.position.edge_out)
        else:
            raise ValueError("pos_pred_type must be either 'dihedral' or 'cartesians'")
    
    def forward(self,
                compose_feature, compose_pos, idx_ligand, idx_protein,  # input for embed
                compose_knn_edge_index, compose_knn_edge_feature,  # input for interaction
                idx_protein_attch_mask,  # input for the first attach 
                idx_focal,  # input for the position next possible growing point
                pos_subpocket, edge_index_q_cps_knn, edge_index_q_cps_knn_batch,# input for the next atom/fragment
                node_feat_frags, edge_index_frags, edge_features_frags, current_wid, next_motif_wid, node_batch_frags,  # input for the next attach point
                bonded_a_nei_edge_features,bonded_b_nei_edge_features, next_motif_bonded_atom_feature, # # input for the bond prediction
                compose_next_feature, compose_pos_next, idx_ligand_next, idx_protein_next,  
                edge_feature_pos_pred, edge_index_pos_pred,  # input for the position embedding,
                a,b, ligand_idx, b_next, batch_b_next, batch_mol, # input for the dihedral angle prediction
                ligand_pos_mask_idx):

        # first embed the seperate ligand and protein graph
        h_compose = embed_compose(compose_feature, compose_pos, idx_ligand, idx_protein,
            self.ligand_atom_emb, self.protein_atom_emb, self.emb_dim)
        
        # mix up the ligand and protein graph
        h_compose = self.encoder(
            node_attr = h_compose,
            pos = compose_pos,
            edge_index = compose_knn_edge_index,
            edge_feature = compose_knn_edge_feature,
        )

        # predict the first attach point 
        # two circumstances (1) the first attach point is in the ligand 
        #                         (2) the first attach point is in the protein
        y_protein_frontier_pred = self.frontier_pred(
            h_compose,
            idx_protein_attch_mask
        )

        y_frontier_pred = self.frontier_pred(
            h_compose,
            idx_ligand,
        )
        # y_frontier_norm = scatter_softmax(y_frontier_pred,index=ligand_context_pos_batch, dim=0)

        # predict the position next possible growing point
        relative_pos_mu, abs_pos_mu, pos_sigma, pos_pi  = self.cavity_detector(
            h_compose,
            idx_focal,
            compose_pos,
        )

        # predict the next atom/fragment
        y_type_pred = self.type_predictor(
            pos_subpocket,
            compose_pos,
            h_compose,
            edge_index_q_cps_knn, #data['edge_new_site_knn_index']
            edge_index_q_cps_knn_batch
        )
        focal_info = h_compose[0][idx_focal], h_compose[1][idx_focal]

        # predict the next attach point
        # two circumstances (1) for fragment growing (symmetric ones are considered)
        #                   (2) for atom
        # if ligand_pos_mask_idx.shape[0] == 1: # for atom, the attach point is the atom itself
            # frag_node_2d = torch.tensor(float('nan'))

        if check_equality(node_feat_frags) & check_equality(edge_features_frags): # for the symmetric fragment, the attachment point is random
            frag_node_2d = torch.tensor(float('nan'))
        else:
            frag_node_2d = self.attacher(node_feat_frags, edge_index_frags, edge_features_frags, 
                    current_wid, next_motif_wid, focal_info, node_batch_frags)
            #frag_node_2d_norm = scatter_softmax(frag_node_2d,index=node_batch_frags, dim=0)
        # predict the bond type     

        if idx_ligand.shape[0] == 0: # for the full mask, there is no bond
            bond_pred = torch.tensor(float('nan'))
        else:   

            bond_pred = self.bonder(focal_info, compose_pos[idx_focal], pos_subpocket, current_wid, next_motif_wid, \
                                    next_motif_bonded_atom_feature, bonded_a_nei_edge_features, bonded_b_nei_edge_features)
            # bond_pred_norm = F.softmax(bond_pred, dim=1)
        # predict the position 
        # two circumstances (1) for fragment
        #                   (2) for atom
        # CartesianHandler
        
        h_compose_pos_next_pred = embed_compose(compose_next_feature, compose_pos_next, idx_ligand_next, idx_protein_next,
                                    self.ligand_atom_emb, self.protein_atom_emb, self.emb_dim)

        if self.pos_pred_type == 'dihedral':
            alpha = self.pos_predictor(h_compose_pos_next_pred, compose_pos_next, edge_index_pos_pred, edge_feature_pos_pred, \
                        a, b, ligand_idx, batch_mol, b_next, batch_b_next)
            updated_pos = None
        elif self.pos_pred_type == 'cartesian':
            _, _, _, updated_pos = self.pos_predictor(h_compose_pos_next_pred[0], edge_feature_pos_pred, edge_index_pos_pred, \
                compose_pos_next, ligand_pos_mask_idx, update_pos=True)
            alpha = None

        

        return y_protein_frontier_pred, y_frontier_pred,\
            abs_pos_mu, pos_sigma, pos_pi,\
            y_type_pred,\
            frag_node_2d,\
            bond_pred,\
            alpha, \
            updated_pos