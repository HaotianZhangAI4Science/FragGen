import torch
import torch.nn.functional as F
from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn import knn, radius
from utils.geom_utils import rotate_matrix_around_axis, rotate_axis_w_centered_point
from torch_geometric.utils.subgraph import subgraph
from utils.frag import get_fragment_smiles, find_bonds_in_mol, query_clique
import random
from rdkit import Chem
from utils.cluster import get_bfs_perm
import numpy as np
from utils.chem import transfer_conformers, filter_possible_match
import copy
from utils.featurizer import featurize_frag
from copy import deepcopy
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
import torch.nn.functional as F
from utils.dihedral_utils import batch_dihedrals, rand_rotate

def remove_edges(edge_index, edge_features, edges_to_remove):
    remove_indices = []
    for i in range(edge_index.size(1)):
        edge = edge_index[:, i]
        if any(torch.equal(edge, e) or torch.equal(edge.flip(0), e) for e in edges_to_remove):
            remove_indices.append(i)

    # Filter out the identified edges and their features
    keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    keep_mask[remove_indices] = False

    edge_index = edge_index[:, keep_mask]
    edge_features = edge_features[keep_mask]

    return edge_index, edge_features

class FeaturizeLigandAtom(object):
    
    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1,6,7,8,9,15,16,17])  # H C N O F P S Cl
        self.atomic_numbers = torch.LongTensor([6,7,8,9,15,16,17,35])  # C N O F P S Cl,Br
        # assert len(self.atomic_numbers) == 7, NotImplementedError('fix the staticmethod: chagne_bond')
        # 15
    # @property
    # def num_properties(self):
        # return len(ATOM_FAMILIES)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + (1 + 1 + 1) + 4 # 8+3+4=15

    def __call__(self, data):
        element = data['ligand_element'].view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        # chem_feature = data.ligand_atom_feature
        is_mol_atom = torch.ones([len(element), 1], dtype=torch.long)
        n_neigh = data['ligand_num_neighbors'].view(-1, 1)
        n_valence = data['ligand_atom_valence'].view(-1, 1)
        ligand_atom_num_bonds = data['ligand_atom_num_bonds']
        # x = torch.cat([element, chem_feature, ], dim=-1)
        x = torch.cat([element, is_mol_atom, n_neigh, n_valence, ligand_atom_num_bonds], dim=-1)
        data['ligand_atom_feature_full'] = x
        return data

class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand_bond_feature = F.one_hot(data['ligand_bond_type'] -1, num_classes=4)    # (1,2,3) to (0,1,2)-onehot
        return data
    
    
class LigandCountNeighbors(object):
    '''
    This is designed to count the neighbors, valence, and the bonded atoms of each atom in the ligand.
    For the full ligand, it can be replaced by the rdkit function.
    However, in our generation context, we should consider the partial ligand effect, thus explicitly count the neighbors is favorable.
    '''
    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()
    
    @staticmethod
    def change_features_of_neigh(ligand_feature_full, new_num_neigh, new_num_valence, ligand_atom_num_bonds, atom_types=8):
        ligand_feature_full_new = deepcopy(ligand_feature_full)
        idx_n_neigh = atom_types + 1
        idx_n_valence = idx_n_neigh + 1
        idx_n_bonds = idx_n_valence + 1
        ligand_feature_full_new[:, idx_n_neigh] = new_num_neigh.long()
        ligand_feature_full_new[:, idx_n_valence] = new_num_valence.long()
        ligand_feature_full_new[:, idx_n_bonds:idx_n_bonds+4] = ligand_atom_num_bonds.long()
        return ligand_feature_full_new
    
    @staticmethod
    def change_atom_bonded_features(ligand_feature_full, new_bond_index, new_bond_types, num_nodes):
        if len(new_bond_types.shape) != 1:
            new_bond_types = torch.argmax(new_bond_types, dim=1)+1
        ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
            new_bond_index,
            symmetry=True,
            num_nodes = num_nodes,
        )
        ligand_context_valence = LigandCountNeighbors.count_neighbors(
            new_bond_index,
            symmetry=True,
            valence=new_bond_types,
            num_nodes = num_nodes,
        )
        ligand_context_num_bonds = torch.stack([
            LigandCountNeighbors.count_neighbors(
                new_bond_index,
                symmetry=True,
                valence=(new_bond_types == i),
                num_nodes = num_nodes,
            ) for i in [1, 2, 3, 4]    
        ], dim = -1)
        ligand_feature_full_changed = LigandCountNeighbors.change_features_of_neigh(
            ligand_feature_full,
            ligand_context_num_neighbors,
            ligand_context_valence,
            ligand_context_num_bonds
        )
        return ligand_feature_full_changed

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data['ligand_num_neighbors'] = self.count_neighbors(
            data['ligand_bond_index'], 
            symmetry=True,
            num_nodes=data['ligand_element'].size(0),
        )
        data['ligand_atom_valence'] = self.count_neighbors(
            data['ligand_bond_index'], 
            symmetry=True, 
            valence=data['ligand_bond_type'],
            num_nodes=data['ligand_element'].size(0),
        )
        data['ligand_atom_num_bonds'] = torch.stack([
            self.count_neighbors(
                data['ligand_bond_index'], 
                symmetry=True, 
                valence=(data['ligand_bond_type'] == i).long(),
                num_nodes=data['ligand_element'].size(0),
            ) for i in [1, 2, 3, 4]
        ], dim = -1)
        return data
    
class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.max_num_aa = 20
        #self.atomic_numbers = torch.LongTensor([6,7,8,9,15,16,17,35])
    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1 +1

    def __call__(self, data):
        element = data['protein_element'].view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        if data['protein_atom_to_aa_type'].dtype != torch.int64:
            data['protein_atom_to_aa_type'] = data['protein_atom_to_aa_type'].to(torch.int64)
        amino_acid = F.one_hot(data['protein_atom_to_aa_type'], num_classes=self.max_num_aa)
        is_backbone = data['protein_is_backbone'].view(-1, 1).long()
        is_mol_atom = torch.zeros_like(is_backbone, dtype=torch.long)
        x = torch.cat([element, amino_acid, is_backbone, is_mol_atom], dim=-1)
        data['protein_atom_feature'] = x
        # data['alpha_carbon_indicator'] = torch.tensor([True if name =="CA" else False for name in data['protein_atom_name']])
        return data


def search_neighbor(edge_index, query_bond_index):
    '''
    return the edge_idx of edge_index, note it is not the node_idx
    I had been confused after the development (LOL) hope you will not
    '''
    # Finding edges connected to the first query element
    connected_to_first = (edge_index == query_bond_index[0]).any(dim=0)
    
    # Finding edges connected to the second query element
    connected_to_second = (edge_index == query_bond_index[1]).any(dim=0)
    
    # Finding the direct edge between the query_bond_index
    exact_bond_index = ((edge_index[0] == query_bond_index[0]) & (edge_index[1] == query_bond_index[1])) \
                        | ((edge_index[0] == query_bond_index[1]) & (edge_index[1] == query_bond_index[0]))

    # Get indices of the edges connected to the first and second query elements, excluding the direct edge
    indices_connected_to_first = (connected_to_first & ~exact_bond_index).nonzero().squeeze()
    indices_connected_to_second = (connected_to_second & ~exact_bond_index).nonzero().squeeze()

    return indices_connected_to_first, indices_connected_to_second

bondmaps={'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 4}
class LigandBFSMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=1, frag_base=None):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        self.frag_base = frag_base
        self.frag_component = frag_base['data_base_features'].shape[0]

    def __call__(self, data):
        # global context_motif_ids
        # global bfs_perm
        # global num_masked
        # global context_idx
        # global next_motif_mol
        # global mol
        bfs_perm, bfs_focal = get_bfs_perm(data['cluster_mol'], self.frag_base)
        mol = data['cluster_mol'].mol
        Chem.SanitizeMol(mol)
        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_motifs = len(bfs_perm)
        num_masked = int(num_motifs * ratio)
        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_motifs - num_masked) < self.min_num_unmasked:
            num_masked = num_motifs - self.min_num_unmasked
        num_unmasked = num_motifs - num_masked
        context_motif_ids = bfs_perm[:-num_masked]
        context_idx = set() 
        for i in context_motif_ids:
            context_idx = context_idx | set(data['cluster_mol'].nodes[i].clique_composition)
        context_idx = torch.LongTensor(list(context_idx))
        # print('num_masked',num_masked)
        current_motif = data['cluster_mol'].nodes[bfs_focal[-num_masked]]
        next_motif = data['cluster_mol'].nodes[bfs_perm[-num_masked]]

        # data['context_idx'] = context_idx.numpy().tolist()
        # data['context_next_idx'] = context_idx.numpy().tolist()+next_motif.clique_composition
        # data['current_idx'] =current_motif.clique_composition
        
        data['current_wid'] = torch.tensor([current_motif.wid])
        data['next_motif_wid'] = torch.tensor([next_motif.wid])  # For Prediction
        current_atoms = current_motif.clique_composition
        data['current_atoms'] = torch.cat([torch.where(context_idx == i)[0] for i in current_atoms]) #+ len(data['protein_pos'])
        
        data['ligand_context_element'] = data['ligand_element'][context_idx]
        data['ligand_context_feature_full'] = data['ligand_atom_feature_full'][context_idx]  # For Input
        data['ligand_context_pos'] = data['ligand_pos'][context_idx]
        data['ligand_center'] = torch.mean(data['ligand_pos'], dim=0)
        data['num_atoms'] = torch.tensor([len(context_idx) + len(data['protein_pos'])])
    
        # next motif is an atom
        if len(next_motif.clique_composition) == 1:
            next_smi = self.frag_base['data_base_smiles'][next_motif.wid].item().decode()
            next_motif_mol = Chem.MolFromSmiles(next_smi)
            Chem.SanitizeMol(next_motif_mol)
            bond = find_bonds_in_mol(mol,current_motif.clique_composition, next_motif.clique_composition)[0] 
            next_bonded_atom_index_in_mol = list(set(bond)- set(current_motif.clique_composition))[0]
            current_bonded_atom_idx_in_mol = list(set(bond)- set(next_motif.clique_composition))[0]
            next_bonded_atom_index_in_frag = 0
            next_site_attach_pos = data['ligand_pos'][next_bonded_atom_index_in_mol].reshape(-1,3)
            data['next_site_attach'] = torch.tensor([next_bonded_atom_index_in_frag], dtype=torch.long)
            data['next_site_attach_pos'] = next_site_attach_pos

        # next motif is a fragment
        else: 
            # attachments of current and next fragment prediction
            next_smi = self.frag_base['data_base_smiles'][next_motif.wid].item().decode()
            next_motif_mol = Chem.MolFromSmiles(next_smi)
            Chem.SanitizeMol(next_motif_mol)
            bond = find_bonds_in_mol(mol,current_motif.clique_composition, next_motif.clique_composition)[0]
            next_bonded_atom_index_in_mol = list(set(bond)- set(current_motif.clique_composition))[0]
            # the possible match means that there are several possible frag to mol matcher, we select the one containing the next_bonded_atom
            match = filter_possible_match(mol, next_motif_mol, next_bonded_atom_index_in_mol) 
            next_bonded_atom_index_in_frag = list(match).index(next_bonded_atom_index_in_mol)  # find the attachment site of the new frag conformer
            # query from data['ligand_pos] is more robust than reading from the next_motif_mol, since the some frags are symmetrical, messing up the exact match
            next_site_attach_pos = data['ligand_pos'][next_bonded_atom_index_in_mol].reshape(-1,3)
            current_bonded_atom_idx_in_mol = list(set(bond)- set(next_motif.clique_composition))[0]
            data['next_site_attach'] = torch.tensor([next_bonded_atom_index_in_frag], dtype=torch.long)
            data['next_site_attach_pos'] = next_site_attach_pos.reshape(-1,3)
        
        data['next_motif_mol'] = next_motif_mol
        data['next_bond'] = torch.tensor(bondmaps[str(mol.GetBondBetweenAtoms(*bond).GetBondType())]).reshape(-1,1)
        focal_id_in_context = torch.where(context_idx==current_bonded_atom_idx_in_mol)[0]
        data['focal_id_in_context'] = focal_id_in_context
        context_w_next_mol_idx = torch.LongTensor(context_idx.numpy().tolist()+next_motif.clique_composition)
        # create the bond index
        a_bond, b_bond = find_bonds_in_mol(mol, current_motif.clique_composition, next_motif.clique_composition)[0]
        if a_bond in context_idx: # suppose a_idx is the focal atom
            pass
        else:
            tmp_idx = a_bond
            a_bond = b_bond
            b_bond = tmp_idx
        a_idx = context_w_next_mol_idx.tolist().index(a_bond)  # reindex in the context_w_next_mol
        b_idx = context_w_next_mol_idx.tolist().index(b_bond)  # reindex in the context_w_next_mol
        data['ligand_context_next_bond_pred_index'] = torch.tensor([[a_idx, b_idx]], dtype=torch.int64).T
        data['ligand_context_next_feature_full'] = data['ligand_atom_feature_full'][context_w_next_mol_idx]
        data['ligand_context_next_pos'] = data['ligand_pos'][context_w_next_mol_idx]

        pos_mask = ~torch.tensor([element.item() in context_idx for element in context_w_next_mol_idx])
        #pos_mask=tensor([False, False, False, False, False, False, False, False, False, False,
        #False, False, False, False, False, False, False, False, False, False,
        #True])

        data['next_motif_bonded_atom_feature'] = data['ligand_context_next_feature_full'][b_idx].reshape(1,-1)[:,:8]
        data['ligand_pos_mask'] = pos_mask

        node_feat_frags, edge_index_frags, edge_features_frags = featurize_frag(next_motif_mol)
        data['node_feat_frags'] = node_feat_frags
        data['edge_index_frags'] = edge_index_frags
        data['edge_features_frags'] = edge_features_frags
        data['current_attach_pos'] = data['ligand_pos'][current_bonded_atom_idx_in_mol]

        data['ligand_context_bond_index'], data['ligand_context_bond_feature'] = subgraph(
            context_idx,
            data['ligand_bond_index'],
            edge_attr=data['ligand_bond_feature'],
            relabel_nodes=True,
        )
        # global context_next_idx
        context_next_idx = context_idx.numpy().tolist()+next_motif.clique_composition
        data['ligand_context_next_bond_index'], data['ligand_context_next_bond_feature'] = subgraph(
            context_next_idx,
            data['ligand_bond_index'],
            edge_attr=data['ligand_bond_feature'],
            relabel_nodes=True)

        associated_edges = search_neighbor(data['ligand_context_next_bond_index'], data['ligand_context_next_bond_pred_index'])
        data['bonded_a_nei_edge_features'] = torch.sum(data['ligand_context_next_bond_feature'][associated_edges[0]], dim=0).unsqueeze(0)
        data['bonded_b_nei_edge_features'] = torch.sum(data['ligand_context_next_bond_feature'][associated_edges[1]], dim=0).unsqueeze(0)
        data['focal_id_ligand'] = focal_id_in_context
        # a_nei_with_a = data['ligand_context_next_bond_index'][:,associated_edges[0]][0].unique()
        # a_nei = a_nei_with_a[a_nei_with_a != data['ligand_context_next_bond_pred_index'][0]]
        # b_nei_with_b = data['ligand_context_next_bond_index'][:,associated_edges[1]][1].unique()
        # b_nei = b_nei_with_b[b_nei_with_b != data['ligand_context_next_bond_pred_index'][1]]
        # data['a'] = data['ligand_context_next_bond_pred_index'][0]
        # data['b'] = data['ligand_context_next_bond_pred_index'][1]
        # data['a_neigh'] = a_nei
        # data['b_neigh'] = b_nei
        # data['idx_protein_all_mask'] = torch.empty(0, dtype=torch.long)
        # data['y_protein_frontier'] = torch.empty(0, dtype=torch.bool)
        data['mask'] = 'bfs'
        
        # change the feature of the bonded atoms
        data['ligand_context_feature_full'] = LigandCountNeighbors.change_atom_bonded_features(data['ligand_context_feature_full'], 
                                                                          data['ligand_context_bond_index'],
                                                                            data['ligand_context_bond_feature'],
                                                                              num_nodes=len(context_idx))
        
        data['ligand_context_next_feature_full'] = LigandCountNeighbors.change_atom_bonded_features(data['ligand_context_next_feature_full'], 
                                                                          data['ligand_context_next_bond_index'],
                                                                            data['ligand_context_next_bond_feature'],
                                                                              num_nodes=len(context_next_idx))
        
        return data

class FullMask(object):
    def __init__(self, frag_base):
        self.frag_base = frag_base
        self.frag_component = frag_base['data_base_features'].shape[0]
    
    def __call__(self, data):
        
        # the first predicted motif is determined by the distance
        for i, node in enumerate(data['cluster_mol'].nodes):
            query_idx = query_clique(data['cluster_mol'].mol, node.clique_composition, **self.frag_base)
            node.nid = i
            node.wid = query_idx

        mol = data['cluster_mol'].mol
        next_motif = data['cluster_mol'].nodes[0]
        context_idx = torch.LongTensor([])

        data['current_wid'] = torch.tensor([self.frag_component])
        data['current_atoms'] = torch.tensor([data['protein_contact_idx']])
        data['next_motif_wid'] = torch.tensor([next_motif.wid])
        
        data['ligand_context_element'] = data['ligand_element'][context_idx]
        data['ligand_context_feature_full'] = data['ligand_atom_feature_full'][context_idx]  # For Input
        data['ligand_context_pos'] = data['ligand_pos'][context_idx]
        data['ligand_center'] = torch.mean(data['ligand_pos'], dim=0)
        data['num_atoms'] = torch.tensor([len(data['ligand_pos']) + len(data['protein_pos'])])

        # data['next_motif_pos'] = data['ligand_pos'][next_motif.clique_composition]
        # if len(data['next_motif_pos'].shape) == 1: # next_motif is an atom
        #     data['next_motif_pos'] = data['next_motif_pos'].reshape(-1,3)
        next_smi = self.frag_base['data_base_smiles'][next_motif.wid].item().decode()
        next_motif_mol = Chem.MolFromSmiles(next_smi)
        data['next_motif_mol'] = next_motif_mol

        if len(next_motif.clique_composition) == 1: # next_motif is an atom

            data['next_site_attach'] = torch.tensor([0], dtype=torch.long)
            data['next_site_attach_pos'] = data['ligand_pos'][next_motif.clique_composition].reshape(-1,3)
            
        else:
            match = filter_possible_match(mol, next_motif_mol, next_motif.min_idx)
            data['next_site_attach'] = torch.tensor([match.index(next_motif.min_idx)], dtype=torch.long)
            data['next_site_attach_pos'] = data['ligand_pos'][next_motif.min_idx].reshape(-1,3)
            # data['next_motif_pos'] = torch.tensor(next_motif_mol.GetConformer().GetPositions(), dtype=torch.float32)
        
        context_w_next_mol_idx = torch.LongTensor(context_idx.numpy().tolist()+next_motif.clique_composition)

        node_feat_frags, edge_index_frags, edge_features_frags = featurize_frag(next_motif_mol)
        data['node_feat_frags'] = node_feat_frags
        data['edge_index_frags'] = edge_index_frags
        data['edge_features_frags'] = edge_features_frags
        data['ligand_context_next_feature_full'] = data['ligand_atom_feature_full'][context_w_next_mol_idx]
        data['ligand_context_next_pos'] = data['ligand_pos'][context_w_next_mol_idx]
        # ligand_context_w_next_mol = Chem.MolFromSmiles(get_fragment_smiles(mol,context_idx.numpy().tolist()+next_motif.clique_composition))
        # _, full_conformer = transfer_conformers(ligand_context_w_next_mol, mol)
        # ligand_context_w_next_mol.AddConformer(full_conformer)
        # data['ligand_context_w_next_mol'] = ligand_context_w_next_mol
        pos_mask = ~torch.tensor([element.item() in context_idx for element in context_w_next_mol_idx])
        data['ligand_pos_mask'] = pos_mask
        data['ligand_context_next_bond_index'], data['ligand_context_next_bond_feature'] = subgraph(
            context_idx.numpy().tolist()+next_motif.clique_composition,
            data['ligand_bond_index'],
            edge_attr=data['ligand_bond_feature'],
            relabel_nodes=True,
        ) # this is the reduced smiles, just for visualization, no mapping realtionship with the training data
        data['current_attach_pos'] = data['protein_pos'][data['protein_contact_idx']]
        data['focal_id_in_context'] = data['protein_contact_idx'].reshape(1)
        data['focal_id_ligand'] = torch.empty(0, dtype=torch.int64)
        data['next_bond'] = torch.zeros(1).reshape(-1,1)
        data['ligand_context_bond_index'] = torch.empty(2,0, dtype=torch.int32)
        data['ligand_context_bond_feature'] = torch.empty(0,4, dtype=torch.float32)

        # bond predi ction
        data['ligand_context_next_bond_pred_index'] = torch.empty(2,0, dtype=torch.int32)
        data['bonded_a_nei_edge_features'] = torch.zeros(4, dtype=torch.float32).reshape(1,-1)
        data['bonded_b_nei_edge_features'] = torch.zeros(4, dtype=torch.float32).reshape(1,-1)
        data['next_motif_bonded_atom_feature'] = torch.zeros(8, dtype=torch.float32).reshape(1,-1)
        data['mask'] = 'full'

        return data

class ComplexBuilder(object):
    def __init__(self, protein_dim=26, ligand_dim=45, knn=36, knn_pos_pred=None):
        super().__init__()
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.knn = knn  
        if knn_pos_pred is None:
            self.knn_pos_pred = knn
    
    def __call__(self, data):

        ligand_context_pos = data['ligand_context_pos']
        ligand_context_featrure_full = data['ligand_context_feature_full']
        protein_pos = data['protein_pos']
        protein_feature = data['protein_atom_feature']
        len_ligand_ctx = len(ligand_context_pos)
        len_protein = len(protein_pos)

        data['compose_pos'] = torch.cat([ligand_context_pos, protein_pos], dim=0).to(torch.float32)
        len_compose = len_ligand_ctx + len_protein
        # protein_feature_expended = torch.cat([
        #     protein_feature,torch.zeros([len_protein, 45-26], dtype=torch.long)
        # ], dim=1)

        ligand_context_featrure_expanded = torch.cat([
            ligand_context_featrure_full,torch.zeros([len_ligand_ctx, 27-15], dtype=torch.long)
        ], dim=1)

        data['compose_feature'] = torch.cat([ligand_context_featrure_expanded, protein_feature], dim=0)
        data['idx_ligand_ctx_in_compose'] = torch.arange(len_ligand_ctx, dtype=torch.long)
        data['idx_protein_in_compose'] = torch.arange(len_ligand_ctx, len_compose, dtype=torch.long)
        data = self.get_knn_graph(data, self.knn, len_ligand_ctx, len_compose, num_workers=16)
        
        if data['mask'] == 'full':
            data['idx_protein_attch_mask'] = data['idx_protein_in_compose']
            y_protein_attach = torch.zeros_like(data['idx_protein_attch_mask'], dtype=torch.bool)
            y_protein_attach[torch.unique(data['protein_contact_idx'])] = True
            data['y_protein_attach'] = y_protein_attach
        else:
            data['idx_protein_attch_mask'] = torch.empty(0, dtype=torch.long)
            data['y_protein_attach'] = torch.empty(0, dtype=torch.bool)
        
        # build the edge_new_site_knn_index for predicting the atom/frag type
        if data['mask'] == 'full' or data['mask'] == 'bfs':
            data['edge_new_site_knn_index'] = knn(x=data['compose_pos'], y=data['next_site_attach_pos'], k=self.knn, num_workers=16)
            
        return data

    @staticmethod
    def get_knn_graph(data, num_knn, len_ligand_ctx, len_compose, num_workers=1, num_edge_types=4):
        data['compose_knn_edge_index'] = knn_graph(data['compose_pos'], num_knn, flow='target_to_source', num_workers=num_workers)

        id_compose_edge = data['compose_knn_edge_index'][0, :len_ligand_ctx*num_knn] * len_compose + data['compose_knn_edge_index'][1, :len_ligand_ctx*num_knn]
        id_ligand_ctx_edge = data['ligand_context_bond_index'][0] * len_compose + data['ligand_context_bond_index'][1]
        idx_edge = [torch.nonzero(id_compose_edge == id_) for id_ in id_ligand_ctx_edge]
        idx_edge = torch.tensor([a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge], dtype=torch.long)
        mask = (idx_edge >= 0)
        valid_idx = idx_edge[mask]

        data['compose_knn_edge_type'] = torch.zeros(len(data['compose_knn_edge_index'][0]),num_edge_types, dtype=torch.float32)
        data['compose_knn_edge_feature'] = torch.cat([
            torch.ones([len(data['compose_knn_edge_index'][0]), 1], dtype=torch.float32),
            torch.zeros([len(data['compose_knn_edge_index'][0]), num_edge_types], dtype=torch.float32),
        ], dim=-1) 

        if valid_idx.numel() > 0:  # Check if there are any valid indices
            data['compose_knn_edge_type'][valid_idx] = data['ligand_context_bond_feature'][mask].float()
            data['compose_knn_edge_feature'][valid_idx, 1:] = data['ligand_context_bond_feature'][mask].float()
            data['compose_knn_edge_feature'][valid_idx, :1] =torch.zeros_like(data['ligand_context_bond_feature'][mask][:,:1]).float()
            # data['compose_knn_edge_feature'][idx_edge[idx_edge>=0]][:,1:] = data['ligand_context_bond_feature'][idx_edge>=0]

        return data

class PosPredMaker(object):

    def __init__(self, protein_dim=26, ligand_dim=45, knn_pos_pred=24, lig_noise_scale=0.5):
        super().__init__()
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.knn_pos_pred = knn_pos_pred  
        self.lig_noise_scale = lig_noise_scale

    def __call__(self, data, rot_occupy=0.5):

        
        protein_pos = data['protein_pos']
        len_protein = len(protein_pos)
        protein_feature = data['protein_atom_feature']
        # protein_feature_expended = torch.cat([
        #     protein_feature,torch.zeros([len_protein, 45-26], dtype=torch.long)
        # ], dim=1)
        len_ligand_ctx_next = data['ligand_context_next_feature_full'].size()[0]

        ligand_context_next_featrure_full = data['ligand_context_next_feature_full']
        ligand_context_next_featrure_expanded = torch.cat([
            ligand_context_next_featrure_full,torch.zeros([len_ligand_ctx_next, 27-15], dtype=torch.long)
        ], dim=1)
        
        len_compose = len_ligand_ctx_next + len_protein
        # data['compose_with_next_pos'] = torch.cat([data['ligand_context_next_pos'], protein_pos], dim=0).to(torch.float32)
        data['compose_next_feature'] = torch.cat([ligand_context_next_featrure_expanded, protein_feature], dim=0)
        data['idx_ligand_ctx_next_in_compose'] = torch.arange(len_ligand_ctx_next, dtype=torch.long)
        data['idx_protein_in_compose_with_next'] = torch.arange(len_ligand_ctx_next, len_compose, dtype=torch.long)
        data['idx_protein_in_compose_with_next'] = torch.arange(len_ligand_ctx_next, len_compose, dtype=torch.long)
        

        if data['ligand_pos_mask'].size()[0] > 0: # if the next motif is a fragment  
            # create the pos_learning target
            ligand_context_next_pos_target = copy.deepcopy(data['ligand_context_next_pos'])
            data['compose_with_next_pos_target'] = torch.cat([ligand_context_next_pos_target, protein_pos], dim=0).to(torch.float32)
            # Of note, noised ligand pos should be added here
            if data['mask'] == 'full':
                protein_contact_idx_in_next_compose = data['ligand_context_next_pos'].shape[0] + data['protein_contact_idx'] 
                y_id = protein_contact_idx_in_next_compose.reshape(-1)
                data['a'] = y_id
                contact_pos = data['current_attach_pos'].reshape(-1, 3)
                pkt_pkt_node_dist = torch.norm(contact_pos.unsqueeze(1) - data['protein_pos'].unsqueeze(0), p=2, dim=-1)
                yn_dist, yn =torch.topk(-pkt_pkt_node_dist, k=3, dim=1)
                yn = yn.reshape(-1)

                frag_pos = data['ligand_context_next_pos']
                pkt_frag_dist = torch.norm(contact_pos.unsqueeze(1) - frag_pos.unsqueeze(0), p=2, dim=-1)
                x_id = torch.argmin(pkt_frag_dist, dim=1)          
                data['b'] = x_id
                x_connected = (data['ligand_context_next_bond_index'][0,:] == x_id)
                xn = data['ligand_context_next_bond_index'][:,x_connected][1,:]

                y_pos = contact_pos
                x_pos = data['ligand_context_next_pos'][x_id]
                xn_pos, yn_pos = torch.zeros(3, 3), torch.zeros(3, 3)
                xn_pos[:len(xn)], yn_pos[:len(yn)] = deepcopy(data['ligand_context_next_pos'][xn]), deepcopy(data['protein_pos'][yn])
                xn_idx, yn_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
                xn_idx = xn_idx.squeeze(-1)
                yn_idx = yn_idx.squeeze(-1)
                dihedral_x, dihedral_y = torch.zeros(3), torch.zeros(3)
                dihedral_x[:len(xn)] = 1
                dihedral_y[:len(yn)] = 1
                data['dihedral_mask'] = torch.matmul(dihedral_x.view(3, 1), dihedral_y.view(1, 3)).view(-1).bool()
                data['true_sin'], data['true_cos'] = batch_dihedrals(xn_pos[xn_idx], x_pos.repeat(9, 1), y_pos.repeat(9, 1),
                                                                yn_pos[yn_idx])
                dir = (y_pos - x_pos).reshape(-1)
                ref = x_pos.reshape(-1)
                next_motif_pos = deepcopy(data['ligand_context_next_pos'][data['ligand_pos_mask']])
                data['ligand_context_next_pos'][data['ligand_pos_mask']] = rand_rotate(dir, ref, next_motif_pos)
                
                
                data['compose_with_next_pos'] = torch.cat([data['ligand_context_next_pos'], protein_pos], dim=0).to(torch.float32)
                
                data['y_pos'] = y_pos - x_pos
                data['xn_pos'], data['yn_pos'] = torch.zeros(3, 3), torch.zeros(3, 3)
                data['xn_pos'][:len(xn)], data['yn_pos'][:len(yn)] = data['ligand_context_next_pos'][xn] - x_pos, data['protein_pos'][yn] - x_pos
                data['xn_pos'] = data['xn_pos'].unsqueeze(0)
                data['yn_pos'] = data['yn_pos'].unsqueeze(0) 
                
            if data['mask'] == 'bfs':
                y_id = data['ligand_context_next_bond_pred_index'][0].reshape(-1)
                x_id = data['ligand_context_next_bond_pred_index'][1].reshape(-1)
                data['b'] = x_id
                data['a'] = y_id
                y_connected = (data['ligand_context_next_bond_index'][0,:] == y_id)
                yn = data['ligand_context_next_bond_index'][:,y_connected][1,:]
                x_connected = (data['ligand_context_next_bond_index'][0,:] == x_id)
                xn = data['ligand_context_next_bond_index'][:,x_connected][1,:]
                y_pos = data['ligand_context_next_pos'][y_id]
                x_pos = data['ligand_context_next_pos'][x_id]
                yn = yn[yn != x_id]
                xn = xn[xn != y_id]
                xn_pos, yn_pos = torch.zeros(3, 3), torch.zeros(3, 3)
                xn_pos[:len(xn)], yn_pos[:len(yn)] = deepcopy(data['ligand_context_next_pos'][xn]), deepcopy(data['ligand_context_next_pos'][yn])
                xn_idx, yn_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
                xn_idx = xn_idx.squeeze(-1)
                yn_idx = yn_idx.squeeze(-1)
                dihedral_x, dihedral_y = torch.zeros(3), torch.zeros(3)
                dihedral_x[:len(xn)] = 1
                dihedral_y[:len(yn)] = 1
                data['dihedral_mask'] = torch.matmul(dihedral_x.view(3, 1), dihedral_y.view(1, 3)).view(-1).bool()
                data['true_sin'], data['true_cos'] = batch_dihedrals(xn_pos[xn_idx], x_pos.repeat(9, 1), y_pos.repeat(9, 1),
                                                                yn_pos[yn_idx])

                dir = (y_pos - x_pos).reshape(-1)
                ref = x_pos.reshape(-1)
                next_motif_pos = deepcopy(data['ligand_context_next_pos'][data['ligand_pos_mask']])
                data['ligand_context_next_pos'][data['ligand_pos_mask']] = rand_rotate(dir, ref, next_motif_pos)
                
                
                data['compose_with_next_pos'] = torch.cat([data['ligand_context_next_pos'], protein_pos], dim=0).to(torch.float32)
                
                data['y_pos'] = data['ligand_context_next_pos'][y_id] - x_pos
                data['xn_pos'], data['yn_pos'] = torch.zeros(3, 3), torch.zeros(3, 3)
                data['xn_pos'][:len(xn)], data['yn_pos'][:len(yn)] = data['ligand_context_next_pos'][xn] - x_pos, data['ligand_context_next_pos'][yn] - x_pos
                data['xn_pos'] = data['xn_pos'].unsqueeze(0)
                data['yn_pos'] = data['yn_pos'].unsqueeze(0) 

            # vec = a_pos - b_pos
            # random_rotate_angle = torch.rand(1) * 2 * torch.pi
            # rotate_matrix = rotate_matrix_around_axis(vec, random_rotate_angle)
            # b_next_pos = data['compose_with_next_pos'][data['idx_ligand_ctx_next_in_compose'][data['ligand_pos_mask']]]
            # rotated = rotate_axis_w_centered_point(rotate_matrix, a_pos, b_next_pos)
            # data['compose_with_next_pos'][data['idx_ligand_ctx_next_in_compose'][data['ligand_pos_mask']]] = rotated


            
            data = self.get_knn_graph_pos_pred(data, self.knn_pos_pred, len_ligand_ctx_next, len_compose, num_workers=16)

        else: # if the next motif is an atom
            
            data['rotate_angle'] = torch.tensor([0], dtype=torch.float32)
            data['compose_with_next_pos_target'] = torch.cat([data['ligand_context_next_pos'], protein_pos], dim=0).to(torch.float32)

            data['compose_next_knn_edge_feature'] = torch.empty([0, 6], dtype=torch.float32)
            data['compose_next_knn_edge_index'] = torch.empty([2, 0], dtype=torch.long)
        
        data['ligand_pos_mask_idx'] = torch.where(data['ligand_pos_mask'] == 1)[0]  
        return data
        
    @staticmethod
    def get_knn_graph_pos_pred(data, num_knn, len_ligand_ctx_next, len_compose, num_workers=1, num_edge_types=4):
        data['compose_next_knn_edge_index'] = knn_graph(data['compose_with_next_pos'], num_knn, flow='target_to_source', num_workers=num_workers)

        id_compose_edge = data['compose_next_knn_edge_index'][0, :len_ligand_ctx_next*num_knn] * len_compose + data['compose_next_knn_edge_index'][1, :len_ligand_ctx_next*num_knn]
        id_ligand_ctx_edge = data['ligand_context_next_bond_index'][0] * len_compose + data['ligand_context_next_bond_index'][1]
        idx_edge = [torch.nonzero(id_compose_edge == id_) for id_ in id_ligand_ctx_edge]
        idx_edge = torch.tensor([a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge], dtype=torch.long)
        data['compose_next_knn_edge_type'] = torch.zeros(len(data['compose_next_knn_edge_index'][0]),num_edge_types, dtype=torch.float32)  # for encoder edge embedding
        data['compose_next_knn_edge_type'][idx_edge[idx_edge>=0]] = data['ligand_context_next_bond_feature'][idx_edge>=0].float()
        data['compose_next_knn_edge_feature'] = torch.cat([
            torch.ones([len(data['compose_next_knn_edge_index'][0]), 1], dtype=torch.float32),
            torch.zeros([len(data['compose_next_knn_edge_index'][0]), num_edge_types], dtype=torch.float32),
        ], dim=-1) 
        mask = (idx_edge >= 0)
        valid_idx = idx_edge[mask]
        data['compose_next_knn_edge_feature'][valid_idx, 1:] = data['ligand_context_next_bond_feature'][mask].float()
        data['compose_next_knn_edge_feature'][valid_idx, :1] =torch.zeros_like(data['ligand_context_next_bond_feature'][mask][:,:1]).float()
        # data['compose_next_knn_edge_feature'][idx_edge[idx_edge>=0]][:,1:] = data['ligand_context_next_bond_feature'][idx_edge>=0]
        return data


class MixMasker(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=1, p_bfs=0.8, p_full=0.2, frag_base=None):
        super().__init__()

        if frag_base is None:
            assert NotImplementedError('Please provide the fragment base! masker = MixMasker(frag_base=frag_base))')
        self.masker = [
            FullMask(frag_base=frag_base),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, frag_base=frag_base),
        ]
        self.masker_choosep = [p_full, p_bfs]

    def __call__(self, data):
        f = random.choices(self.masker, k=1, weights=self.masker_choosep)[0]
        return f(data)