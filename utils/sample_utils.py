import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from torch_geometric.nn.pool import knn_graph

from utils.pdb_parser import PDBProtein
from utils.dataset import merge_protein_ligand_dicts, torchify_dict
from torch_geometric.data import Data, Batch
from utils.featurizer import featurize_frag, parse_rdmol, read_ply
from rdkit import Chem
from utils.dataset import ComplexData


def ply_to_pocket_data(ply_file):

    protein_dict = read_ply(ply_file) 
    
    data = merge_protein_ligand_dicts(
        protein_dict = protein_dict,
        ligand_dict = {
            'element': torch.empty([0,], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0,], dtype=torch.long),
            'context_pos': torch.empty([0, 3], dtype=torch.float),
            'context_feature_full': torch.empty([0, 15], dtype=torch.float),
            'context_bond_index': torch.empty([2, 0], dtype=torch.long),
            }
    )
    data = torchify_dict(data)
    data['ply_file'] = ply_file
    data['mask'] = 'placeholder'

    return ComplexData(**data)

def pdb_to_pocket_data(pdb_file):
    '''
    use the sdf_file as the center 
    '''
    pocket_dict = PDBProtein(pdb_file).to_dict_atom()

    data = merge_protein_ligand_dicts(
        protein_dict = pocket_dict,
        ligand_dict = {
            'element': torch.empty([0,], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0,], dtype=torch.long),
            'context_pos': torch.empty([0, 3], dtype=torch.float),
            'context_feature_full': torch.empty([0, 15], dtype=torch.float),
            'context_bond_index': torch.empty([2, 0], dtype=torch.long),
        }
    )
    data['pdb_file'] = pdb_file
    data['mask'] = 'placeholder'
    data = torchify_dict(data)
    
    return ComplexData(**data)

atomic_map = {6:0, 7:1, 8:2, 9:3, 15:4, 16:5, 17:6, 35:7}
atoms = [6, 7, 8, 9, 15, 16, 17, 35]
ptable = Chem.GetPeriodicTable()
def element2feature(atomic_numbers):
    if type(atomic_numbers) == torch.Tensor:
        atomic_numbers = [int(i) for i in atomic_numbers]
    else:
        if atomic_numbers[0] != int:
            atomic_numbers = [int(i) for i in atomic_numbers]
    for i in atomic_numbers:
        if i not in atoms:
            atomic_map[i] = 6
    return torch.tensor([atomic_map[i] for i in atomic_numbers], dtype=torch.int64)

def batch_frags(frags):
    data_list = []
    for frag in frags:
        node_feat_frag, edge_index_frag, edge_feat_frag = featurize_frag(frag)
        data_frag = Data(x=node_feat_frag, edge_index=edge_index_frag, edge_attr=edge_feat_frag)
        data_list.append(data_frag)
    data_batch = Batch.from_data_list(data_list)
    return data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch

def split_and_sample_topk(tensor, batch, k):
    '''
    Split tensor according to batch information
    Return a list of topk indices for each batch
    '''
    topk_list = []
    for batch_id in batch.unique():
        fragment = tensor[batch == batch_id]
        values, indices = fragment.mean(dim=1).topk(k)
        topk_list.append(indices)
    return topk_list

def rotation_matrix_align_vectors(vec1, vec2):
    """ 
    Find the rotation matrix that aligns each vec1 to each vec2 using PyTorch for batched input
        :param vec1: A batch of 3d "source" vectors with shape [sample_size, 3]
        :param vec2: A batch of 3d "destination" vectors with shape [sample_size, 3]
        :return mat: A batch of transform matrices (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = vec1 / torch.norm(vec1, dim=1, keepdim=True), vec2 / torch.norm(vec2, dim=1, keepdim=True)
    v = torch.cross(a, b)
    c = (a * b).sum(dim=1, keepdim=True)
    s = torch.norm(v, dim=1, keepdim=True)
    # Skew-symmetric cross-product matrix of v
    kmat = torch.zeros(vec1.size(0), 3, 3, device=vec1.device, dtype=vec1.dtype)
    kmat[:, 0, 1], kmat[:, 0, 2], kmat[:, 1, 0] = -v[:, 2], v[:, 1], v[:, 2]
    kmat[:, 1, 2], kmat[:, 2, 0], kmat[:, 2, 1] = -v[:, 0], -v[:, 1], v[:, 0]
    
    # Compute the rotation matrix
    I = torch.eye(3, device=vec1.device, dtype=vec1.dtype).unsqueeze(0).repeat(vec1.size(0), 1, 1)
    rotation_matrix = I + kmat + torch.bmm(kmat, kmat) * ((1 - c) / (s ** 2).unsqueeze(2))
    
    return rotation_matrix

def get_pos(mol):
    return torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

def gen_pos(mol):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    AllChem.EmbedMolecule(mol)
    return mol

def normalize_vec(vec):
    return vec / torch.norm(vec, dim=1, keepdim=True)

def set_mol_position(mol, pos):
    mol = copy.deepcopy(mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol 

def bond_mols(mol, frag, attch_point_mol, attch_point_frag, bond_type=Chem.rdchem.BondType.SINGLE):
    """
    Bond mol and frag by assigning bond between attch_point_mol and attch_point_frag
    """
    # Create an editable version of mol1
    combine_mol = Chem.CombineMols(mol, frag)

    # Record the number of atoms in mol1 for later use
    mol_n_atoms = mol.GetNumAtoms()
    combine_mol_edit = Chem.EditableMol(combine_mol)
    combine_mol_edit.AddBond(attch_point_mol, mol_n_atoms + attch_point_frag, order=bond_type)
    bonded_mol = combine_mol_edit.GetMol()

    # Return the new molecule
    return bonded_mol

from torch_geometric.data import Data, Batch
class Compose_data(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_pos_pred':
            return self['compose_pos_next'].size(0)
        elif key == 'idx_ligand_next':
            return self['compose_pos_next'].size(0)
        elif key == 'idx_protein_next':
            return self['compose_pos_next'].size(0)
        elif key == 'focal_idx':
            return self['compose_pos_next'].size(0)
        elif key == 'next_attach_idx':
            return self['compose_pos_next'].size(0)
        elif key == 'ligand_pos_mask_idx':
            return self['compose_pos_next'].size(0)
        elif key == 'next_frag_idx':
            return 0
        else:
            return super().__inc__(key, value)

def split_by_batch(tensor, batch):
    unique_batches = torch.unique(batch)
    pos_list = [tensor[batch == b] for b in unique_batches]
    return pos_list

def double_masking(first_mask, second_mask_in_first_mask):
    # If second_mask_in_first_mask is empty, return the first_mask directly
    first_mask = first_mask.to('cpu')
    second_mask_in_first_mask = second_mask_in_first_mask.to('cpu')
    if second_mask_in_first_mask.shape[0] == 0:
        return first_mask

    first_mask_copy = first_mask.clone()
    
    # Select indices from first_mask where the value is True
    true_indices_in_frag = torch.masked_select(torch.arange(len(first_mask)), first_mask)

    # Select indices from second_mask_in_first_mask where the value is False
    false_indices_in_second_mask = torch.masked_select(torch.arange(len(second_mask_in_first_mask)), ~second_mask_in_first_mask)

    # Using the indices from the second mask to update the first mask
    if len(false_indices_in_second_mask) > 0:
        indices_to_set_false = true_indices_in_frag[false_indices_in_second_mask]
        first_mask_copy[indices_to_set_false] = False

    return first_mask_copy


def combine_atom_frag_list(elements1, elements2, mask1, mask2):
    new_elements = [None] * len(mask1)
    pointer1, pointer2 = 0, 0

    for i in range(len(mask1)):
        if mask1[i]:
            new_elements[i] = elements1[pointer1]
            pointer1 += 1
        elif mask2[i]:
            new_elements[i] = elements2[pointer2]
            pointer2 += 1
    return new_elements

from torch_scatter import scatter_add
def atom_bonded_neighbor_aggregation(bond_index, bond_feature, idx, num_nodes):
    bonded_feature = scatter_add(bond_feature, bond_index[0], dim=0, dim_size=num_nodes)[idx] * 2
    return bonded_feature.float()

def compose_feature_catdim(ligand_atom_feature, protein_atom_feature):
    device = ligand_atom_feature.device

    ligand_feat_dim = ligand_atom_feature.shape[1]
    protein_feat_dim = protein_atom_feature.shape[1]
    len_ligand = ligand_atom_feature.shape[0]
    len_protein = protein_atom_feature.shape[0]
    if ligand_feat_dim > protein_feat_dim:
        protein_expanded = torch.cat([
            protein_atom_feature, torch.zeros([len_protein, ligand_feat_dim - protein_feat_dim], dtype=torch.long).to(device)
        ], dim=1)
        compose_feature = torch.cat([ligand_atom_feature, protein_expanded], dim=0)
    elif ligand_feat_dim < protein_feat_dim:
        ligand_expanded = torch.cat([
            ligand_atom_feature, torch.zeros([len_ligand, protein_feat_dim - ligand_feat_dim], dtype=torch.long).to(device)
        ], dim=1)
        compose_feature = torch.cat([ligand_expanded, protein_atom_feature], dim=0)
    else:
        compose_feature = torch.cat([ligand_atom_feature, protein_atom_feature], dim=0)
    
    idx_ligand_in_complex = torch.arange(len_ligand, dtype=torch.long).to(device)
    idx_protein_in_complex = torch.arange(len_ligand, len_ligand + len_protein, dtype=torch.long).to(device)

    return compose_feature.float(), idx_ligand_in_complex, idx_protein_in_complex

def get_spatial_pl_graph(ligand_bond_index, ligand_bond_feature, ligand_pos, protein_pos, num_knn=24,num_workers=16, num_edge_types=4):
    '''
    merge ligand and protein into interaction graph
    The edge_feature represent the spatial contact (0) and covalent bond (remaining)
    '''
    device = ligand_pos.device

    compose_pos_next = torch.cat([ligand_pos, protein_pos], dim=0)
    compose_knn_edge_index = knn_graph(compose_pos_next, num_knn, flow='target_to_source', num_workers=num_workers).to(device)
    len_ligand_ctx = ligand_pos.shape[0]
    len_protein_ctx = protein_pos.shape[0]
    len_compose = len_ligand_ctx + len_protein_ctx
    id_compose_edge = compose_knn_edge_index[0, :len_ligand_ctx*num_knn] * len_compose + compose_knn_edge_index[1, :len_ligand_ctx*num_knn]
    id_ligand_ctx_edge = ligand_bond_index[0] * len_compose + ligand_bond_index[1]
    idx_edge = [torch.nonzero(id_compose_edge == id_) for id_ in id_ligand_ctx_edge]
    idx_edge = torch.tensor([a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge], dtype=torch.long)
    mask = (idx_edge >= 0)
    valid_idx = idx_edge[mask]
    compose_knn_edge_type = torch.zeros(len(compose_knn_edge_index[0]),num_edge_types, dtype=torch.float32).to(device)
    compose_knn_edge_feature = torch.cat([
        torch.ones([len(compose_knn_edge_index[0]), 1], dtype=torch.float32),
        torch.zeros([len(compose_knn_edge_index[0]), num_edge_types], dtype=torch.float32),
    ], dim=-1).to(device)

    if valid_idx.numel() > 0:  # Check if there are any valid indices
        compose_knn_edge_type[valid_idx] = ligand_bond_feature[mask].float()
        compose_knn_edge_feature[valid_idx, 1:] = ligand_bond_feature[mask].float()
        compose_knn_edge_feature[valid_idx, :1] =torch.zeros_like(ligand_bond_feature[mask][:,:1]).float()
        # compose_knn_edge_feature[idx_edge[idx_edge>=0]][:,1:] = ligand_bond_feature[idx_edge>=0]
    
    return compose_knn_edge_index, compose_knn_edge_feature

def add_next_mol2data(mol, data, current_wid, transform_ligand):
    device = data['protein_atom_feature'].device
    
    compose_data = ComplexData(**{})
    mol_dict = parse_rdmol(mol, implicit_h=True)
    mol_parse = merge_protein_ligand_dicts(protein_dict={}, ligand_dict=mol_dict)
    mol_parse = torchify_dict(mol_parse)
    mol_parse = transform_ligand(ComplexData(**mol_parse)).to(device)

    compose_feature, idx_ligand, idx_protein = compose_feature_catdim(mol_parse['ligand_atom_feature_full'], data['protein_atom_feature'])
    compose_pos = torch.cat([mol_parse['ligand_pos'], data['protein_pos']], dim=0)
    # ligand_bond_index = mol_parse['ligand_bond_index']
    # ligand_bond_feature = F.one_hot(mol_parse['ligand_bond_type'] -1, num_classes=4)
    ligand_bond_index = mol_parse['ligand_bond_index']
    ligand_bond_feature = mol_parse['ligand_bond_feature']
    
    compose_knn_edge_index, compose_knn_edge_feature = get_spatial_pl_graph(ligand_bond_index, ligand_bond_feature, mol_parse['ligand_pos'], data['protein_pos'], \
                                                                            num_knn=36,num_workers=18)
    compose_data['protein_pos'] = data['protein_pos']
    compose_data['protein_atom_feature'] = data['protein_atom_feature']
    
    compose_data['ligand_pos'] = mol_parse['ligand_pos']
    compose_data['ligand_atom_feature'] = mol_parse['ligand_atom_feature']
    compose_data['ligand_element'] = mol_parse['ligand_element']
    compose_data['ligand_bond_index'] = mol_parse['ligand_bond_index']
    compose_data['ligand_bond_feature'] = mol_parse['ligand_bond_feature']
    compose_data['ligand_bond_type'] = mol_parse['ligand_bond_type']
    compose_data['ligand_context_pos'] = mol_parse['ligand_pos']
    compose_data['ligand_implicit_hydrogens'] = mol_parse['ligand_implicit_hydrogens']
    compose_data['ligand_mol'] = mol
    compose_data['compose_feature'] = compose_feature
    compose_data['compose_pos'] = compose_pos
    compose_data['compose_knn_edge_index'] = compose_knn_edge_index
    compose_data['compose_knn_edge_feature'] = compose_knn_edge_feature
    compose_data['idx_ligand_ctx_in_compose'] = idx_ligand
    compose_data['idx_protein_in_compose'] = idx_protein
    compose_data['current_wid'] = current_wid.to(device)
    compose_data['current_motif_pos'] = mol_parse['ligand_pos']
    if data.pkt_mol is not None:
        compose_data.pkt_mol = data.pkt_mol
    
    # compose_data['status'] = data['status']
    return compose_data

from collections import defaultdict
def select_data_with_limited_smi(data_list, smi_tolorance=2):
    smiles_count = defaultdict(int)
    limited_data_list = []

    for data in data_list:
        smiles = data['smiles']
        if smiles_count[smiles] < smi_tolorance:
            limited_data_list.append(data)
            smiles_count[smiles] += 1
            
    return limited_data_list

'''
LTK example
# intermidiate fragment
from utils.featurizer import parse_rdmol
from utils.dataset import ComplexData
from models.embed import embed_compose

frag_file = './data/pan/partial_ltk.sdf'
frag_mol = read_sdf(frag_file)[0]
mol_dict = parse_rdmol(frag_mol)
data = merge_protein_ligand_dicts(protein_dict=pocket_dict, ligand_dict=mol_dict)
data = ComplexData(**torchify_dict(data))
transform = Compose([
    LigandCountNeighbors(),
    FeaturizeLigandBond(),
    FeaturizeLigandAtom()
])
data = transform(data)
'''

