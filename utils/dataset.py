import torch
import numpy as np
from utils.cluster import FragCluster, ring_decompose, filter_terminal_seeds
from utils.pdb_parser import PDBProtein
from utils.chem import read_sdf
from utils.featurizer import featurize_mol, parse_rdmol
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os.path as osp
import pickle
import lmdb
from rdkit import Chem
def merge_protein_ligand_dicts(protein_dict=None, ligand_dict=None):
    instance = {}

    if protein_dict is not None:
        for key, item in protein_dict.items():
            instance['protein_' + key] = item

    if ligand_dict is not None:
        for key, item in ligand_dict.items():
            if key == 'moltree':
                instance['moltree'] = item
            else:
                instance['ligand_' + key] = item
    return instance

def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


class ProteinLigand(Dataset):
    '''
    Pair file list version, convenient way for debuging or testing
    For example:
    data_base = '/home/haotian/Molecule_Generation/MG/FLAG-main/data/crossdocked_pocket10'
    index = read_pkl(osp.join(data_base, 'index.pkl'))
    file_list = []
    for idx in range(len(index)):
        try:
            file_pair = [osp.join(data_base, index[idx][0]), osp.join(data_base, index[idx][1])]
            file_list.append(file_pair)
        except Exception as e:
            ...
    dataset = ProteinLigand(file_list, transform=transform)
    '''
    def __init__(self, pair_list, transform=None, data_base=None, mode='min'):
        super().__init__()

        self.pair_list = pair_list


        self.transform = transform
        self.mode = mode
        self.data_base = data_base

    def __len__(self):
        return len(self.pair_list)
    
    def __getitem__(self, index):
        # print(self.pair_list[index])
        pair = self.pair_list[index]
        if self.data_base is not None:
            protein_file = osp.join(self.data_base, pair[0])
            ligan_file = osp.join(self.data_base, pair[1])
        else:
            protein_file = pair[0]
            ligan_file = pair[1]

        pdb_dict = PDBProtein(protein_file).to_dict_atom()
        mol = read_sdf(ligan_file)[0]
        Chem.SanitizeMol(mol)
        # mol_dict = featurize_mol(mol)
        mol_dict = parse_rdmol(mol)
        cluster_mol = FragCluster(mol)
        data = merge_protein_ligand_dicts(protein_dict=pdb_dict, ligand_dict=mol_dict)
        data = torchify_dict(data)
        cluster_mol, contact_protein_id = terminal_reset(cluster_mol,data['ligand_pos'],data['protein_pos'], dist_mode=self.mode)
        data['protein_contact_idx'] = contact_protein_id
        data['cluster_mol'] = cluster_mol
        data = ComplexData(**data)
        if self.transform is not None:
            data = self.transform(data)
        return data

class ProteinLigandLMDB(Dataset):
    '''
    Read preprocessed data from lmdb 
    This equals to the ProteinLigand class, but it is faster for pre-storing the protein and ligand data
    If you want to train on your own dataset, please find the lmdb_create.py, it main appears in the main directory or in the ./script
    '''
    def __init__(self, lmdb_file, name2id_file, transform=None) -> None:
        super().__init__()
        self.lmdb_file = lmdb_file
        self.name2id_file = name2id_file
        self.transform = transform

        self.db = None
        self.keys = None

        if not (osp.exists(self.lmdb_file) and osp.exists(self.name2id_file)):
            raise FileNotFoundError('LMDB file or name2id file not found')
        
        self.name2id = torch.load(self.name2id_file)
    
    def _connect_db(self):
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.lmdb_file,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
    
    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)
    
    def __getitem__(self, index):
        if self.db is None:
            self._connect_db()
        key = self.keys[index]
        data = pickle.loads(self.db.begin().get(key))
        data = ComplexData(**data)
        data.id = index
        if self.transform is not None:
            data = self.transform(data)
        return data
    

class ComplexData(Data):
    '''
    This is used for batching the graph data, you should be very careful for this
    Once it created, you can employ it as ComplexData(**data) to create a ComplexData instance
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __inc__(self, key, value, *args, **kwargs):
        
        # print(key)
        if key == 'compose_feature':
            return 0
        
        elif key == 'context_idx':
            return 0
        elif key == 'context_next_idx':
            return 0

        elif key == 'compose_pos':
            return 0
        elif key == 'idx_ligand_ctx_in_compose':
            return self['compose_pos'].size(0)
        elif key == 'idx_protein_in_compose':
            return self['compose_pos'].size(0)
        # interaction
        elif key == 'compose_knn_edge_index':
            return self['compose_pos'].size(0)
        elif key == 'compose_knn_edge_feature':
            return 0
        # first attach NOTE
        elif key == 'idx_protein_attch_mask':
            return self['compose_pos'].size(0)
        # next possible growing point
        elif key == 'focal_id_in_context':
            return self['compose_pos'].size(0)
        
        elif key == 'focal_id_ligand':
            return self['ligand_context_pos'].size(0)
        # next atom/fragment
        if key == 'next_site_attach_pos':
            return 0
        elif key == 'edge_new_site_knn_index':
            return self['compose_pos'].size(0)

        # next attach point
        elif key == 'node_feat_frags':
            return 0
        elif key == 'edge_index_frags':
            return self['node_feat_frags'].size(0)
        elif key == 'edge_features_frags':
            return 0
        elif key == 'next_site_attach':
            return self['node_feat_frags'].size(0)
        # current_wid
        # next_motif_wid
        
        # bond prediction
        elif key == 'ligand_context_next_bond_pred_index':
            return self['ligand_context_next_feature_full'].size(0)
        elif key == 'ligand_context_next_breaked_bond_index':
            return self['ligand_context_next_feature_full'].size(0)
        
        elif key == 'ligand_context_next_feature_full':
            return 0
        elif key == 'ligand_context_next_feature_breaked_full':
            return 0
        # elif key == 'ligand_context_next_bond_index':
        #     return self['ligand_context_next_feature_full'].size(0)
        elif key == 'ligand_context_next_bond_feature':
            return 0


        
        # position prediction
        elif key == 'compose_next_feature':
            return 0
        elif key == 'compose_with_next_pos':
            return 0
        elif key == 'idx_ligand_ctx_next_in_compose':
            return self['compose_with_next_pos'].size(0)
        elif key == 'idx_protein_in_compose_with_next':
            return self['compose_with_next_pos'].size(0)
        elif key == 'compose_next_knn_edge_feature':
            return 0 
        elif key == 'compose_next_knn_edge_index':
            return self['compose_with_next_pos'].size(0)
        elif key == 'ligand_pos_mask_idx':
            return self['compose_with_next_pos'].size(0)
        elif key == 'a':
            return self['compose_with_next_pos'].size(0)
        elif key == 'b':
            return self['compose_with_next_pos'].size(0)
        elif key == 'a_neigh':
            return self['compose_with_next_pos'].size(0)
        elif key == 'b_neigh':
            return self['compose_with_next_pos'].size(0)
        elif key == 'ligand_context_next_bond_feature':
            return 0
        else:
            return super().__inc__(key, value)

def terminal_select(distances, all_terminals, mode='min'):
    '''
    Select the terminal with the minumun distance to the protein
    two mode: min, prob, where prob is use the normalized distances to select the terminal
    return the chosen terminal and the index of the closet atom in the mol (next attachment point)
    '''
    min_distance = 100
    min_idx = 0
    if mode == 'min':
        for terminal in all_terminals:
            for atom_idx_in_mol in terminal:
                if distances[atom_idx_in_mol] < min_distance:
                    min_distance = distances[atom_idx_in_mol]
                    choose_terminal = terminal
                    min_idx = atom_idx_in_mol
    
    if mode == 'prob':
        terminal_distances = [distances[terminal] for terminal in all_terminals]
        terminal_distances_mean = -torch.tensor(np.array([torch.mean(i) for i in terminal_distances]))
        terminal_weight = torch.softmax(terminal_distances_mean, dim=0)
        choose_terminal = all_terminals[torch.multinomial(terminal_weight, 1).item()]
        for atom_idx_in_mol in choose_terminal:
            if distances[atom_idx_in_mol] < min_distance:
                min_distance = distances[atom_idx_in_mol]
                min_idx = atom_idx_in_mol

    return choose_terminal, min_idx

def terminal_reset(cluster_mol, ligand_pos, protein_pos, dist_mode='min'):
    mol = cluster_mol.mol
    cliques, edge_index, _ = ring_decompose(mol)
    all_seeds = [list(i) for i in cliques]
    all_terminals = filter_terminal_seeds(all_seeds, mol)
    pkt_lig_dist = torch.norm(protein_pos.unsqueeze(1) - ligand_pos.unsqueeze(0), p=2, dim=-1)
    values, index = torch.min(pkt_lig_dist, dim=0)
    choose_terminal, min_idx = terminal_select(values, all_terminals, mode=dist_mode)
    for i, node in enumerate(cluster_mol.nodes):
        if min_idx in node.clique_composition:
            cluster_mol.nodes[0], cluster_mol.nodes[i] = cluster_mol.nodes[i], cluster_mol.nodes[0]
            cluster_mol.nodes[0].min_idx = min_idx
            break
    contact_protein_id = index[min_idx]
    
    return cluster_mol, contact_protein_id


