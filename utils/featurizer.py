import numpy as np
import rdkit
from rdkit import Chem
from copy import deepcopy
import torch 
from plyfile import PlyData

def read_ply(path, read_face=None):
    with open(path, 'rb') as f:
        data = PlyData.read(f)

    features = ([torch.tensor(data['vertex'][axis.name]) for axis in data['vertex'].properties if axis.name not in ['nx', 'ny', 'nz'] ])
    pos = torch.stack(features[:3], dim=-1)
    features = torch.stack(features[3:], dim=-1)
    if read_face is not None:
        if 'face' in data:
            faces = data['face']['vertex_indices']
            faces = [torch.tensor(fa, dtype=torch.long) for fa in faces]
            face = torch.stack(faces, dim=-1)
            data = {'feature':features,\
                'pos':pos,
                'face':face}
    else:
        data = {'feature':features,\
            'pos':pos}
    return data

atomTypes = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'] #12
# atomTypes = ['C', 'N', 'F', 'P', 'S', 'Cl', 'Br']

formalCharge = [-1, -2, 1, 2, 0] 
hybridization = [
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
]
num_single_bonds = [0,1,2,3,4,5,6]
num_double_bonds = [0,1,2,3,4]
num_triple_bonds = [0,1,2]
num_aromatic_bonds = [0,1,2,3,4]
bondTypes = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

def adjacency_to_undirected_edge_index(adj):
    adj = np.triu(np.array(adj, dtype = int)) #keeping just upper triangular entries from sym matrix
    array_adj = np.array(np.nonzero(adj), dtype = int) #indices of non-zero values in adj matrix
    edge_index = np.zeros((2, 2*array_adj.shape[1]), dtype = int) #placeholder for undirected edge list
    edge_index[:, ::2] = array_adj
    edge_index[:, 1::2] = np.flipud(array_adj)
    return edge_index

def one_hot_embedding(value, options):
    embedding = [0]*(len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding


def getNodeFeatures(list_rdkit_atoms):
    # NOTE: Deprecated, since it faces the data leakage problem
    '''
    Input: list of rdkit atoms
    Output: node features 
    node_feat_dim = 12 + 1 + 5 + 1 + 1 + 1 + 8 + 6 + 4 + 6 = 45
    atom_types feature (13), formal charge (6), aromatic (1), atomic mass (1), Bond related features (24)
    '''
    F_v = (len(atomTypes)+1) # 12 +1 
    F_v += (len(formalCharge)+1) # 5 + 1
    F_v += (1 + 1) 
    
    F_v += 8
    F_v += 6
    F_v += 4
    F_v += 6
    
    node_features = np.zeros((len(list_rdkit_atoms), F_v))
    for node_index, node in enumerate(list_rdkit_atoms):
        features = one_hot_embedding(node.GetSymbol(), atomTypes) # atom symbol, dim=12 + 1 
        features += one_hot_embedding(node.GetFormalCharge(), formalCharge) # formal charge, dim=5+1 
        features += [int(node.GetIsAromatic())] # whether atom is part of aromatic system, dim = 1
        features += [node.GetMass()  * 0.01] # atomic mass / 100, dim=1
        
        atom_bonds = np.array([b.GetBondTypeAsDouble() for b in node.GetBonds()])
        N_single = int(sum(atom_bonds == 1.0) + node.GetNumImplicitHs() + node.GetNumExplicitHs())
        N_double = int(sum(atom_bonds == 2.0))
        N_triple = int(sum(atom_bonds == 3.0))
        N_aromatic = int(sum(atom_bonds == 1.5))
        
        features += one_hot_embedding(N_single, num_single_bonds)
        features += one_hot_embedding(N_double, num_double_bonds)
        features += one_hot_embedding(N_triple, num_triple_bonds)
        features += one_hot_embedding(N_aromatic, num_aromatic_bonds)
        
        node_features[node_index,:] = features
        
    return np.array(node_features, dtype = np.float32)



def getEdgeFeatures(list_rdkit_bonds):
    '''
    Input: list of rdkit bonds
    Return: undirected edge features

    bondTypes = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'] + 1
    edge_feautre_dim = 5
    '''
    F_e = (len(bondTypes)+1) #+ 1 + (4+1)
    
    edge_features = np.zeros((len(list_rdkit_bonds)*2, F_e))
    for edge_index, edge in enumerate(list_rdkit_bonds):
        features = one_hot_embedding(str(edge.GetBondType()), bondTypes) # dim=4+1

        # Encode both directed edges to get undirected edge
        edge_features[2*edge_index: 2*edge_index+2, :] = features
        
    return np.array(edge_features, dtype = np.float32)


# This is from the SQUID method, which is used for the partial structure matching 
def featurize_mol(mol):
    '''
    featurize mol 
    '''
    if type(mol) == Chem.rdchem.Mol:
        mol_ = deepcopy(mol)
    elif type(mol) == str:
        if mol[-3:] == 'mol' or mol[-3:] == 'sdf':
            mol_ = Chem.MolFromMolFile(mol)
        elif mol[-4:] == 'mol2':
            mol_ = Chem.MolFromMol2File(mol)
        else:
            mol_ = Chem.MolFromSmiles(mol)

    Chem.SanitizeMol(mol_)
    adj = Chem.GetAdjacencyMatrix(mol_)
    edge_index = adjacency_to_undirected_edge_index(adj)
    bonds = []
    for b in range(int(edge_index.shape[1]/2)):
        bond_index = edge_index[:,::2][:,b]
        bond = mol_.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = torch.tensor(getEdgeFeatures(bonds)) # PRECOMPUTE
    edge_index = torch.tensor(edge_index)

    atoms = Chem.rdchem.Mol.GetAtoms(mol_)
    node_features = torch.tensor(getNodeFeatures(atoms))
    xyz = torch.tensor(mol_.GetConformer().GetPositions(), dtype=torch.float32)
    element = torch.tensor(np.array([i.GetAtomicNum() for i in mol_.GetAtoms()]))

    data = {
        'element': element,
        'pos': xyz,
        'bond_index': edge_index,
        'bond_feature': edge_features,
        'atom_feature': node_features
    }
    return data

def featurize_frag(mol):
    mol_ = deepcopy(mol)
    Chem.SanitizeMol(mol_)
    adj = Chem.GetAdjacencyMatrix(mol_)
    edge_index = adjacency_to_undirected_edge_index(adj)
    bonds = []
    for b in range(int(edge_index.shape[1]/2)):
        bond_index = edge_index[:,::2][:,b]
        bond = mol_.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = torch.tensor(getEdgeFeatures(bonds)) # PRECOMPUTE
    edge_index = torch.tensor(edge_index)
    atoms = Chem.rdchem.Mol.GetAtoms(mol_)
    node_features = torch.tensor(getNodeFeatures(atoms))
    return node_features, edge_index, edge_features



from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
import os.path as osp

from rdkit.Chem.rdchem import BondType
atomic_num_to_type = {5:0, 6:1, 7:2, 8:3, 9:4, 12:5, 13:6, 14:7, 15:8, 16:9, 17:10, 21:11, 23:12, 26:13, 29:14, \
    30:15, 33:16, 34:17, 35:18, 39:19, 42:20, 44:21, 45:22, 51:23, 53:24, 74:25, 79:26}
atomic_element_to_type = {'C':27, 'N':28, 'O':29, 'NA':30, 'MG':31, 'P':32, 'S':33, 'CL':34, 'K':35, \
    'CA':36, 'MN':37, 'CO':38, 'CU':39, 'ZN':40, 'SE':41, 'CD':42, 'I':43, 'CS':44, 'HG':45}
bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC:1.5}

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}

def parse_rdmol(rdmol, implicit_h=False):
    Chem.SanitizeMol(rdmol)
    fdefName = osp.join('utils','BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    num_atoms = rdmol.GetNumAtoms()
    num_bonds = rdmol.GetNumBonds()
    feat_mat = np.zeros([num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1


    ptable = Chem.GetPeriodicTable()

    element, pos = [], []
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    confs = np.array(rdmol.GetConformer().GetPositions())
    pos = np.array(rdmol.GetConformer().GetPositions())
    for i in range(num_atoms):
        x, y, z = map(float, confs[i])
        symb = rdmol.GetAtomWithIdx(i).GetSymbol()
        atomic_number = ptable.GetAtomicNumber(symb.capitalize())
        element.append(atomic_number)
        atomic_weight = ptable.GetAtomicWeight(atomic_number)
        accum_pos += np.array([x, y, z]) * atomic_weight
        accum_mass += atomic_weight

    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)

    element = np.array(element, dtype=np.int32)
    pos = np.array(pos, dtype=np.float32)

    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
    row, col, edge_type = [], [], []
    bond_type_map = {
        BOND_TYPES[BondType.SINGLE]:1,
        BOND_TYPES[BondType.DOUBLE]:2,
        BOND_TYPES[BondType.TRIPLE]:3,
        BOND_TYPES[BondType.AROMATIC]:4,
    }

    for i in range(num_bonds):
        bond = rdmol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        row += [u,v]
        col += [v,u]
        edge_type += 2*[bond_type_map[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.compat.long)
    edge_type = np.array(edge_type, dtype=np.compat.long)

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
    }
    
    if implicit_h: 
        implicit_hydrogens = np.array([a.GetTotalNumHs() for a in rdmol.GetAtoms()], dtype=np.int32)
        data['implicit_hydrogens'] = implicit_hydrogens

    return data

def parse_rdmol_2d(rdmol):
    Chem.SanitizeMol(rdmol)
    fdefName = osp.join('utils','BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    num_atoms = rdmol.GetNumAtoms()
    num_bonds = rdmol.GetNumBonds()
    feat_mat = np.zeros([num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
    ptable = Chem.GetPeriodicTable()
    element = []
    for i in range(num_atoms):
        symb = rdmol.GetAtomWithIdx(i).GetSymbol()
        atomic_number = ptable.GetAtomicNumber(symb.capitalize())
        element.append(atomic_number)

    element = np.array(element, dtype=np.int32)
    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
    row, col, edge_type = [], [], []
    bond_type_map = {
        BOND_TYPES[BondType.SINGLE]:1,
        BOND_TYPES[BondType.DOUBLE]:2,
        BOND_TYPES[BondType.TRIPLE]:3,
        BOND_TYPES[BondType.AROMATIC]:4,
    }

    for i in range(num_bonds):
        bond = rdmol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        row += [u,v]
        col += [v,u]
        edge_type += 2*[bond_type_map[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.compat.long)
    edge_type = np.array(edge_type, dtype=np.compat.long)

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'element': element,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'atom_feature': feat_mat,
    }
    return data

type_reduce = {'C':0, 'N':1, 'O':2, 'F':3, 'P':4, 'S':5, 'Cl':6, 'Br':7}
def parse_rdmol_base(rdmol):
    Chem.SanitizeMol(rdmol)
    num_atoms = rdmol.GetNumAtoms()
    num_bonds = rdmol.GetNumBonds()

    ptable = Chem.GetPeriodicTable()
    element = []
    type_feature = []
    for i in range(num_atoms):
        symb = rdmol.GetAtomWithIdx(i).GetSymbol()
        atomic_number = ptable.GetAtomicNumber(symb.capitalize())
        element.append(atomic_number)
        type_feature.append(type_reduce[symb.capitalize()])
    
    element = np.array(element, dtype=np.int32)
    type_feature = np.array(type_feature, dtype=np.int32)
    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
    row, col, edge_type = [], [], []
    bond_type_map = {
        BOND_TYPES[BondType.SINGLE]:1,
        BOND_TYPES[BondType.DOUBLE]:2,
        BOND_TYPES[BondType.TRIPLE]:3,
        BOND_TYPES[BondType.AROMATIC]:4,
    }

    for i in range(num_bonds):
        bond = rdmol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        row += [u,v]
        col += [v,u]
        edge_type += 2*[bond_type_map[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.compat.long)
    edge_type = np.array(edge_type, dtype=np.compat.long)

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'element': torch.tensor(element, dtype=torch.int64),
        'type_feature': torch.tensor(type_feature, dtype=torch.int64),
        'bond_index': torch.tensor(edge_index, dtype=torch.int64),
        'bond_type': torch.tensor(edge_type, dtype=torch.int64)
    }
    return data

if __name__ == '__main__':
    mol = mols[0]
    mol_dict = featurize_mol(mol)
    mol_dict = parse_rdmol(mol)
