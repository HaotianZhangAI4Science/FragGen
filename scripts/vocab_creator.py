import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdMolTransforms
import rdkit.Chem.rdmolops
from tqdm import tqdm
import pickle
from rdkit import Chem

atomTypes = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
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

def one_hot_embedding(value, options):
    embedding = [0]*(len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding
    
def adjacency_to_undirected_edge_index(adj):
    adj = np.triu(np.array(adj, dtype = int)) #keeping just upper triangular entries from sym matrix
    array_adj = np.array(np.nonzero(adj), dtype = int) #indices of non-zero values in adj matrix
    edge_index = np.zeros((2, 2*array_adj.shape[1]), dtype = int) #placeholder for undirected edge list
    edge_index[:, ::2] = array_adj
    edge_index[:, 1::2] = np.flipud(array_adj)
    return edge_index
    
def getNodeFeatures(list_rdkit_atoms):
    F_v = (len(atomTypes)+1)
    F_v += (len(formalCharge)+1)
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
    F_e = (len(bondTypes)+1) #+ 1 + (6+1)
    
    edge_features = np.zeros((len(list_rdkit_bonds)*2, F_e))
    for edge_index, edge in enumerate(list_rdkit_bonds):
        features = one_hot_embedding(str(edge.GetBondType()), bondTypes) # dim=4+1

        # Encode both directed edges to get undirected edge
        edge_features[2*edge_index: 2*edge_index+2, :] = features
        
    return np.array(edge_features, dtype = np.float32)

##########################Get Ring Fragments##########################
def get_multiple_bonds_to_ring(mol):
    BondToRing = rdkit.Chem.MolFromSmarts('[r]!@[*]')
    bonds_to_rings = mol.GetSubstructMatches(BondToRing)
    NonSingleBond = rdkit.Chem.MolFromSmarts('[*]!-[*]')
    non_single_bonds = mol.GetSubstructMatches(NonSingleBond)
    
    bonds_to_rings = [tuple(sorted(b)) for b in bonds_to_rings]
    non_single_bonds = [tuple(sorted(b)) for b in non_single_bonds]
    
    return tuple(set(bonds_to_rings).intersection(set(non_single_bonds)))

def get_rigid_ring_linkers(mol):
    RingLinker = rdkit.Chem.MolFromSmarts('[r]!@[r]')
    ring_linkers = mol.GetSubstructMatches(RingLinker)
    
    NonSingleBond = rdkit.Chem.MolFromSmarts('[*]!-[*]')
    non_single_bonds = mol.GetSubstructMatches(NonSingleBond)
    
    ring_linkers = [tuple(sorted(b)) for b in ring_linkers]
    non_single_bonds = [tuple(sorted(b)) for b in non_single_bonds]
    
    return tuple(set(ring_linkers).intersection(set(non_single_bonds)))
    
def get_rings(mol):
    return mol.GetRingInfo().AtomRings()
    
def get_ring_fragments(mol):
    rings = get_rings(mol)
    
    rings = [set(r) for r in rings]
    
    # combining rigid ring structures connected by rigid (non-single) bond (they will be combined in the next step)
    rigid_ring_linkers = get_rigid_ring_linkers(mol)
    new_rings = []
    for ring in rings:
        new_ring = ring
        for bond in rigid_ring_linkers:
            if (bond[0] in ring) or (bond[1] in ring):
                new_ring = new_ring.union(set(bond))
        new_rings.append(new_ring)
    rings = new_rings
    
    # joining ring structures
    N_rings = len(rings)
    done = False
    
    joined_rings = []
    for i in range(0, len(rings)):
        
        joined_ring_i = set(rings[i])            
        done = False
        while not done:
            for j in range(0, len(rings)): #i+1
                ring_j = set(rings[j])
                if (len(joined_ring_i.intersection(ring_j)) > 0) & (joined_ring_i.union(ring_j) != joined_ring_i):
                    joined_ring_i = joined_ring_i.union(ring_j)
                    done = False
                    break
            else:
                done = True
        
        if joined_ring_i not in joined_rings:
            joined_rings.append(joined_ring_i)
    
    rings = joined_rings
    
    # adding in rigid (non-single) bonds to these ring structures
    multiple_bonds_to_rings = get_multiple_bonds_to_ring(mol)
    new_rings = []
    for ring in rings:
        new_ring = ring
        for bond in multiple_bonds_to_rings:
            if (bond[0] in ring) or (bond[1] in ring):
                new_ring = new_ring.union(set(bond))
        new_rings.append(new_ring)
    rings = new_rings
    
    return rings
##########################Get Ring Fragments##########################
def get_fragment_smiles(mol, ring_fragment):
    ring_fragment = [int(r) for r in ring_fragment]
    
    bonds_indices, bonded_atom_indices_sorted, atoms_bonded_to_ring = get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment)
    
    pieces = rdkit.Chem.FragmentOnSomeBonds(mol, bonds_indices, numToBreak=len(bonds_indices), addDummies=False) 

    fragsMolAtomMapping = []
    fragments = rdkit.Chem.GetMolFrags(pieces[0], asMols = True, sanitizeFrags = True, fragsMolAtomMapping = fragsMolAtomMapping)
    
    frag_mol = [m_ for i,m_ in enumerate(fragments) if (set(fragsMolAtomMapping[i]) == set(ring_fragment))][0]
    
    for a in range(frag_mol.GetNumAtoms()):
        N_rads = frag_mol.GetAtomWithIdx(a).GetNumRadicalElectrons()
        N_Hs = frag_mol.GetAtomWithIdx(a).GetTotalNumHs()
        if N_rads > 0:
            frag_mol.GetAtomWithIdx(a).SetNumExplicitHs(N_rads + N_Hs)
            frag_mol.GetAtomWithIdx(a).SetNumRadicalElectrons(0)
    
    smiles = rdkit.Chem.MolToSmiles(frag_mol, isomericSmiles = False)
    
    smiles_mol = rdkit.Chem.MolFromSmiles(smiles)
    if not smiles_mol:
        logger(f'failed to extract fragment smiles: {smiles}, {ring_fragment}')

        return None

    reduced_smiles = rdkit.Chem.MolToSmiles(smiles_mol, isomericSmiles = False)
    
    return reduced_smiles

def get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment):
    bonds_indices = [b.GetIdx() for b in mol.GetBonds() if len(set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(set(ring_fragment))) == 1]
    bonded_atom_indices = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if len(set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(set(ring_fragment))) == 1]
    bonded_atom_indices_sorted = [(b[0], b[1]) if (b[0] in ring_fragment) else (b[1], b[0]) for b in bonded_atom_indices]
    atoms = [b[1] for b in bonded_atom_indices_sorted]
    
    return bonds_indices, bonded_atom_indices_sorted, atoms

def generate_conformer(smiles, addHs = False):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    mol = rdkit.Chem.AddHs(mol)
    rdkit.Chem.AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
    
    if not addHs:
        mol = rdkit.Chem.RemoveHs(mol)
    return mol

import pickle
def read_sdf(sdf_file):
    suppl = rdkit.Chem.SDMolSupplier(sdf_file, removeHs=False)
    mols = [m for m in suppl if m is not None]
    return mols

def read_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        mols = pickle.load(f)
    return mols

def write_pkl(file, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(file, f)

def cananical(mol):
    smiles = rdkit.Chem.MolToSmiles(mol, isomericSmiles = False)
    smiles_mol = rdkit.Chem.MolFromSmiles(smiles)
    if not smiles_mol:
        return None
    reduced_smiles = rdkit.Chem.MolToSmiles(smiles_mol, isomericSmiles = False)
    return reduced_smiles

def translate_node_features(feat):
    n = 0
    atom_type = atomTypes[np.argmax(feat[0:len(atomTypes)+1])]
    n += len(atomTypes) + 1
    formal_charge = formalCharge[np.argmax(feat[n: n + len(formalCharge)+1])]
    n += len(formalCharge) + 1
    #hybrid = hybridization[np.argmax(feat[n: n + len(hybridization) + 1])]
    #n += len(hybridization) + 1
    aromatic = feat[n]
    n += 1
    mass = feat[n]
    n += 1
    n_single = num_single_bonds[np.argmax(feat[n: n + len(num_single_bonds) + 1])]
    n += len(num_single_bonds) + 1
    n_double = num_double_bonds[np.argmax(feat[n: n + len(num_double_bonds) + 1])]
    n += len(num_double_bonds) + 1
    n_triple = num_triple_bonds[np.argmax(feat[n: n + len(num_triple_bonds) + 1])]
    n += len(num_triple_bonds) + 1
    n_aromatic = num_aromatic_bonds[np.argmax(feat[n: n + len(num_aromatic_bonds) + 1])]
    n += len(num_aromatic_bonds) + 1
    
    return atom_type, formal_charge, aromatic, mass * 100, n_single, n_double, n_triple, n_aromatic

if __name__ == '__main__':

    '''
    This script is used to create vocabulary for the dataset
    crossdock_top_100_fragment_database.pkl 
    crossdock_unique_atoms.npy
    CrossDock_AtomFragment_database.pkl
    Final AtomFragment database is the mearged version of the above two files
        [0:25] are atom objects and [25:125] are top100 fragments 
    '''
    all_mols = read_pkl('crossdock_all_mols.pkl')

    fragments_smiles = {}
    fragments_smiles_mols = {}

    failed = 0
    succeeded = 0
    ignore = set([])
    total = len(all_mols)
    for inc in tqdm(range(len(all_mols))):
        mol = all_mols[inc]
        ring_fragments = get_ring_fragments(mol)
        for frag in ring_fragments:
            if len(frag) == mol.GetNumAtoms():
                smiles = cananical(mol)
            else:
                smiles = get_fragment_smiles(mol, frag)
            if not smiles:
                failed += 1
                continue
            else:
                succeeded += 1

            if smiles in ignore:
                continue

        m = rdkit.Chem.MolFromSmiles(smiles)
        N_rads = sum([a.GetNumRadicalElectrons() for a in m.GetAtoms()])
        if N_rads > 0:
            ignore.add(smiles)
            continue
        try:
            m_conf = generate_conformer(smiles)
            m_conf.GetConformer()
        except:
            ignore.add(smiles)
            continue
        if smiles not in fragments_smiles:
            fragments_smiles[smiles] = 1
            fragments_smiles_mols[smiles] = rdkit.Chem.MolToSmiles(mol) 
        else:
            fragments_smiles[smiles] += 1
    top_k_fragments = {k: v for k, v in sorted(fragments_smiles.items(), key=lambda item: item[1], reverse = True)}
    
    # select top 100 fragments
    top_100_fragments = {}
    for k in range(0, min(100, len(top_k_fragments))):
        key = list(top_k_fragments.keys())[k]
        top_100_fragments[key] = top_k_fragments[key]

    top_100_fragments_smiles = list(top_100_fragments.keys())
    top_100_fragment_library_dict = {s: i for i,s in enumerate(top_100_fragments_smiles)}

    # generating optimized fragment geometries
    top_100_fragments_mols = []
    top_100_mols_Hs = []
    for s in top_100_fragments_smiles:
        m_Hs = generate_conformer(s, addHs = True)
        rdkit.Chem.AllChem.MMFFOptimizeMolecule(m_Hs, maxIters = 1000)
        m = rdkit.Chem.RemoveHs(deepcopy(m_Hs))
        top_100_fragments_mols.append(m)
        top_100_mols_Hs.append(m_Hs)

    top_100_fragment_database = pd.DataFrame()
    top_100_fragment_database['smiles'] = top_100_fragments_smiles
    top_100_fragment_database['mols'] = top_100_fragments_mols
    top_100_fragment_database['mols_Hs'] = top_100_mols_Hs
    top_100_fragment_database.to_pickle('crossdock_top_100_fragment_database.pkl')


    ## find unique atoms 
    unique_atoms_rdkit = []

    for i in tqdm(range(len(all_mols))):
        mol = all_mols[i]
        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        node_features = getNodeFeatures(atoms)
        if i == 0:
            unique_atoms = np.zeros((1, node_features.shape[1])) # STOP token
        
        for f, feat in enumerate(node_features):
            N_unique = unique_atoms.shape[0]
            if list(feat) not in unique_atoms.tolist():
                unique_atoms = np.concatenate([unique_atoms, np.expand_dims(feat, axis = 0)], axis = 0) #np.unique(np.concatenate([unique_atoms, np.expand_dims(feat, axis = 0)], axis = 0), axis = 0)
                atom = atoms[f]
                atom.SetChiralTag(rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                unique_atoms_rdkit.append(atom)          

    np.save('crossdock_unique_atoms.npy', unique_atoms)
    # Use 24 atom types 

    AtomFragment_database = pd.DataFrame()
    AtomFragment_database['mol'] = [None]*unique_atoms.shape[0] + top_100_fragments_mols
    AtomFragment_database['atom_features'] = [ar for ar in np.concatenate((unique_atoms, -1 * np.ones((len(top_100_fragments_mols), unique_atoms.shape[1]))))]
    AtomFragment_database['is_fragment'] = [0]*unique_atoms.shape[0] + [1]*len(top_100_fragments_mols)
    AtomFragment_database['smiles'] = ['']*unique_atoms.shape[0] + top_100_fragments_smiles
    AtomFragment_database['equiv_atoms'] = [list(rdkit.Chem.CanonicalRankAtoms(m, breakTies=False)) if m != None else [0] for m in AtomFragment_database.mol]
    AtomFragment_database['mol_Hs'] = [None]*unique_atoms.shape[0] + top_100_mols_Hs

    # atom_objects = [i for i in AtomFragment_database_old['atom_objects'][:25]]
    atom_objects = [None] # stop token
    for atom in unique_atoms_rdkit:
        if atom.GetIsAromatic():
            atom_objects.append(None)
            continue
        n_single = sum([b.GetBondTypeAsDouble() == 1.0 for b in atom.GetBonds()])
        atom_rw = rdkit.Chem.RWMol()
        atom_idx = atom_rw.AddAtom(atom)
        atom_rw = rdkit.Chem.AddHs(atom_rw)
        atom_rw = rdkit.Chem.RWMol(atom_rw)
        for i in range(n_single):
            H = rdkit.Chem.Atom('H')
            H_idx = atom_rw.AddAtom(H)
            atom_rw.AddBond(atom_idx, H_idx)
            atom_rw.GetBondBetweenAtoms(0, i+1).SetBondType(rdkit.Chem.rdchem.BondType.SINGLE)
        atom_rw = rdkit.Chem.RemoveHs(atom_rw)
        atom_rw = rdkit.Chem.AddHs(atom_rw)
        rdkit.Chem.SanitizeMol(atom_rw)
        atom_objects.append(atom_rw)
    AtomFragment_database['atom_objects'] = atom_objects + [None]*len(top_100_fragments_mols)

    bond_counts = [translate_node_features(feat)[4:8] for feat in AtomFragment_database.atom_features]
    N_single = [i[0] if AtomFragment_database.iloc[idx].is_fragment == 0 else sum([a.GetTotalNumHs() for a in AtomFragment_database.iloc[idx].mol.GetAtoms()]) for idx, i in enumerate(bond_counts)]
    N_double = [i[1] if AtomFragment_database.iloc[idx].is_fragment == 0 else 0 for idx, i in enumerate(bond_counts)]
    N_triple = [i[2] if AtomFragment_database.iloc[idx].is_fragment == 0 else 0 for idx, i in enumerate(bond_counts)]
    N_aromatic = [i[3] if AtomFragment_database.iloc[idx].is_fragment == 0 else 0 for idx, i in enumerate(bond_counts)]
    AtomFragment_database['N_single'] = N_single
    AtomFragment_database['N_double'] = N_double
    AtomFragment_database['N_triple'] = N_triple
    AtomFragment_database['N_aromatic'] = N_aromatic

    AtomFragment_database.to_pickle('CrossDock_AtomFragment_database.pkl')