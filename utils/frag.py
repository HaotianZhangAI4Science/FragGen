import torch_geometric
import torch
import torch_scatter
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdMolTransforms
import rdkit.Chem.rdmolops
from rdkit.Chem.rdmolops import AssignStereochemistryFrom3D
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

import networkx as nx
import random
from tqdm import tqdm
import pickle
import os
from rdkit import Chem
from rdkit.Chem import EditableMol
from .featurizer import getNodeFeatures, getEdgeFeatures 


def enumerate_possible_attach(fragment1, fragment2, attachment_point1):
    '''
    Input:
        fragment1: the fragment that is already in the molecule
        fragment2: the fragment that is going to be added to the molecule
        attachment_point1: the atom index of the fragment1 that is going to be attached to fragment2
    Output:
        possible bonded combinations of fragment1 and fragment2
    
    This function serves as the chemical constraints for the model
    E.g.: 
    context_mol = Chem.MolFromSmiles('c1nc(NC2CC2)c2nc[nH]c2n1')
    next_mol = Chem.MolFromSmiles('C1CCCC1')
    posssible_attach = enumerate_possible_attach(context_mol, next_mol, 1)
    '''
    if type(fragment1) == str:
        mol1 = Chem.MolFromSmiles(fragment1)
    else:
        mol1 = fragment1
    if type(fragment2) == str:
        mol2 = Chem.MolFromSmiles(fragment2)
    else:
        mol2 = fragment2

    # Identify potential attachment points on fragment2
    # Here, we consider all atoms with at least one 'free' (non-bonded) valence as potential attachment points
    attachment_points2 = []
    for atom in mol2.GetAtoms():
        free_valence = atom.GetImplicitValence()
        if free_valence > 0:
            attachment_points2.append(atom.GetIdx())

    # Generate combinations
    combinations = []
    for ap2 in attachment_points2:
        # Make editable copies of the original molecules
        emol1 = EditableMol(Chem.Mol(mol1))
        emol2 = EditableMol(Chem.Mol(mol2))

        # Add fragment2 to fragment1
        # Create a mapping from old atom indices in fragment2 to new atom indices in the combined molecule
        index_map = {}
        for atom in mol2.GetAtoms():
            index_map[atom.GetIdx()] = emol1.AddAtom(atom)

        # Add bonds from fragment2 to the combined molecule
        for bond in mol2.GetBonds():
            emol1.AddBond(index_map[bond.GetBeginAtomIdx()], index_map[bond.GetEndAtomIdx()], bond.GetBondType())

        # Create the new bond between fragment1 and fragment2
        emol1.AddBond(attachment_point1, index_map[ap2], Chem.BondType.SINGLE)  # You can change the bond type if needed

        # Get the final combined molecule
        combined_mol = emol1.GetMol()
        Chem.SanitizeMol(combined_mol)
        # Convert to SMILES for demonstration purposes (you may want to keep it as an RDKit Mol object for further use)
        # combined_smiles = Chem.MolToSmiles(combined_mol)
        combinations.append(combined_mol)

    return combinations

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
    '''
    This function solves the problem of extracting a fragment from a molecule, otherwise the rdkit failures are quite common
    Input: 
        mol: the molecule object
        ring_fragment: the atom indices of the fragment
    Return:
        the smiles of the fragment
    '''
    
    if mol.GetNumAtoms() == len(ring_fragment):
        return rdkit.Chem.MolToSmiles(mol, isomericSmiles = False)
    
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
        print(f'failed to extract fragment smiles: {smiles}, {ring_fragment}')

        return None

    reduced_smiles = rdkit.Chem.MolToSmiles(smiles_mol, isomericSmiles = False)
    
    return reduced_smiles

def get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment):
    bonds_indices = [b.GetIdx() for b in mol.GetBonds() if len(set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(set(ring_fragment))) == 1]
    bonded_atom_indices = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if len(set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(set(ring_fragment))) == 1]
    bonded_atom_indices_sorted = [(b[0], b[1]) if (b[0] in ring_fragment) else (b[1], b[0]) for b in bonded_atom_indices]
    atoms = [b[1] for b in bonded_atom_indices_sorted]
    
    return bonds_indices, bonded_atom_indices_sorted, atoms

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol

def generate_conformer(smiles, addHs = False):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    mol = rdkit.Chem.AddHs(mol)
    rdkit.Chem.AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
    
    if not addHs:
        mol = rdkit.Chem.RemoveHs(mol)
    return mol

def get_atoms_in_fragment(atom_idx, ring_fragments):
    ring_idx = None
    for r, ring in enumerate(ring_fragments):
        if atom_idx in ring:
            ring_idx = r
            break
    else:
        return [atom_idx]
    
    return list(ring_fragments[ring_idx])

def get_bonded_connections(mol, atoms = [], completed_atoms = []):
    mol_bonds = [sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]) for bond in mol.GetBonds()]
    bonds = []
    for idx in atoms:
        for b in mol_bonds:
            if idx in b:
                if (int((set(b) - set([idx])).pop())) not in atoms:
                    bonds.append([idx, int((set(b) - set([idx])).pop())])

    # remove any completed atoms from list of new bonds
    if len(completed_atoms) > 0:
        bonds = [b for b in bonds if b[1] not in completed_atoms]
    return bonds, [b[1] for b in bonds]

def add_to_queue_BFS(queue, indices, canonical = False, mol = None, subgraph_indices = None):
    if canonical:
        if subgraph_indices is None:
            canon_ranks = list(rdkit.Chem.rdmolfiles.CanonicalRankAtomsInFragment(
                mol, 
                atomsToUse = list(range(0, len(mol.GetAtoms()))),
                bondsToUse = [b.GetIdx() for b in mol.GetBonds() if ((b.GetBeginAtomIdx() in list(range(0, len(mol.GetAtoms())))) & (b.GetEndAtomIdx() in list(range(0, len(mol.GetAtoms())))))],
                breakTies = True, 
                includeChirality = False, 
                includeIsotopes= True))
        else:
            subgraph_indices = [int(s) for s in subgraph_indices]
            canon_ranks = list(rdkit.Chem.rdmolfiles.CanonicalRankAtomsInFragment(
                mol, 
                atomsToUse = subgraph_indices,
                bondsToUse = [b.GetIdx() for b in mol.GetBonds() if ((b.GetBeginAtomIdx() in subgraph_indices) & (b.GetEndAtomIdx() in subgraph_indices))],
                breakTies = True, 
                includeChirality = False, 
                includeIsotopes= True))
        
        ranks = [canon_ranks[i] for i in indices]        
        ranks_index = sorted(range(len(ranks)), key=lambda k: ranks[k])
        indices = [indices[r] for r in ranks_index]

    else:
        random.shuffle(indices)
        
    new_queue = queue + indices
    return new_queue

def get_dihedral_indices(mol, source, focal):
    _, source_bonds = get_bonded_connections(mol, atoms = [source], completed_atoms = [focal])
    _, focal_bonds = get_bonded_connections(mol, atoms = [focal], completed_atoms = [source])
    if (len(source_bonds) == 0) or (len(focal_bonds) == 0):
        return [-1, source, focal, -1]
    return [source_bonds[0], source, focal, focal_bonds[0]]

def update_completed_atoms(completed_atoms, atoms = []):
    completed_atoms = list(set(completed_atoms + atoms))
    return completed_atoms

def get_source_atom(mol, focal, completed_atoms):
    mol_bonds = [sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]) for bond in mol.GetBonds()]
    source = -1
    source_of_source = -1
    for c in completed_atoms:
        if sorted([c, focal]) in mol_bonds:
            source = c
            for s in completed_atoms:
                if sorted([s, source]) in mol_bonds:
                    source_of_source = s
                    break
            break
    return source, source_of_source


atomic_map = {6:0, 7:1, 8:2, 9:3, 15:4, 16:5, 17:6, 35:7}
def query_clique(mol, query, data_base_features, data_base_smiles):
    '''
    Find the index of the query in the data base
        atom is queried by the features, since it is difficult to save the atom object solely
        frag is queried by the cananical smiles
    '''
    if len(query) == 1:
        atom_fragment_ID_index = atomic_map[mol.GetAtomWithIdx(query[0]).GetAtomicNum()]
    else:
        frag_ID_smiles = get_fragment_smiles(mol, query)
        atom_fragment_ID_index = np.where(data_base_smiles == np.string_(frag_ID_smiles))[0][0]
    return atom_fragment_ID_index

# def query_clique(mol, query, data_base_features, data_base_smiles):
#     '''
#     Find the index of the query in the data base
#         atom is queried by the features, since it is difficult to save the atom object solely
#         frag is queried by the cananical smiles
#     '''
#     if len(query) == 1:
#         atom_features = getNodeFeatures([mol.GetAtomWithIdx(query[0])])
#         atom_fragment_ID_index = np.where(np.all(data_base_features == atom_features, axis = 1))[0][0]
#     else:
#         frag_ID_smiles = get_fragment_smiles(mol, query)
#         atom_fragment_ID_index = np.where(data_base_smiles == np.string_(frag_ID_smiles))[0][0]
#     return atom_fragment_ID_index

def find_bonds_in_mol(mol ,clique1, clique2):
    '''
    return the bonds between two cliques, the id of the bonds are the index of the atom in the mol
    '''
    if type(clique1) == Chem.rdchem.Mol:
        clique1 = mol.GetSubstructMatches(clique1)[0]
    if type(clique2) == Chem.rdchem.Mol:
        clique2 = mol.GetSubstructMatches(clique2)[0]
    bonds = [(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in mol.GetBonds()]
    clique_bonds = [bond for bond in bonds if (len(set(bond) - set(clique1)) == 1 ) & (len(set(bond) - set(clique2)) == 1)]
    return clique_bonds
    
def find_query_attach_point(mol, existing_frag, grown_frag, new_frag, existing_query=None):
    '''
    Input:
        existing_frag is the index of the existing fragment in the mol
        grown_frag is the index of the grown fragment in the mol
        new_frag is the mol object which we want to know the attachment point
        (Optional): existing_query is the mol object which we want to know the attachment point
    Return:
        the attachment point index of the new_fragment
        (Optional): the attachment point index of the existing_query
    '''
    if type(existing_frag) == Chem.rdchem.Mol:
        existing_frag = mol.GetSubstructMatch(existing_frag)
    if type(grown_frag) == Chem.rdchem.Mol:
        grown_frag = mol.GetSubstructMatch(grown_frag)
        
    bond = find_bonds_in_mol(mol,existing_frag, grown_frag)[0] 
    ori_next_bonded_atom_index = list(set(bond)- set(existing_frag))[0]
    new_frag_mol_match = mol.GetSubstructMatch(new_frag)
    new_frag_bonded_index = list(new_frag_mol_match).index(ori_next_bonded_atom_index)

    if existing_query is not None:
        ori_existing_frag_bonded_index = list(set(bond)- set(grown_frag))[0]
        existing_frag_mol_match = mol.GetSubstructMatch(existing_query)
        existing_frag_bonded_index = list(existing_frag_mol_match).index(ori_existing_frag_bonded_index)
        return new_frag_bonded_index, existing_frag_bonded_index
    
    return new_frag_bonded_index