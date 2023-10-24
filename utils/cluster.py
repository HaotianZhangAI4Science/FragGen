from rdkit import Chem
from .frag import get_ring_fragments, query_clique
import random

###########robust way for extracting the fragment from the full mol###############
def get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment):
    bonds_indices = [b.GetIdx() for b in mol.GetBonds() if len(set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(set(ring_fragment))) == 1]
    bonded_atom_indices = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if len(set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(set(ring_fragment))) == 1]
    bonded_atom_indices_sorted = [(b[0], b[1]) if (b[0] in ring_fragment) else (b[1], b[0]) for b in bonded_atom_indices]
    atoms = [b[1] for b in bonded_atom_indices_sorted]
    
    return bonds_indices, bonded_atom_indices_sorted, atoms
    
def get_fragment_smiles(mol, ring_fragment):
    ring_fragment = [int(r) for r in ring_fragment]
    
    bonds_indices, bonded_atom_indices_sorted, atoms_bonded_to_ring = get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment)
    
    pieces = Chem.FragmentOnSomeBonds(mol, bonds_indices, numToBreak=len(bonds_indices), addDummies=False) 

    fragsMolAtomMapping = []
    fragments = Chem.GetMolFrags(pieces[0], asMols = True, sanitizeFrags = True, fragsMolAtomMapping = fragsMolAtomMapping)
    
    frag_mol = [m_ for i,m_ in enumerate(fragments) if (set(fragsMolAtomMapping[i]) == set(ring_fragment))][0]
    
    for a in range(frag_mol.GetNumAtoms()):
        N_rads = frag_mol.GetAtomWithIdx(a).GetNumRadicalElectrons()
        N_Hs = frag_mol.GetAtomWithIdx(a).GetTotalNumHs()
        if N_rads > 0:
            frag_mol.GetAtomWithIdx(a).SetNumExplicitHs(N_rads + N_Hs)
            frag_mol.GetAtomWithIdx(a).SetNumRadicalElectrons(0)
    
    smiles = Chem.MolToSmiles(frag_mol, isomericSmiles = False)
    
    smiles_mol = Chem.MolFromSmiles(smiles)
    if not smiles_mol:
        logger(f'failed to extract fragment smiles: {smiles}, {ring_fragment}')

        return None

    reduced_smiles = Chem.MolToSmiles(smiles_mol, isomericSmiles = False)
    
    return reduced_smiles
##################################################################################

def get_clique_mol_simple(mol, cluster):
    '''
    Of note, the siile_cluster is made to ensure the following mol.GetSubstructMatch, the bond orderings 
        in the rings would be confused by the kekulize 
    '''
    smile_cluster = Chem.MolFragmentToSmiles(mol, cluster, canonical=True, kekuleSmiles=False)
    mol_cluster = Chem.MolFromSmiles(smile_cluster, sanitize=False)
    return mol_cluster

def ring_decompose(mol):
    '''
    decompose the molecule into cliques (frags + single atoms)and external edges
    '''
    frags_ids = get_ring_fragments(mol)
    num_atoms = mol.GetNumAtoms()
    atom_ids = [{i} for i in (set(range(num_atoms)) - set([item for r in frags_ids for item in r]))]
    cliques = frags_ids + atom_ids

    external_edges = []
    external_edge_ids = []
    
    bonds = [(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in mol.GetBonds()]
    # enumerate all bonds for find the external edges
    for bond in bonds:
        cand_clique = []
        cand_clique_id = []
        for clique_id, clique in enumerate(cliques):
            # for any clique, if the bond is in the clique and the clique is not the bond itself
            if len(set(bond) - set(clique)) == 1:
                cand_clique.append(clique)
                cand_clique_id.append(clique_id)

        # append the external edges
        if len(cand_clique) > 0:
            external_edges.append(cand_clique)
            external_edge_ids.append(cand_clique_id)
            
    return cliques, external_edge_ids, external_edges

def filter_terminal_seeds(all_seeds, mol):
    '''
    Filter Condition:
    1. single atom seed & atom only have been bonded once
    2. fragment seed & fragment only has 1 outside bond
    '''
    terminal_seeds = []
    for seed in all_seeds:
        if len(seed) == 1:
            atom = mol.GetAtomWithIdx(seed[0])
            num_bonds = len(atom.GetBonds())
            if num_bonds == 1:
                terminal_seeds.append(seed)
        else: # fragment -> "terminal" fragment, NOT a fragment inside the overall structure
            atoms_bonded_to_fragment = [[a.GetIdx() for a in mol.GetAtomWithIdx(f).GetNeighbors()] for f in seed]
            atoms_bonded_to_fragment = set([a for sublist in atoms_bonded_to_fragment for a in sublist])
            if len(atoms_bonded_to_fragment - set(seed)) == 1: # fragment only has 1 outside bond
                terminal_seeds.append(seed)
    return tuple(terminal_seeds)

class ClusterNode(object):

    def __init__(self, mol, clique_mol, clique):
        self.smiles = Chem.MolToSmiles(clique_mol)
        self.mol = clique_mol
        self.clique_composition = [x for x in clique]

        self.neighbors = []
        self.rotatable = False

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
        
class FragCluster(object):

    def __init__(self, mol):
        '''
        root refers to the first node in the mol, which is useful for BFS search 
        This version of randomly select the terminal node as root, for training, the root could be 
            specified externally.
        TODO: Error: index[34] cand_roots is None, since it failed to pass the terminal seeds (No Terminal Seeds)
        NOTE: Error: read_sdf is None variable 'mol_' referenced before assignment (Parse Mol Failure)
        TODO: Error: index 0 is out of bounds for axis 0 with size 0 (Fragment base doesn't cover that type)
        '''
        self.smiles = Chem.MolToSmiles(mol)
        self.mol = mol
        self.num_rotatable_bond = 0

        cliques, edge_index, _ = ring_decompose(self.mol)

        self.nodes = []
        all_seeds = [list(i) for i in cliques]
        all_terminals = filter_terminal_seeds(all_seeds, mol)

        cand_roots = []
        for i, clique in enumerate(cliques):
            clique_mol = get_clique_mol_simple(self.mol, clique)
            node = ClusterNode(self.mol, clique_mol, clique)
            self.nodes.append(node)
            for terminal in all_terminals:
                if set(clique) == set(terminal):
                    cand_roots.append(i)
                    break
        
        for x, y in edge_index:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])
                
        root = random.choice(cand_roots)
        if root > 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]
        
        for i, node in enumerate(self.nodes):
            node.nid = i + 1 
        
    def size(self):
        return len(self.nodes)


def get_bfs_perm(clustered_mol, data_base, root=None):
    '''
    Get the BFS permutation of the clustered molecule
    The root could be provided externally, otherwise, the first terminal node will be selected as root
    '''    
    for i, node in enumerate(clustered_mol.nodes):
        query_idx = query_clique(clustered_mol.mol, node.clique_composition, **data_base)
        node.nid = i
        node.wid = query_idx
        # print(query_idx)
    if root is not None:
        bfs_queue = [root]
    else:  
        bfs_queue = [0]
    bfs_perm = []
    bfs_focal = []
    visited = {bfs_queue[0]}
    while len(bfs_queue) > 0:
        current = bfs_queue.pop(0)
        bfs_perm.append(current)
        next_candid = []
        for motif in clustered_mol.nodes[current].neighbors:
            if motif.nid in visited: continue
            next_candid.append(motif.nid)
            visited.add(motif.nid)
            bfs_focal.append(current)

        random.shuffle(next_candid)
        bfs_queue += next_candid

    return bfs_perm, bfs_focal

import torch
import numpy as np
def terminal_select(distances, all_terminals, mode='min'):
    if mode == 'min':
        min_distance = 100
        choose_terminal = 0
        min_idx = 0

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

        min_distance = 100
        min_idx = 0
        for atom_idx_in_mol in choose_terminal:
            if distances[atom_idx_in_mol] < min_distance:
                min_distance = distances[atom_idx_in_mol]
                min_idx = atom_idx_in_mol
    return choose_terminal, min_idx

def terminal_reset(cluster_mol, ligand_pos, protein_pos):
    # the min_idx is the index of the minimum atom index in mol
    mol = cluster_mol.mol
    cliques, edge_index, _ = ring_decompose(mol)
    all_seeds = [list(i) for i in cliques]
    all_terminals = filter_terminal_seeds(all_seeds, mol)
    pkt_lig_dist = torch.norm(protein_pos.unsqueeze(1) - ligand_pos.unsqueeze(0), p=2, dim=-1)
    values, index = torch.min(pkt_lig_dist, dim=0)
    choose_terminal, min_idx = terminal_select(values, all_terminals, mode='min')
    for i, node in enumerate(cluster_mol.nodes):
        if min_idx in node.clique_composition:
            cluster_mol.nodes[0], cluster_mol.nodes[i] = cluster_mol.nodes[i], cluster_mol.nodes[0]
            cluster_mol.nodes[0].min_idx = min_idx
            break
    contact_protein_id = index[min_idx]
    
    return cluster_mol, contact_protein_id

if __name__ == '__main__':
    mol = mols[0]
    frag_cluster = FragCluster(mol)
    atom_frag_database = read_pkl('./mols/crossdock/CrossDock_AtomFragment_database.pkl')
    data_base = {
        'data_base_features': np.concatenate(atom_frag_database['atom_features'], axis = 0).reshape((len(atom_frag_database), -1)),
        'data_base_smiles': np.string_(atom_frag_database.smiles)
    }
    bfs_perm, bfs_focal = get_bfs_perm(frag_cluster, data_base)