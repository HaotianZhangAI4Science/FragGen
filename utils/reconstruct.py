import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit import Geometry

class MolReconsError(Exception):
    pass

def modify_submol(mol):  # modify mols containing C=N(C)O
    submol = Chem.MolFromSmiles('C=N(C)O', sanitize=False)
    sub_fragments = mol.GetSubstructMatches(submol)
    for fragment in sub_fragments:
        atomic_types = np.array([mol.GetAtomWithIdx(atom).GetAtomicNum() for atom in fragment])
        idx_atom_N = fragment[np.where(atomic_types == 7)[0][0]]
        idx_atom_O = fragment[np.where(atomic_types == 8)[0][0]]
        mol.GetAtomWithIdx(idx_atom_N).SetFormalCharge(1)  # set N to N+
        mol.GetAtomWithIdx(idx_atom_O).SetFormalCharge(-1)  # set O to O-
    return mol

def reconstruct_from_data(data, raise_error=True, sanitize=True):
    '''
    Reconstruct a molecule from data, where data contains the following keys:
    - ligand_pos: (n_atoms, 3)
    - ligand_element: (n_atoms,)
    - ligand_bond_index: (2, n_bonds)
    - ligand_bond_type: (n_bonds,)
    - Optional: ligand_implicit_hydrogens: (n_atoms,)
    '''    
    atomic_pos = data['ligand_pos'].clone().cpu().tolist()
    atomic_types = data['ligand_element'].clone().cpu().tolist()
    # indicators = data.ligand_context_feature_full[:, -len(ATOM_FAMILIES_ID):].clone().cpu().bool().tolist()
    bond_index = data['ligand_bond_index'].clone().cpu().tolist()
    bond_type = data['ligand_bond_type'].clone().cpu().tolist()
    
    if 'ligand_implicit_hydrogens' in data:
        implicit_hydrogens = data['ligand_implicit_hydrogens'].clone().cpu().tolist()
    else:
        implicit_hydrogens = None
    n_atoms = len(atomic_types)

    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    # add atoms and coordinates
    for i, atom in enumerate(atomic_types):
        rd_atom = Chem.Atom(atom)
        
        if implicit_hydrogens is not None:
            rd_atom.SetNumExplicitHs(implicit_hydrogens[i])
        else:
            if atom == 7 and all(b == 4 for b in bond_type if (bond_index[0][i] == 6 or bond_index[1][i] == 6)):
                rd_atom.SetNumExplicitHs(1)

        rd_mol.AddAtom(rd_atom)
        rd_coords = Geometry.Point3D(*atomic_pos[i])
        rd_conf.SetAtomPosition(i, rd_coords)

    rd_mol.AddConformer(rd_conf)

    # add bonds
    for i, type_this in enumerate(bond_type):
        node_i, node_j = bond_index[0][i], bond_index[1][i]
        if node_i < node_j:
            if type_this == 1:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
            elif type_this == 2:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
            elif type_this == 3:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
            elif type_this == 4:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
            else:
                raise Exception('unknown bond order {}'.format(type_this))
    # modify
    try:
        rd_mol = modify_submol(rd_mol)
    except:
        if raise_error:
            raise MolReconsError()
        else:
            print('MolReconsError')

    # check valid
    rd_mol_check = Chem.MolFromSmiles(Chem.MolToSmiles(rd_mol))
    if rd_mol_check is None:
        if raise_error:
            raise MolReconsError()
        else:
            print('MolReconsError')
    
    rd_mol = rd_mol.GetMol()
    if sanitize:
        # Chem.SanitizeMol(rd_mol, Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE^Chem.SANITIZE_SETAROMATICITY)
        Chem.SanitizeMol(rd_mol)
    return rd_mol