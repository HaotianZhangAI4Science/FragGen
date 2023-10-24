from rdkit import Chem
from rdkit.Chem import AllChem 
from rdkit.Chem.rdMolAlign import CalcRMS
import numpy as np
import copy
from rdkit import Chem, Geometry
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdRGroupDecomposition
import pickle
import os.path as osp
from easydict import EasyDict
import yaml

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))
    
def read_sdf(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file)
    mols_list = [i for i in supp]
    return mols_list

def write_sdf(mol_list,file, voice=False):
    writer = Chem.SDWriter(file)
    mol_cnt = 0
    for i in mol_list:
        try:
            writer.write(i)
            mol_cnt+=1
        except:
            pass
    writer.close()
    if voice: 
        print('Write {} molecules to {}'.format(mol_cnt,file))

def read_pkl(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(list,file):
    with open(file,'wb') as f:
        pickle.dump(list,f)
        print('pkl file saved at {}'.format(file))

def combine_mols(mols):
    ref_mol = mols[0]
    for add_mol in mols[1:]:
        ref_mol = Chem.CombineMols(ref_mol,add_mol)
    return ref_mol


def qsmis(smis):
    return [Chem.MolFromSmiles(i) for i in smis]

def set_mol_position(mol, pos):
    mol = copy.deepcopy(mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol 

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    tmp_mol = Chem.Mol(mol)
    for idx in range(atoms):
        tmp_mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(tmp_mol.GetAtomWithIdx(idx).GetIdx()))
    return tmp_mol

def check_atom_type(mol):
    flag=True
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        # The Protein
        if int(atomic_number) not in [6,7,8,9,15,16,17]:
            flag=False
    return flag
    
def create_conformer(coords):
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    return conformer

def filter_possible_match(mol, next_motif_mol, query):
    matches = mol.GetSubstructMatches(next_motif_mol)
    if len(matches) == 1:
        exact_match = matches[0]
    else:
        exact_match = [match for match in matches if query in match][0]
    return exact_match

def transfer_conformers(frag, mol, match=None):
    """
    Computes coordinates from molecule to fragment (for all matchings)
    """
    if match is None:
        match = mol.GetSubstructMatch(frag)
        if len(match) < 1:
            raise Exception('Could not find matches')

    mol_coords = mol.GetConformer().GetPositions()
    frag_coords = mol_coords[np.array(match)]
    frag_conformer = create_conformer(frag_coords)
    
    return match, frag_conformer

def transfer_coord(frag, mol):
    matches = mol.GetSubstructMatches(frag)
    if len(matches) < 1:
        raise Exception('Could not find fragment or linker matches')

    match2conf = {}
    for match in matches:
        mol_coords = mol.GetConformer().GetPositions()
        frag_coords = mol_coords[np.array(match)]
        frag_conformer = create_conformer(frag_coords)
        match2conf[remove_mark_mol] = frag_conformer
    new_frag = copy.deepcopy(frag)
    new_frag.AddConformer(frag_conformer)
    return new_frag



def remove_dummys_mol(molecule):
    '''
    Input: mol / str containing dummy atom
    Return: Removed mol, anchor_idx
    '''
    if type(molecule) == str:
        dum_mol = Chem.MolFromSmiles(molecule)
    else:
        dum_mol = molecule
    Chem.SanitizeMol(dum_mol)
    exits = get_exits(dum_mol)
    exit = exits[0]
    bonds = exit.GetBonds()
    if len(bonds) > 1:
        raise Exception('Exit atom has more than 1 bond')
    bond = bonds[0]
    exit_idx = exit.GetIdx()
    source_idx = bond.GetBeginAtomIdx()
    target_idx = bond.GetEndAtomIdx()
    anchor_idx = source_idx if target_idx == exit_idx else target_idx
    efragment = Chem.EditableMol(dum_mol)
    efragment.RemoveBond(source_idx, target_idx)
    efragment.RemoveAtom(exit_idx)

    return efragment.GetMol(), anchor_idx


def rmradical(mol):
    for atom in mol.GetAtoms():
        atom.SetNumRadicalElectrons(0)
    return mol

def docked_rmsd(ref_mol, docked_mols):
    rmsd_list  =[]
    for mol in docked_mols:
        clean_mol = rmradical(mol)
        rightref = AllChem.AssignBondOrdersFromTemplate(clean_mol, ref_mol) #(template, mol)
        rmsd = CalcRMS(rightref,clean_mol)
        rmsd_list.append(rmsd)
    return rmsd_list

from rdkit import Chem
from rdkit.Chem import rdMMPA
from rdkit.Chem import AllChem
import sys



def pocket_trunction(pdb_file, threshold=10, sdf_file=None, centroid=None, outname=None, out_dir=None):
    from utils.pdb_parser import PDBProtein
    pdb_parser = PDBProtein(pdb_file)
    if centroid is None:
        centroid = sdf2centroid(sdf_file)
    else:
        centroid = centroid
    residues = pdb_parser.query_residues_radius(centroid,threshold)
    residue_block = pdb_parser.residues_to_pdb_block(residues)
    pdb_file_name = osp.basename(pdb_file)

    if outname is None:
        outname = pdb_file_name+f'_pocket{threshold}.pdb'
    else:
        outname = outname
    if out_dir is not None:
        out_file = osp.join(out_dir, outname)
    else:
        out_file = osp.join(osp.dirname(pdb_file), outname)

    f = open(out_file,'w')
    f.write(residue_block)
    f.close()
    return out_file

def sdf2centroid(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    lig_xyz = supp[0].GetConformer().GetPositions()
    centroid_x = lig_xyz[:,0].mean()
    centroid_y = lig_xyz[:,1].mean()
    centroid_z = lig_xyz[:,2].mean()
    return centroid_x, centroid_y, centroid_z


from rdkit import DataStructs
def count_parameters(model):
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Convert total parameters to MB and GB (assuming float32 for each parameter)
    total_size_MB = total_params * 4 / (1024 ** 2)  # 4 bytes for each float32
    total_size_GB = total_params * 4 / (1024 ** 3)  # 4 bytes for each float32
    print(f"Total number of parameters: {total_params}")
    print(f"Total size of parameters in MB: {total_size_MB}")
    print(f"Total size of parameters in GB: {total_size_GB}")
    return total_params, total_size_MB, total_size_GB


def compute_sim(ref, gen, source='mol'):
    if source =='mol':
        fp_refmol = Chem.RDKFingerprint(ref)
        fp_genmol = Chem.RDKFingerprint(gen)
        sim = DataStructs.TanimotoSimilarity(fp_refmol, fp_genmol)
    elif source == 'fp':
        sim = DataStructs.TanimotoSimilarity(ref, gen)
    else:
        raise NotImplementedError('Error: you must choose the mol or fp to compute the similariy')
    return sim


def compute_fps(mols):
    fps = [Chem.RDKFingerprint(i) for i in mols]
    return fps


def compute_sims(gen_mols, ref_mols):
    gen_fps = compute_fps(gen_mols)
    ref_fps = compute_fps(ref_mols)
    sim_mat = np.zeros([len(gen_fps), len(ref_fps)])
    for gen_id, gen_fp in enumerate(gen_fps):
        for ref_id, ref_fp in enumerate(ref_fps):
            sim_mat[gen_id][ref_id] = DataStructs.TanimotoSimilarity(gen_fp, ref_fp)
    return sim_mat


def rm_shared(mols, shared):
    '''
    remove the shared structures for a mol list
    e.g.: a series of generated scaffold-constrained molecules, to get the generated part
    '''
    return [Chem.DeleteSubstructs(i, shared) for i in mols]


def anchorfinder(mol, frag):
    '''
    Checking the bound bonds and find where is anchor nodes
    '''
    anchor_idx = []
    anchor_bonds = []
    match = mol.GetSubstructMatch(frag)
    for atom_idx in match:
        atom = mol.GetAtomWithIdx(atom_idx)
        bonds = atom.GetBonds()
        tmp_idx = []
        for bond in bonds:
            src = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            tmp_idx.append(src)
            tmp_idx.append(end)
            if (src not in match) & (end in match):
                anchor_bonds.append(bond.GetIdx())
            if (end not in match) & (src in match):
                anchor_bonds.append(bond.GetIdx())
        tmp_idx = set(tmp_idx)
        if not tmp_idx.issubset(set(match)):
            anchor_idx.append(atom_idx)
    return anchor_idx, anchor_bonds

def find_linker_from_indices(mol, anchor_idx):
    '''
    Using the topological search to find the linked substructure
    '''
    path = Chem.GetShortestPath(mol,anchor_idx[0],anchor_idx[1])
    return path # I need to discover how to get the substructure according to the path tuple

def find_linker_from_bonds(mol, bond_indices):
    '''
    Using the bond_indices to fragmentation the mol and get the linker with two dymmy atoms
    '''
    frags = Chem.FragmentOnSomeBonds(mol, bond_indices, numToBreak=2)[0]
    frags = Chem.GetMolFrags(frags, asMols=True)
    for frag in frags:
        dummy_atom_idxs = [atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy_atom_idxs) == 2:
            linker = frag
    return linker, frags

def find_genpart(mol, frag, return_large=True):
    '''
    Delete fragment in mol, return the residue substructs (generated part)
    Optional: 
        return_max: return the largest frag in the fragments
    '''
    ress = Chem.DeleteSubstructs(mol,frag)
    ress = Chem.GetMolFrags(ress, asMols=True)
    if return_large:
        ress_num = [i.GetNumAtoms() for i in ress]
        max_id = np.argmax(ress_num)
        return ress[max_id]
    else:
        return ress
    

from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig
fdefName = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
fdef = AllChem.BuildFeatureFactory(fdefName)
fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 
        'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')
def get_FeatureMapScore(query_mol, ref_mol):
    featLists = []
    for m in [query_mol, ref_mol]:
        rawFeats = fdef.GetFeaturesForMol(m)
        # filter that list down to only include the ones we're intereted in
        featLists.append([f for f in rawFeats if f.GetFamily() in keep])
    fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=fmParams) for x in featLists]
    fms[0].scoreMode=FeatMaps.FeatMapScoreMode.Best
    fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))
    
    return fm_score

def calc_SC_RDKit_score(query_mol, ref_mol):
    fm_score = get_FeatureMapScore(query_mol, ref_mol)

    protrude_dist = rdShapeHelpers.ShapeProtrudeDist(query_mol, ref_mol,
            allowReordering=False)
    SC_RDKit_score = 0.5*fm_score + 0.5*(1 - protrude_dist)
    #SC_RDKit_score = (1 - protrude_dist)
    return SC_RDKit_score