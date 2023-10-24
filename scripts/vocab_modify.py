from utils.chem import write_pkl, read_pkl
from rdkit import Chem
import pandas as pd
from utils.featurizer import getNodeFeatures

atom_frag_database = read_pkl('./mols/crossdock/CrossDock_AtomFragment_database.pkl')
atomTypes = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
# atomic_numbers = [6, 7, 8, 9, 15, 16, 17, 35]
atom_mols = [Chem.MolFromSmiles(atom) for atom in atomTypes]
atom_atoms = [Chem.rdchem.Mol.GetAtoms(mol) for mol in atom_mols]
atom_features = [getNodeFeatures(atom)[0] for atom in atom_atoms]

fragment_feature = atom_frag_database['atom_features'][25:].tolist()
fragment_smiles = atom_frag_database['smiles'][25:].tolist()

smiles = atomTypes + fragment_smiles
features = atom_features + fragment_feature
is_fragment = [0] * len(atomTypes) + [1] * len(fragment_smiles)
data_base = pd.DataFrame({
    'smiles': smiles,
    'atom_features': features,
    'is_fragment': is_fragment
})

write_pkl(data_base,'./data/fragment_base.pkl')