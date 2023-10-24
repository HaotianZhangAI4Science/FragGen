import lmdb
import pickle
from utils.pdb_parser import PDBProtein
import os.path as osp
import sys
# sys.path.append('/home/haotian/Molecule_Generation/MG/Flex-SBDD')
from tqdm import tqdm
from utils.dataset import merge_protein_ligand_dicts, torchify_dict
from utils.featurizer import featurize_mol, parse_rdmol
from utils.cluster import FragCluster, terminal_reset
from utils.frag import query_clique
from utils.chem import read_sdf, read_pkl
import argparse
import torch
import numpy as np
from torch_geometric.transforms import Compose
from utils.transform import FeaturizeProteinAtom, FeaturizeLigandAtom, LigandBFSMask, FullMask, ComplexBuilder, PosPredMaker
from rdkit import Chem

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base', type=str, default='/home/haotian/Molecule_Generation/MG/FLAG-main/data/crossdocked_pocket10',
                        help='The place where I store the pocket and ligand file, you can adjust this script for your own data or follow me')
    parser.add_argument('--index_file', type=str, default='/home/haotian/Molecule_Generation/MG/FLAG-main/data/crossdocked_pocket10/index.pkl',
                        help='index.pkl include the relative path to the pocket and ligand file, like [./xxx_disease/1vjr.pdb, ./xxx_disease/1vjr.sdf]')
    parser.add_argument('--processed_path', type=str, default='./data/crossdock_processed_fraggen.lmdb',
                        help='The place where I store the processed lmdb data, during training time, reading files consumes a lot of time, lmdb storage could mitigate this issue')
    parser.add_argument('--frag_base', type=str, default='./data/fragment_base.pkl',
                        help='The place where I store the fragment database, this is used to filter the training data out of scope')
    parser.add_argument('--name2id_path', type=str, default='./data/name2id.pt',
                        help='name2id.pt used for index the data in the data base')
    args = parser.parse_args()
    
    
    data_base = args.data_base
    index = read_pkl(args.index_file)
    atom_frag_database = read_pkl(args.frag_base)
    frag_base = {
        'data_base_features': np.concatenate(atom_frag_database['atom_features'], axis = 0).reshape((len(atom_frag_database), -1)),
        'data_base_smiles': np.string_(atom_frag_database.smiles)
    }

    processed_path = args.processed_path
    db = lmdb.open(
        processed_path,
        map_size=10*(1024*1024*1024),   # 10GB
        create=True,
        subdir=False,
        readonly=False, # Writable
    )

    failed = []
    num_skipped = 0
    with db.begin(write=True, buffers=True) as txn:
        for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
            if pocket_fn is None: continue
            try:
                pocket_dict = PDBProtein(osp.join(data_base, pocket_fn)).to_dict_atom()
                mol = read_sdf(osp.join(data_base, ligand_fn))[0]
                Chem.SanitizeMol(mol)
                if mol is None:
                    continue
                mol_dict = parse_rdmol(mol)
                cluster_mol = FragCluster(mol)
                data = merge_protein_ligand_dicts(protein_dict=pocket_dict, ligand_dict=mol_dict)
                data = torchify_dict(data)
                cluster_mol, contact_protein_id = terminal_reset(cluster_mol,data['ligand_pos'], data['protein_pos'])
                data['protein_contact_idx'] = contact_protein_id
                data['cluster_mol'] = cluster_mol
                data['protein_filename'] = pocket_fn
                data['ligand_filename'] = ligand_fn
                try:
                    for node_i, node in enumerate(cluster_mol.nodes):
                        query_idx = query_clique(cluster_mol.mol, node.clique_composition, **frag_base)
                        node.nid = node_i
                        node.wid = query_idx
                except Exception as e:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    failed.append((i, ligand_fn))
                    print(e)
                    continue
    
                txn.put(
                    key = str(i).encode(),
                    value = pickle.dumps(data)
                )
            except Exception as e:
                num_skipped += 1
                print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                print(e)
                failed.append((i, ligand_fn))
                continue
    db.close()

    db = lmdb.open(
            processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    print('saved data {}'.format(len(keys)))

    
    atom_frag_database = read_pkl('./data/fragment_base.pkl')
    frag_base = {
        'data_base_features': np.concatenate(atom_frag_database['atom_features'], axis = 0).reshape((len(atom_frag_database), -1)),
        'data_base_smiles': np.string_(atom_frag_database.smiles)
    }
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()

    # choose one of the masker
    masker = LigandBFSMask(frag_base)
    # masker = FullMask(frag_base)

    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        masker,
        ComplexBuilder(),
        PosPredMaker()
    ])

    name2id = {}
    for i in tqdm(range(len(keys))):
        try:
            data = transform(pickle.loads(db.begin().get(keys[i])))
        except Exception as e:
            print(i, e)
            continue
        name = (data['protein_filename'], data['ligand_filename'])
        name2id[name] = i
    torch.save(name2id, args.name2id_path)
    print('saved name2id at {}'.format(args.name2id_path))