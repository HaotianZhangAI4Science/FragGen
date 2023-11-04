import os
import shutil
import os.path as osp
import numpy as np
from glob import glob
from tqdm import tqdm
from easydict import EasyDict
import gc
import random
import torch
from utils.chem import read_pkl, write_pkl, read_sdf, write_sdf, mol_with_atom_index, load_config, pocket_trunction
from copy import deepcopy
from torch_geometric.transforms import Compose
from utils.transform import FeaturizeProteinAtom, FeaturizeLigandAtom, \
    LigandBFSMask, FullMask, ComplexBuilder, PosPredMaker, LigandCountNeighbors,\
         FeaturizeLigandBond, MixMasker, FeaturizeProteinSurface
from models.IDGaF import FragmentGeneration
from utils.sample_utils import pdb_to_pocket_data, add_next_mol2data, select_data_with_limited_smi, ply_to_pocket_data
from utils.sample import get_init, get_init_only_frag, get_next
from rdkit import Chem
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sample_dihedral.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--surf_file', type=str, default='./data/crosssdock_test/4tos_A_rec_4tos_355_lig_tt_min_0/4tos_A_rec_4tos_355_lig_tt_min_0_pocket_8.0_res_1.5.ply',
                            help='surface file, generate basded on this')
    parser.add_argument('--pdb_file', type=str, default='./data/crosssdock_test/4tos_A_rec_4tos_355_lig_tt_min_0/4tos_A_rec.pdb')
    parser.add_argument('--sdf_file',type=str,default='./data/crosssdock_test/4tos_A_rec_4tos_355_lig_tt_min_0/4tos_A_rec_4tos_355_lig_tt_min_0.sdf',
                            help='original ligand sdf_file, only for providing center')
    parser.add_argument('--center',type=str,default=None,
                            help='provide center explcitly, e.g., "32.33,25.56,45.67", where , is used to split x y z coordinates')
    parser.add_argument('--frag_base', type=str, default='./data/fragment_base.pkl')
    parser.add_argument('--save_dir', type=str, default='./output')
    args = parser.parse_args()
    print('Thanks to use the SurfFrag!')
    print('If you do not assign the sdf_file and center, the model will treat the pdb_file as a already truncated pocket file, and use the center of the pdb_file as the center of the pocket')

    config = load_config(args.config)
    
    saved_dir = osp.join(args.save_dir, osp.basename(args.sdf_file)[:-4])
    if not osp.exists(saved_dir):
        os.makedirs(saved_dir, exist_ok=True)
    SDF_dir = osp.join(saved_dir, 'SDF')
    os.makedirs(SDF_dir, exist_ok=True)

    protein_featurizer = FeaturizeProteinSurface()
    ligand_featurizer = FeaturizeLigandAtom()
    protein_atom_feature_dim = protein_featurizer.feature_dim
    ligand_atom_feature_dim = ligand_featurizer.feature_dim

    # two transform, whole transform is used to build the initial sample, while the 
    # transform_ligand is used in the following sample function
    transform = Compose([
        LigandCountNeighbors(),
        FeaturizeLigandBond(),
        protein_featurizer,
        ligand_featurizer,
        ComplexBuilder(protein_dim=protein_atom_feature_dim, ligand_dim=ligand_atom_feature_dim),
    ])
    transform_ligand = Compose([
        LigandCountNeighbors(),
        FeaturizeLigandBond(),
        ligand_featurizer
    ])

    # load the fragment database
    atom_frag_database = read_pkl(args.frag_base)
    frag_base = {
        'data_base_features': np.concatenate(atom_frag_database['atom_features'], axis = 0).reshape((len(atom_frag_database), -1)),
        'data_base_smiles': np.string_(atom_frag_database.smiles)
    }
    print('The fragment database containing {} fragments has been loaded'.format(frag_base['data_base_smiles'].shape[0]))

    # model loading 
    ckpt = torch.load(config.model.checkpoint , map_location=args.device)
    model = FragmentGeneration(ckpt['config'].model, protein_atom_feature_dim, \
                                ligand_atom_feature_dim, frag_atom_feature_dim=45, num_edge_types=5,\
                                num_classes=frag_base['data_base_smiles'].shape[0], pos_pred_type=ckpt['config'].model.pos_pred_type).to(args.device)
    
    print('Num of parameters is {0:.4}M'.format(np.sum([p.numel() for p in model.parameters()]) /100000 ))
    model.load_state_dict(ckpt['model'])
    model = model.to(args.device)

    # the model.pos_pred_type would determine how the neural geometry prediction is performed
    model.pos_pred_type = ckpt['config'].model.pos_pred_type
    print("The Nueral Geometry version is {}".format(model.pos_pred_type))
    # load the pocket data

    pkt_data = transform(ply_to_pocket_data(args.surf_file)).to(args.device)
    if config.model.pos_pred_type == 'geomopt':
        pkt_file = pocket_trunction(args.pdb_file, sdf_file = args.sdf_file)
        pkt_mol = Chem.MolFromPDBFile(pkt_file)
        pkt_data.pkt_mol = pkt_mol
    else:
        pkt_data.pkt_mol = None
        
    def reduce_threshold_dict(threshold_dict):
        new_threshold= EasyDict({})
        for key in threshold_dict.keys():
            if threshold_dict[key] > 0.05:
                new_threshold[key] = threshold_dict[key]/2
            else:
                new_threshold[key] = threshold_dict[key] - 1
        return new_threshold
    sample_threshold = config.sample.threshold
    next_threshold = config.sample.next_threshold
    # start to generate 
    data_initial_list = get_init(model, pkt_data, frag_base, transform_ligand, sample_threshold,)
    # data_next_list = get_init_only_frag(model, data)


    pool = EasyDict({
        'queue': [],
        'smiles': [],
        'mols': []
    })

    pool.queue = data_initial_list
    random_sample_queue = False

    global_step = 0 
    while len(pool.mols) < config.sample.num_samples:
        global_step += 1
        if global_step > config.sample.max_steps:
            break
        queue_size = len(pool.queue)
        queue_tmp = []
        for data in tqdm(pool.queue, desc=f"{global_step} step"): # iterate each data in the queue
            nexts = []
            try:
                next_data = add_next_mol2data(data['ligand_mol'], pkt_data, data.current_wid, transform_ligand) # add the current ligand to the pocket data
            except Exception as e:
                # print(e)
                # print('next_data construction error')
                continue
            if global_step < config.sample.initial_num_steps:
                data_next_list = get_next(model, next_data, frag_base, transform_ligand, threshold_dict=sample_threshold, force_frontier=3)
            else:
                data_next_list = get_next(model, next_data, frag_base, transform_ligand, threshold_dict=next_threshold, force_frontier=-1) # you can change a low threshold here for broader generation
            # assign the smiles to the data
            for data in data_next_list:
                data['smiles'] = Chem.MolToSmiles(data['ligand_mol'])
            
            # data_next_list is the result of each data in the queue, 
            # here we iterate them to see whether they are finished, if not, we filter them for the next step queue
            for data_next in data_next_list: 
                if data_next.status == 'finished':
                    rdmol = data['ligand_mol']
                    smiles = data['smiles']
                    if rdmol.GetNumAtoms() < 7: # The molecules are too small, it may generate outside the pocket
                        continue
                    if smiles in pool.smiles:
                        continue
                    elif '.' in smiles:
                        continue
                    else:
                        pool.smiles.append(smiles)
                        pool.mols.append(rdmol)
                        write_sdf([rdmol], osp.join(SDF_dir, f'{len(pool.mols)-1}.sdf'))
                        num_mols = len(pool.mols)
                        if num_mols % 50 == 0:
                            print(smiles, f'This is the {num_mols} molecules')
                
                # if it is not finished, we add it to the next step queue
                elif data_next.status == 'running':
                    nexts.append(data_next)
            
            queue_tmp += nexts
            if len(queue_tmp) % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            #print(len(queue_tmp))
            # if len(queue_tmp) > 1000:
            #     break

        gc.collect()
        torch.cuda.empty_cache()
        
        limited_data_list = select_data_with_limited_smi(queue_tmp, smi_tolorance=config.sample.queue_same_smi_tolorance) # filter the data with the same smiles, smi_tolorance is the maximum number of the same smiles in the limited_data_list
        # if len(limited_data_list) < 50:
        #     queue_tmp = pool.queue
        #     if global_step < config.sample.initial_num_steps:
        #         sample_threshold = reduce_threshold_dict(sample_threshold)
        #     else:
        #         next_threshold = reduce_threshold_dict(next_threshold)
        #         # print(next_threshold)
        # else:
        queue_tmp = limited_data_list

        p_next_data_list = torch.tensor([data['p_all'] for data in queue_tmp]) # get the probability of each data in the queue

        n_tmp = len(queue_tmp)

        if n_tmp == 0:
            print('generated_finished_mols: ', len(pool.mols))
            break

        if random_sample_queue: # random sample the queue. If the len(queue_tmp) < config.sample.beam_size, you will get the same results no matter the random_sample_queue is True or False
            perm = torch.randperm(len(p_next_data_list))
            next_idx = perm[:min(config.sample.beam_size, n_tmp)]
        else:
            next_idx = torch.multinomial(p_next_data_list, min(config.sample.beam_size, n_tmp), replacement=False)

        pool.queue = [queue_tmp[idx] for idx in next_idx]
    
    sdf_file = osp.join(saved_dir, f'{osp.basename(args.pdb_file)[:-4]}_gen.sdf')
    write_sdf(pool.mols, sdf_file, voice=True)
    if args.sdf_file is not None:
        shutil.copy(args.sdf_file, saved_dir)
        ori_dir = osp.join(saved_dir, 'ori')
        os.makedirs(ori_dir, exist_ok=True)
        shutil.copy(args.sdf_file, osp.join(ori_dir,'0.sdf'))
    
    shutil.copy(args.pdb_file, saved_dir)
    
