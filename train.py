# cdk2 1.800, -8.937, -28.294
import os
import shutil
import os.path as osp
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from utils.chem import read_pkl, write_pkl, read_sdf, write_sdf, mol_with_atom_index, load_config
from utils.frag import query_clique, find_bonds_in_mol, find_query_attach_point
from utils.cluster import FragCluster, get_bfs_perm
from utils.featurizer import featurize_mol
from copy import deepcopy
from utils.pdb_parser import PDBProtein
from torch_geometric.transforms import Compose

from utils.transform import FeaturizeProteinAtom, FeaturizeLigandAtom, \
    LigandBFSMask, FullMask, ComplexBuilder, PosPredMaker, LigandCountNeighbors, FeaturizeLigandBond, MixMasker
from utils.dataset import merge_protein_ligand_dicts, torchify_dict
from utils.cluster import ring_decompose, get_clique_mol_simple, filter_terminal_seeds,\
     ClusterNode, terminal_reset
from utils.dataset import ProteinLigand, ProteinLigandLMDB
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from utils.train import get_model_loss, get_new_log_dir, get_logger, get_optimizer, get_scheduler, seed_all
from models.IDGaF import FragmentGeneration
from time import time
from torch.utils.data import Subset

# define the config
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./configs/train_idgaf.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--split', type=str, default='./data/split_by_name.pt')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    # construct the config
    config = load_config(args.config)
    config_model = config['model']
    seed_all(2022)
    # Build the logger file, you cannot trust the python automatic logger
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # atom_frag_database = read_pkl('./mols/crossdock/CrossDock_AtomFragment_database.pkl')
    atom_frag_database = read_pkl('./data/fragment_base.pkl')
    frag_base = {
        'data_base_features': np.concatenate(atom_frag_database['atom_features'], axis = 0).reshape((len(atom_frag_database), -1)),
        'data_base_smiles': np.string_(atom_frag_database.smiles)
    }

    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    # masker = FullMask(frag_base)
    # choose one
    # masker = LigandBFSMask(frag_base=frag_base)
    masker = MixMasker(frag_base=frag_base)
    transform = Compose([
        LigandCountNeighbors(),
        FeaturizeLigandBond(),
        protein_featurizer,
        ligand_featurizer,
        masker,
        ComplexBuilder(),
        PosPredMaker()
    ])

    dataset = ProteinLigandLMDB('./data/crossdock_processed_fraggen.lmdb', './data/name2id.pt', transform=transform)

    protein_atom_feature_dim = protein_featurizer.feature_dim
    ligand_atom_feature_dim = ligand_featurizer.feature_dim
    model = FragmentGeneration(config_model, protein_atom_feature_dim, \
                            ligand_atom_feature_dim, frag_atom_feature_dim=45, num_edge_types=5).to(args.device)

    # split the train and val
    split_by_name = torch.load(args.split)
    split = {
        k: [dataset.name2id[n] for n in names if n in dataset.name2id]
        for k, names in split_by_name.items()
    }
    subsets = {k:Subset(dataset, indices=v) for k, v in split.items()}
    train_set, val_set = subsets['train'], subsets['test']

    #follow_batch = ['ligand_context_pos','compose_pos', 'node_feat_frags', 'edge_new_site_knn_index','compose_next_feature','bonded_a_nei_edge_features']
    follow_batch = ['ligand_context_pos','compose_pos', 'node_feat_frags', 'edge_new_site_knn_index','compose_next_feature','a_neigh','b_neigh','ligand_pos_mask','idx_ligand_ctx_next_in_compose']
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, follow_batch=follow_batch)
    # train_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, follow_batch=follow_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, follow_batch=follow_batch)

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    def evaluate(epoch, verbose=1):
        model.eval()
        eval_start = time()
        #eval_losses = {'total':[], 'frontier':[], 'pos':[], 'cls':[], 'edge':[], 'real':[], 'fake':[], 'surf':[] }
        eval_losses = []
        for batch in val_loader:
            batch = batch.to(args.device)  
            loss, loss_protein_frontier, loss_frontier, loss_cav, loss_class, loss_nx_attch, loss_bond, loss_pos = get_model_loss(model, batch)
            eval_losses.append(loss.item())    
            torch.cuda.empty_cache()
        average_loss = sum(eval_losses) / len(eval_losses)
        if verbose:
            logger.info('Evaluate Epoch %d | Avg Loss %.6f LastBatch: Loss(Fron) %.6f | Loss(Cav) %.6f | Loss(Cls) %.6f | Loss(NextAttach) %.6f | Loss(Bond) %.6f | Loss(Pos) %.6f | Loss(Protein_fron) %.6f  ' % (
                    epoch, average_loss, loss_frontier.item(), loss_cav.item(), loss_class.item(), loss_nx_attch.item(), loss_bond.item(), loss_pos.item(), loss_protein_frontier.item()
                    ))
        return average_loss

    def load(checkpoint, epoch=None, load_optimizer=False, load_scheduler=False):
        
        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint,epoch)
        logger.info("Load checkpoint from %s" % checkpoint)

        state = torch.load(checkpoint, map_location=args.device)   
        model.load_state_dict(state["model"])
        #self._model.load_state_dict(state["model"], strict=False)
        #best_loss = state['best_loss']
        #start_epoch = state['cur_epoch'] + 1

        if load_scheduler:
            scheduler.load_state_dict(state["scheduler"])
            
        if load_optimizer:
            optimizer.load_state_dict(state["optimizer"])
            if args.device == 'cuda':
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(args.device)
        return state['best_loss']


    def train(verbose=1, num_epoches = 300):

        train_start = time()
        train_losses = []
        val_losses = []
        start_epoch = 0
        best_loss = 1000
        if config.train.resume_train:
            ckpt_name = config.train.ckpt_name
            start_epoch = int(config.train.start_epoch)
            best_loss = load(config.train.checkpoint_path,ckpt_name)
        logger.info('start training...')
        
        for epoch in range(num_epoches):
            model.train()
            epoch_start = time()
            batch_losses = []
            batch_cnt = 0

            for batch in train_loader:
                batch_cnt+=1
                batch = batch.to(args.device)

                loss, loss_protein_frontier, loss_frontier, loss_cav, loss_class, loss_nx_attch, loss_bond, loss_pos = get_model_loss(model, batch)
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping
                optimizer.step()
                batch_losses.append(loss.item())
                if (epoch==0 and batch_cnt <= 100) or (epoch==20 and batch_cnt <= 10):
                    logger.info('Training Epoch %d | Step %d | Loss %.6f | Loss(Fron) %.6f | Loss(Cav) %.6f | Loss(Cls) %.6f | Loss(NextAttach) %.6f | Loss(Bond) %.6f | Loss(Pos) %.6f | Loss(Protein_fron) %.6f  ' % (
                            epoch+start_epoch, batch_cnt, loss.item(), loss_frontier.item(), loss_cav.item(), loss_class.item(), loss_nx_attch.item(), loss_bond.item(), loss_pos.item(), loss_protein_frontier.item()
                            ))
            average_loss = sum(batch_losses) / (len(batch_losses)+1)
            train_losses.append(average_loss)
            if verbose:
                logger.info('Training Epoch %d | Loss %.6f | Loss(Fron) %.6f | Loss(Cav) %.6f | Loss(Cls) %.6f | Loss(NextAttach) %.6f | Loss(Bond) %.6f | Loss(Pos) %.6f | Loss(Protein_fron) %.6f  ' % (
                    epoch+start_epoch, loss.item(), loss_frontier.item(), loss_cav.item(), loss_class.item(), loss_nx_attch.item(), loss_bond.item(), loss_pos.item(), loss_protein_frontier.item()
                    ))
            average_eval_loss = evaluate(epoch+start_epoch, verbose=1)
            val_losses.append(average_eval_loss)

            if config.train.scheduler.type=="plateau":
                scheduler.step(average_eval_loss)
            else:
                scheduler.step()
                
            # if val_losses[-1] < best_loss:
            if True:
                best_loss = val_losses[-1]
                if config.train.save:
                    ckpt_path = os.path.join(ckpt_dir, 'val_%d.pt' % int(epoch+start_epoch))
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': start_epoch + epoch,
                        'best_loss': best_loss
                    }, ckpt_path)
            else:
                if len(train_losses) > 20:
                    if (train_losses[-1]<train_losses[-2]):
                        if config.train.save:
                            ckpt_path = os.path.join(ckpt_dir, 'train_%d.pt' % int(epoch+start_epoch))
                            torch.save({
                                'config': config,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'epoch': start_epoch + epoch,
                                'best_loss': best_loss
                            }, ckpt_path)                      
            torch.cuda.empty_cache()

    train()
