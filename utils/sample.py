import torch
from torch.nn import functional as F
from models.embed import embed_compose
from utils.train import get_mdn_probability
from torch_geometric.nn import knn
from utils.sample_utils import rotation_matrix_align_vectors, get_pos, gen_pos, set_mol_position, bond_mols, \
    compose_feature_catdim, get_spatial_pl_graph, batch_frags, split_and_sample_topk, element2feature, Compose_data,\
    split_by_batch, double_masking, combine_atom_frag_list, atom_bonded_neighbor_aggregation
from utils.featurizer import parse_rdmol
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
from utils.dataset import merge_protein_ligand_dicts, torchify_dict
from rdkit import Chem
import copy
from utils.featurizer import parse_rdmol_base
from utils.pocket_uff import uff_geomopt
from utils.cluster import elem2frag

def sample_focal(model, h_compose, 
                idx_ligand, idx_protein,
                frontier_threshold=0,
                force_frontier=-1,
                n_samples=5, # -1 means all
                topk=False,
                frontier_scale=1):
    '''
    Sample the focal point from the protein-ligand graph
        frontier_threshold: threshold to determine frontiers(default=0)
        force_frontier: if > 0, force to sample from top k frontiers; else only start from frontiers being larger than 0
        topk: if True, sample from top k frontiers; else sample from all frontiers according to the probability
        n_samples: if > 0, force to sample from top k frontiers; else sample from all frontiers

    I recommend to set topk=True when sample from protein (since many atoms in protein part are more easily to be assigned 
             a high score, but actually they should not be the growing point)
    Then set topk=False when sample from ligand
    '''
    if len(idx_ligand) == 0: #initial pocket
        idx_ligand = idx_protein

    y_frontier_pred = model.frontier_pred(
        h_compose,
        idx_ligand,
    )[:, 0]

    ind_frontier = (y_frontier_pred > frontier_threshold)
    has_frontier = torch.sum(ind_frontier) > 0 

    if has_frontier:
        idx_frontier = idx_ligand[ind_frontier]
        p_frontier = torch.sigmoid(y_frontier_pred[ind_frontier])
    else:
        if force_frontier > 0:
            p_frontier, idx_frontier = torch.topk(y_frontier_pred, min(force_frontier, len(y_frontier_pred)))
            p_frontier = torch.sigmoid(p_frontier)
            has_frontier = True
    
    if has_frontier:  # sample from frontiers
        p_frontier_in_compose = torch.zeros(len(h_compose[0]), dtype=torch.float32, device=h_compose[0].device)
        p_frontier_in_compose_sf = torch.zeros_like(p_frontier_in_compose)
        p_frontier_in_compose_sf[idx_frontier] = F.softmax(p_frontier / frontier_scale, dim=0)
        p_frontier_in_compose[idx_frontier] = p_frontier
        if topk:
            _, idx_focal_in_compose = torch.topk(p_frontier_in_compose_sf, n_samples, dim=0)
            p_focal = p_frontier_in_compose[idx_focal_in_compose]
        else:
            idx_focal_in_compose = p_frontier_in_compose_sf.multinomial(num_samples=n_samples, replacement=True)
            p_focal = p_frontier_in_compose[idx_focal_in_compose]
    else:  # get all possible frontiers as focal, only work for frontiers being large than 0
        
        return (has_frontier, None, None, None, None)

    return (has_frontier, idx_frontier, p_frontier, idx_focal_in_compose, p_focal)

def sample_cavity(model,
    h_compose,
    compose_pos,
    idx_focal_in_compose,
    n_samples = 3 # no larger than 3
    ):
    '''
    Sample the cavity position, default generate 3 positions for each idx_focal
    '''

    n_focals = len(idx_focal_in_compose)
    # # 3: get position distributions and sample positions
    relative_pos_mu, abs_pos_mu, pos_sigma, pos_pi  = model.cavity_detector(
        h_compose,
        idx_focal_in_compose,
        compose_pos,
    )

    pos_generated = model.cavity_detector.get_maximum(abs_pos_mu, pos_sigma, pos_pi,)[:,:n_samples,:]  # n_focals, n_per_pos, 3
    n_candidate_samples = pos_generated.size(1)
    pos_generated = torch.reshape(pos_generated, [-1, 3])
    pdf_pos = get_mdn_probability(
        mu=torch.repeat_interleave(abs_pos_mu, repeats=n_candidate_samples, dim=0),
        sigma=torch.repeat_interleave(pos_sigma, repeats=n_candidate_samples, dim=0),
        pi=torch.repeat_interleave(pos_pi, repeats=n_candidate_samples, dim=0),
        pos_target=pos_generated
    )
    idx_parent = torch.repeat_interleave(torch.arange(n_focals), repeats=n_candidate_samples, dim=0).to(compose_pos.device)

    return (pos_generated, pdf_pos, idx_parent, abs_pos_mu, pos_sigma, pos_pi)  # sample_cavity
    
def sample_type(model, compose_pos, h_compose, pos_subpocket,
                n_samples=-1, num_knn=36):
    '''
    Sample the next atom/fragment type using pos_subpocket as the query
        n_samples=-1 means the greedy search; otherwise, sample from the top n candidates
    
    Generally, the n_samples should be set to 5 for broader exploration
    '''
    n_query = len(pos_subpocket)
    edge_new_site_knn_index = knn(x=compose_pos, y=pos_subpocket, k=num_knn, num_workers=16) # k is adjustable
    edge_index_q_cps_knn_batch = torch.arange(n_query).repeat_interleave(num_knn).to(compose_pos.device)
    next_frag_type = model.type_predictor(
        pos_subpocket,
        compose_pos,
        h_compose,
        edge_new_site_knn_index, #data['edge_new_site_knn_index']
        edge_index_q_cps_knn_batch
    )
    next_frag_type = torch.softmax(next_frag_type, dim=-1)
    
    if n_samples > 0:
        element_pred = next_frag_type.multinomial(num_samples=n_samples, replacement=True).reshape(-1)
        idx_parent = torch.repeat_interleave(torch.arange(n_query), n_samples, dim=0).to(compose_pos.device)
        element_prob = next_frag_type[idx_parent, element_pred]
    else:
        element_pred = next_frag_type.argmax(dim=-1)
        idx_parent = torch.arange(n_query).to(compose_pos.device)
        element_prob = next_frag_type[torch.arange(len(next_frag_type)), element_pred]
    
    return element_pred, element_prob, idx_parent

def sample_attach_point(model,current_wids, next_motif_wids, h_compose, idx_focal, frag_base):
    '''
    Only sample the top-1 attachment of each fragment
    '''
    device = idx_focal.device
    next_smis = [frag_base['data_base_smiles'][i].item().decode() for i in next_motif_wids]
    next_mols = [Chem.MolFromSmiles(smi) for smi in next_smis]
    frag_mask = (next_motif_wids>7) # neglect the single-atom
    frag_tobe_attach = [next_mols[i] for i in range(len(next_mols)) if frag_mask[i]] # select the fragment
    
    if frag_tobe_attach == []:
        frag_attch_pred = torch.zeros(len(next_mols), dtype=torch.int64)
        return next_mols, frag_attch_pred
    
    # only predict the attachment of the fragment
    node_feat_frags, edge_index_frags, edge_features_frags, node_batch_frags = batch_frags(frag_tobe_attach)
    focal_info = h_compose[0][idx_focal[frag_mask]], h_compose[1][idx_focal[frag_mask]]
    frag_node_2d = model.attacher(node_feat_frags.to(device), edge_index_frags.to(device), edge_features_frags.to(device), 
            current_wids.to(device), next_motif_wids, focal_info, node_batch_frags)
    
    # frag_node_2d is a series attachment score assigned on each node, while node_batch_frags records the attribution of each node
    # split and sample the attachment according to the batch attribution
    frag_attch_pred = split_and_sample_topk(frag_node_2d, node_batch_frags, k=1) 

    #for next_atom, default is 0; for next_frag, assign the predicted attachment
    attach_point = torch.zeros(len(next_mols), dtype=torch.int64) 
    attach_point[frag_mask] = torch.tensor(frag_attch_pred)

    return next_mols, attach_point.to(device)

def chemical_initialization(first_mol, second_mol, first_attach:int, second_attach:int, first_attach_pos:torch.tensor, \
                            second_attach_pos:torch.tensor, bond_type=Chem.rdchem.BondType.SINGLE, align_first_attach=True):
    '''
    This function solve the point rotation problem by chemical way, where aligned vec is first_attach_pos - second_attach_pos.
    In detail, it uses the first_attach_pos - second_attach_pos as the a target vector, and then generate a conformation for bonded_mol in vacuum.
    And then implement the rotation matric to match the corresponding part in bonded_mol to the target vector.
    
    This version fixes the first_mol position in the bonded_mol. I currently use this version.
    '''
    bonded_mol_vacuum = bond_mols(first_mol, second_mol, first_attach, second_attach, bond_type=bond_type)
    second_attach_in_bonded = first_mol.GetNumAtoms() + second_attach 
    bonded_mol_vacuum.RemoveAllConformers()
    AllChem.EmbedMolecule(bonded_mol_vacuum)
    bonded_mol_pos_vacuum = get_pos(bonded_mol_vacuum)

    vec_vacuum = (bonded_mol_pos_vacuum[first_attach] - bonded_mol_pos_vacuum[second_attach_in_bonded]).reshape(-1, 3)
    vec_target = first_attach_pos - second_attach_pos

    vec_align_mat = rotation_matrix_align_vectors(vec_vacuum,vec_target)
    bonded_mol_pos_rotated = (vec_align_mat @ bonded_mol_pos_vacuum.T).mT.squeeze(0)

    if align_first_attach:
        bonded_mol_pos_aligned = bonded_mol_pos_rotated + (first_attach_pos - bonded_mol_pos_rotated[first_attach]).reshape(-1,3)
    else:
        bonded_mol_pos_aligned = bonded_mol_pos_rotated + (second_attach_pos - bonded_mol_pos_rotated[second_attach_in_bonded]).reshape(-1,3)
    
    # initialized_frag_pos = bonded_mol_pos_aligned[-second_mol.GetNumAtoms():,:]
    # initialized_mol = set_mol_position(bonded_mol_vacuum, bonded_mol_pos_aligned.numpy())
    first_second_pos = torch.zeros(bonded_mol_vacuum.GetNumAtoms(), 3)
    first_second_pos[:first_mol.GetNumAtoms(),:] = get_pos(first_mol)
    first_second_pos[first_mol.GetNumAtoms():,:] = bonded_mol_pos_aligned.detach()[first_mol.GetNumAtoms():,:]

    bonded_mol_aligned = set_mol_position(bonded_mol_vacuum, first_second_pos)

    return bonded_mol_pos_aligned, bonded_mol_aligned

from rdkit.Chem import rdMolAlign
def chemical_initialization_v2(first_mol, second_mol, first_attach:int, second_attach:int, first_attach_pos:torch.tensor, \
                            second_attach_pos:torch.tensor, bond_type=Chem.rdchem.BondType.SINGLE):
    '''
    This version do not fix the first_mol positions, it uses the alighment method to make the first part of molecule as close as possible (rdmolAlign)
    I think it may broaden the search space, but it is not used in the current version.
    My major concerns are:
        (1) the first_mol conformation is regenerated, which may be different from the original one. I cannot detect how much difference it is. 
        (2) the second_attach pos is not guaranteed to be the same as the input one, since the rdmolAlign is implemented after the translation
    '''
    bonded_mol_initial = bond_mols(first_mol, second_mol, first_attach, second_attach, bond_type=bond_type)
    second_attach_in_bonded = first_mol.GetNumAtoms() + second_attach 

    bonded_mol_vacuum = copy.deepcopy(bonded_mol_initial)
    bonded_mol_vacuum.RemoveAllConformers()
    AllChem.EmbedMolecule(bonded_mol_vacuum)
    bonded_mol_pos_vacuum = get_pos(bonded_mol_vacuum)

    vec_vacuum = (bonded_mol_pos_vacuum[first_attach] - bonded_mol_pos_vacuum[second_attach_in_bonded]).reshape(-1, 3)
    vec_target = first_attach_pos - second_attach_pos
    vec_align_mat = rotation_matrix_align_vectors(vec_vacuum,vec_target)
    bonded_mol_pos_rotated = (vec_align_mat @ bonded_mol_pos_vacuum.T).mT.squeeze(0)
    bonded_mol_pos_aligned = bonded_mol_pos_rotated + (first_attach_pos - bonded_mol_pos_rotated[first_attach]).reshape(-1,3)

    atom_map = list(range(first_mol.GetNumAtoms()))
    rmsd = rdMolAlign.AlignMol(first_mol, bonded_mol_vacuum, atomMap=list(zip(atom_map, atom_map)))


    bonded_mol_aligned = set_mol_position(bonded_mol_vacuum, bonded_mol_pos_aligned.detach().numpy())

    return bonded_mol_pos_aligned, bonded_mol_aligned


def mol_bonding(current_mols, next_mols, idx_focal, idx_next_attach, focal_pos, next_attach_pos, bond_types=None, align_first_attach=False):

    bonded_confs = []
    bonded_mols = []
    bonded_fail_mask_in_frag_mask = torch.ones(len(current_mols), dtype=torch.bool)
    if bond_types is None:
        bond_types = [Chem.rdchem.BondType.SINGLE] * len(current_mols)
    
    idx_focal = idx_focal.to('cpu')
    idx_next_attach = idx_next_attach.to('cpu')
    focal_pos = focal_pos.to('cpu')
    next_attach_pos = next_attach_pos.to('cpu')
    for i in range(len(current_mols)):
        try:
            current_mol = current_mols[i]
            next_mol = next_mols[i]
            bonded_conf, bonded_mol = chemical_initialization(current_mol, next_mol, 
                                                idx_focal[i].tolist(),  # since the pocket is empty
                                                idx_next_attach[i].tolist(), \
                                                focal_pos[i].reshape(-1,3), \
                                                next_attach_pos[i].reshape(-1,3),
                                                bond_types[i],
                                                align_first_attach=align_first_attach)
            bonded_confs.append(bonded_conf)
            bonded_mols.append(bonded_mol)

        except Exception as e:
            #print(e)
            bonded_fail_mask_in_frag_mask[i] = False
            continue
        
    return bonded_confs, bonded_fail_mask_in_frag_mask, bonded_mols


def sample_bond(model, added_data, h_compose, compose_pos, idx_focal, next_atom_pos, current_wids, next_wids, next_mols, frag_mask, attach_points):
    device = idx_focal.device

    frag_tobe_attach = [next_mols[i] for i in range(len(next_mols)) if frag_mask[i]] # select the fragment
    atom_element_feature = F.one_hot(next_wids[~frag_mask], num_classes=8)
    fragment_parsed = [parse_rdmol_base(mol) for mol in frag_tobe_attach]
    frag_element_feature = [F.one_hot(fragment_parsed[i]['type_feature'], num_classes=8)[attach_points[frag_mask][i]].view(1,-1) for i in range(len(fragment_parsed))]
    
    if frag_element_feature == []:
        frag_element_feature = torch.zeros(0, 8)
    else:
        frag_element_feature = torch.cat(frag_element_feature, dim=0)

    next_motif_bonded_atom_feature = torch.zeros(len(next_mols), 8).to(device)
    next_motif_bonded_atom_feature[~frag_mask] = atom_element_feature.float().to(device)
    next_motif_bonded_atom_feature[frag_mask] = frag_element_feature.float().to(device)
    

    frag_bonded_b_nei_edge_features = [atom_bonded_neighbor_aggregation(fragment_parsed[i]['bond_index'], 
                                                                                    F.one_hot(fragment_parsed[i]['bond_type']-1, num_classes=4), 
                                                                                    attach_points[frag_mask][i], 
                                                                                    fragment_parsed[i]['element'].shape[0]).view(1,-1) 
                                                                                for i in range(len(attach_points[frag_mask]))]
    if frag_bonded_b_nei_edge_features == []:
        frag_bonded_b_nei_edge_features = torch.zeros(0, 4)
    else:
        frag_bonded_b_nei_edge_features = torch.cat(frag_bonded_b_nei_edge_features, dim=0)
    

    bonded_b_nei_edge_features = torch.zeros(len(next_mols), 4).to(device)
    bonded_b_nei_edge_features[~frag_mask] = torch.zeros((~frag_mask).sum(), 4, dtype=torch.float32).to(device)
    bonded_b_nei_edge_features[frag_mask] = frag_bonded_b_nei_edge_features.float().to(device)

    context_edge_index = added_data['ligand_bond_index']
    context_edge_feature = added_data['ligand_bond_feature']
    bonded_a_nei_edge_features = atom_bonded_neighbor_aggregation(context_edge_index, context_edge_feature, idx_focal, added_data['ligand_pos'].shape[0])
    focal_info = [h_compose[0][idx_focal], h_compose[1][idx_focal]]
    focal_pos = compose_pos[idx_focal]

    bond_pred = model.bonder(focal_info, focal_pos, \
                         next_atom_pos, current_wids, next_wids, \
                                    next_motif_bonded_atom_feature, bonded_a_nei_edge_features, bonded_b_nei_edge_features)
    
    bond_type_id = torch.argmax(bond_pred, dim=-1)
    
    return bond_type_id


def compose_mol_data(mol, data, focal_idx, next_attach_idx, num_atom_frag, transform_ligand, protein=False):
    
    device = focal_idx.device
    compose_data = Compose_data(**{})

    
    mol_dict = parse_rdmol(mol)
    mol_parse = merge_protein_ligand_dicts(protein_dict={}, ligand_dict=mol_dict)
    mol_parse = torchify_dict(mol_parse)
    mol_parse = transform_ligand(Compose_data(**mol_parse)).to(device)

    compose_next_feature, idx_ligand_next, idx_protein_next = compose_feature_catdim(mol_parse['ligand_atom_feature_full'], data['protein_atom_feature'])
    compose_pos_next = torch.cat([mol_parse['ligand_pos'], data['protein_pos']], dim=0)
    ligand_bond_index = mol_parse['ligand_bond_index']
    ligand_bond_feature = F.one_hot(mol_parse['ligand_bond_type'] -1, num_classes=4)
    edge_index_pos_pred, edge_feature_pos_pred = get_spatial_pl_graph(ligand_bond_index, ligand_bond_feature, mol_parse['ligand_pos'], data['protein_pos'], num_knn=24,num_workers=12)
    
    compose_data['compose_next_feature'] = compose_next_feature
    compose_data['compose_pos_next'] = compose_pos_next
    compose_data['edge_index_pos_pred'] = edge_index_pos_pred
    compose_data['edge_feature_pos_pred'] = edge_feature_pos_pred
    compose_data['idx_ligand_next'] = idx_ligand_next
    compose_data['idx_protein_next'] = idx_protein_next
    
    num_atom_ligand = mol_parse['ligand_atom_feature_full'].size(0)
    if protein:
        focal_idx = focal_idx + (num_atom_ligand - 0) # 
        next_attach_idx = next_attach_idx 
        compose_data['next_frag_pos'] = mol_parse['ligand_pos']
        compose_data['next_frag_idx'] = torch.arange(num_atom_ligand)
        compose_data['ligand_pos_mask_idx'] = torch.arange(num_atom_ligand).to(device)
    else:
        focal_idx = focal_idx
        next_attach_idx = next_attach_idx + (num_atom_ligand - num_atom_frag) # remaining part
        compose_data['next_frag_pos'] = mol_parse['ligand_pos'][-num_atom_frag:,:]
        compose_data['next_frag_idx'] = torch.arange(num_atom_ligand-num_atom_frag, num_atom_ligand)
        compose_data['ligand_pos_mask_idx'] = torch.arange(num_atom_ligand-num_atom_frag, num_atom_ligand).to(device)

    compose_data['focal_idx'] = focal_idx
    compose_data['next_attach_idx'] = next_attach_idx
    compose_data['num_nodes'] = compose_data['compose_next_feature'].size(0)

    return compose_data


def sample_pos_dihedral(model, mols, data, idx_focal, idx_next_attach, num_next_frag_nums, transform_ligand, \
                        protein_initial=False, batch_size=None):
    if len(mols) == 0:
        return []
    
    compose_rotate_data = [compose_mol_data(mols[i], data, 
                                            idx_focal[i], \
                                            idx_next_attach[i], 
                                            num_next_frag_nums[i],
                                            transform_ligand,
                                            protein=protein_initial) \
                                            for i in range(len(mols))]
    
    num_sample = len(compose_rotate_data)
    if batch_size == None:
        batch_size = num_sample

    updated_pos_full = []
    for i in range(0, num_sample, batch_size):
        end = min(i + batch_size, num_sample)
        mini_batch_data = compose_rotate_data[i:end]
        rotate_batch = Batch.from_data_list(mini_batch_data, follow_batch=['idx_ligand_next','next_frag_pos'])
        h_compose_pos_next_pred = embed_compose(rotate_batch.compose_next_feature, rotate_batch.compose_pos_next, rotate_batch.idx_ligand_next, rotate_batch.idx_protein_next,
                                model.ligand_atom_emb, model.protein_atom_emb, model.emb_dim)
        
        alpha = model.pos_predictor(h_compose_pos_next_pred, rotate_batch.compose_pos_next, rotate_batch.edge_index_pos_pred, rotate_batch.edge_feature_pos_pred, \
                    rotate_batch.focal_idx, rotate_batch.next_attach_idx, rotate_batch.idx_ligand_next, rotate_batch.idx_ligand_next_batch)
        
        rotated_pos = model.pos_predictor.pos_update(alpha, rotate_batch.compose_pos_next[rotate_batch.focal_idx], \
                                rotate_batch.compose_pos_next[rotate_batch.next_attach_idx], 
                                rotate_batch.next_frag_pos, rotate_batch.next_frag_pos_batch)
        
        rotated_pos = split_by_batch(rotated_pos, rotate_batch.next_frag_pos_batch)
        updated_pos_ = split_by_batch(rotate_batch['compose_pos_next'][rotate_batch['idx_ligand_next']], rotate_batch['idx_ligand_next_batch'])

        for i in range(len(rotated_pos)):
            updated_pos_[i][-len(rotated_pos[i]):] = rotated_pos[i].detach().cpu()
        
        updated_pos_full.extend(updated_pos_)
        
    return updated_pos_full

def sample_pos_cartesian(model, mols, data, idx_focal, idx_next_attach, num_next_frag_nums, transform_ligand, \
                        protein_initial=False, batch_size=None):
    if len(mols) == 0:
        return []
    
    compose_rotate_data = [compose_mol_data(mols[i], data, 
                                            idx_focal[i], \
                                            idx_next_attach[i], 
                                            num_next_frag_nums[i],
                                            transform_ligand,
                                            protein=protein_initial) \
                                            for i in range(len(mols))]
    
    num_sample = len(compose_rotate_data)
    if batch_size == None:
        batch_size = num_sample

    updated_pos_full = []
    for i in range(0, num_sample, batch_size):
        end = min(i + batch_size, num_sample)
        mini_batch_data = compose_rotate_data[i:end]
        rotate_batch = Batch.from_data_list(mini_batch_data, follow_batch=['idx_ligand_next','next_frag_pos'])
        h_compose_pos_next_pred = embed_compose(rotate_batch.compose_next_feature, rotate_batch.compose_pos_next, rotate_batch.idx_ligand_next, rotate_batch.idx_protein_next,
                                model.ligand_atom_emb, model.protein_atom_emb, model.emb_dim)
        
        _, _, _, full_updated_pos = model.pos_predictor(h_compose_pos_next_pred[0], rotate_batch.edge_feature_pos_pred, rotate_batch.edge_index_pos_pred, \
            rotate_batch.compose_pos_next, rotate_batch.ligand_pos_mask_idx, update_pos=True)
        
        updated_pos = split_by_batch(full_updated_pos[rotate_batch['idx_ligand_next']] , rotate_batch['idx_ligand_next_batch'])
        updated_pos_full.extend(updated_pos)
        
    return updated_pos_full

def sample_pos_geomopt(model, mols, data, idx_focal, idx_next_attach, num_next_frag_nums, transform_ligand, \
                        protein_initial=False, batch_size=None):
    if len(mols) == 0:
        return []
    
    compose_rotate_data = [compose_mol_data(mols[i], data, 
                                            idx_focal[i], \
                                            idx_next_attach[i], 
                                            num_next_frag_nums[i],
                                            transform_ligand,
                                            protein=protein_initial) \
                                            for i in range(len(mols))]
    updated_mols = []
    for i in range(len(compose_rotate_data)):
        ligand_fixed_atomid = list(set(compose_rotate_data[i]['idx_ligand_next'].tolist()) - set(compose_rotate_data[i]['next_frag_idx'].tolist()))
        updated_mol = uff_geomopt(rd_mol=mols[i], pkt_mol=data.pkt_mol, lig_constraint=ligand_fixed_atomid, lig_h=False, voice=False)
        if updated_mol is not None:
            updated_mols.append(updated_mol)
        else:
            updated_mols.append(mols[i])
    return updated_mols


from utils.dataset import ComplexData
def sample_initial(model, data, frag_base, transform_ligand):

    compose_feature = data['compose_feature'].float()
    compose_pos = data['compose_pos'].to(torch.float32)
    idx_ligand = data['idx_ligand_ctx_in_compose']
    idx_protein = data['idx_protein_in_compose']
    compose_knn_edge_feature = data['compose_knn_edge_feature']
    compose_knn_edge_index = data['compose_knn_edge_index']

    h_compose = embed_compose(compose_feature, compose_pos, idx_ligand, idx_protein,
        model.ligand_atom_emb, model.protein_atom_emb, model.emb_dim)

    h_compose = model.encoder(
        node_attr = h_compose,
        pos = compose_pos,
        edge_index = compose_knn_edge_index,
        edge_feature = compose_knn_edge_feature,
    )

    has_frontier, idx_frontier, p_frontier, idx_focal_in_compose, p_focal = sample_focal(model, h_compose, idx_ligand, idx_protein, topk=True, n_samples=10)
    if has_frontier:
        # sample_cavity will triple the number of pools
        pos_generated, pdf_pos, idx_parent, abs_pos_mu, pos_sigma, pos_pi= sample_cavity(model, h_compose, compose_pos, idx_focal_in_compose)
        idx_focal_in_compose, p_focal = idx_focal_in_compose[idx_parent], p_focal[idx_parent] 
        # element prediction times the number of idx_focal_in_compose
        element_pred, element_prob, idx_parent = sample_type(model, compose_pos, h_compose, pos_generated, n_samples=5)
        # expand the p_focal and pdf_pos
        p_focal, pdf_pos = p_focal[idx_parent], pdf_pos[idx_parent]

        idx_focal_in_compose_af_element, pos_generated_af_element = idx_focal_in_compose[idx_parent], pos_generated[idx_parent] 
        # focal_protein_wids = element2feature(data['protein_element'][idx_focal_in_compose_af_element])
        focal_protein_wids = torch.tensor([frag_base['data_base_features'].shape[0]]).repeat(len(idx_focal_in_compose_af_element)).to(pos_generated.device)
        # sample the next attachment point, only fragments are considered
        next_mols, attach_points = sample_attach_point(model, focal_protein_wids, element_pred, h_compose, idx_focal_in_compose_af_element, frag_base)
        
        frag_mask = (element_pred>7) # neglect the single-atom

        # current_smis = [frag_base['data_base_smiles'][i].item().decode() for i in focal_protein_wids]
        current_smis = ['C'] * len(focal_protein_wids)
        current_mols = [Chem.MolFromSmiles(smi) for smi in current_smis]
        current_frags = [x for x,m in zip(current_mols, frag_mask) if m]
        current_frags = [gen_pos(mol) for mol in current_frags]
        next_frags = [x for x,m in zip(next_mols, frag_mask) if m]
        next_frags = [gen_pos(mol) for mol in next_frags]
        
        bonded_confs, bonded_fail_mask_in_frag_mask, bonded_mols = mol_bonding(current_frags, next_frags,idx_focal=torch.zeros(len(next_frags), dtype=torch.int64), \
                idx_next_attach=attach_points[frag_mask], focal_pos=compose_pos[idx_focal_in_compose_af_element[frag_mask]], \
                next_attach_pos = pos_generated_af_element[frag_mask], align_first_attach=False) # align the molecule to the predicted position
        
        # here bonded mol is not the next_mols_to_be_rotated, because the first atom is virtual atom in proteins
        next_mols_tobe_rotated = [x for x,m in zip(next_frags, bonded_fail_mask_in_frag_mask) if m]
        next_mols_tobe_rotated = [set_mol_position(next_mols_tobe_rotated[i], bonded_confs[i][1:]) for i in range(len(bonded_confs))]
        
        if model.pos_pred_type == 'dihedral':
            rotated_pos = sample_pos_dihedral(model, next_mols_tobe_rotated, data, idx_focal_in_compose_af_element[frag_mask][bonded_fail_mask_in_frag_mask], \
                                            attach_points[frag_mask][bonded_fail_mask_in_frag_mask], 
                                            [mol.GetNumAtoms() for mol in next_mols_tobe_rotated],
                                            transform_ligand,
                                            protein_initial=True,
                                            batch_size=8) # if raise CUDA Memory Error, reduce the batch_size
            next_frags_predicted = [set_mol_position(next_mols_tobe_rotated[i], rotated_pos[i].detach().cpu()) for i in range(len(rotated_pos))]
        elif model.pos_pred_type == 'cartesian':
            rotated_pos = sample_pos_cartesian(model, next_mols_tobe_rotated, data, idx_focal_in_compose_af_element[frag_mask][bonded_fail_mask_in_frag_mask], \
                                            attach_points[frag_mask][bonded_fail_mask_in_frag_mask], 
                                            [mol.GetNumAtoms() for mol in next_mols_tobe_rotated],
                                            transform_ligand,
                                            protein_initial=True,
                                            batch_size=8) # if raise CUDA Memory Error, reduce the batch_size
            next_frags_predicted = [set_mol_position(next_mols_tobe_rotated[i], rotated_pos[i].detach().cpu()) for i in range(len(rotated_pos))]
        else:
            next_frags_predicted = sample_pos_geomopt(model, next_mols_tobe_rotated, data, idx_focal_in_compose_af_element[frag_mask][bonded_fail_mask_in_frag_mask], \
                                            attach_points[frag_mask][bonded_fail_mask_in_frag_mask], 
                                            [mol.GetNumAtoms() for mol in next_mols_tobe_rotated],
                                            transform_ligand,
                                            protein_initial=True,
                                            batch_size=8) # if raise CUDA Memory Error, reduce the batch_size
            # next_frags_predicted = next_mols_tobe_rotated
            # raise NotImplementedError(f'The {model.pos_pred_type} is not implemented')

        

        success_frag_mask = double_masking(frag_mask, bonded_fail_mask_in_frag_mask) 
        next_atoms = [x for x,m in zip(next_mols, ~frag_mask) if m]
        next_atoms = [gen_pos(mol) for mol in next_atoms]
        next_atoms = [set_mol_position(next_atoms[i], pos_generated_af_element[~frag_mask][i].detach().cpu().reshape(-1,3)) for i in range(len(pos_generated_af_element[~frag_mask]))]
        
        initial_mols = combine_atom_frag_list(next_atoms, next_frags_predicted, ~frag_mask, success_frag_mask)
        
        # _ = [Chem.SanitizeMol(x) for x in initial_mols if x is not None]

        data_list = []
        for i in range(len(initial_mols)):
            if initial_mols[i] is not None:
                
                try:
                    Chem.SanitizeMol(initial_mols[i])
                except Exception as e:
                    print('sample_initial error: ',e)

                new_data = {}
                new_data['ligand_mol'] = initial_mols[i]
                new_data['current_wid'] = element_pred[i].detach().cpu()
                new_data['p_focal'] = p_focal[i].detach().cpu()
                new_data['p_pos'] = pdf_pos[i].detach().cpu()
                new_data['p_element'] = element_prob[i].detach().cpu()
                new_data['p_all'] = new_data['p_focal'] * new_data['p_pos'] * new_data['p_element']
                new_data['is_fragment'] = (element_pred[i]>7).detach().cpu()
                new_data['status'] = 'running'
                data_list.append(ComplexData(**new_data))

        return data_list
    
    else: 
        
        print('Cannot Find the Frontier at the given frontier threshold, this error happens in the sample_initial function')
        print('Please check your data and make sure the model has loaded the pre-trained parameters')
        return []
    
type_to_bond = {0: Chem.rdchem.BondType.SINGLE , 1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC}
# 0 is fake bond for bonding the first atom to the protein
def sample_next_state(model, added_data, frag_base, transform_ligand, force_frontier=-1):
    # single-data-next
    compose_feature = added_data['compose_feature'].float()
    compose_pos = added_data['compose_pos'].to(torch.float32)
    idx_ligand = added_data['idx_ligand_ctx_in_compose']
    idx_protein = added_data['idx_protein_in_compose']
    compose_knn_edge_feature = added_data['compose_knn_edge_feature']
    compose_knn_edge_index = added_data['compose_knn_edge_index']

    h_compose = embed_compose(compose_feature, compose_pos, idx_ligand, idx_protein,
        model.ligand_atom_emb, model.protein_atom_emb, model.emb_dim)

    h_compose = model.encoder(
        node_attr = h_compose,
        pos = compose_pos,
        edge_index = compose_knn_edge_index,
        edge_feature = compose_knn_edge_feature,
    )

    has_frontier, idx_frontier, p_frontier, idx_focal_in_compose, p_focal = sample_focal(model, h_compose, idx_ligand, idx_protein, topk=False, force_frontier=force_frontier, n_samples=10)
    if has_frontier:
        # sample_cavity will triple the number of pools
        pos_generated, pdf_pos, idx_parent, abs_pos_mu, pos_sigma, pos_pi= sample_cavity(model, h_compose, compose_pos, idx_focal_in_compose, n_samples=3)
        idx_focal_in_compose, p_focal = idx_focal_in_compose[idx_parent], p_focal[idx_parent] 
        element_pred, element_prob, idx_parent = sample_type(model, compose_pos, h_compose, pos_generated, n_samples=5)
        p_focal, pdf_pos = p_focal[idx_parent], pdf_pos[idx_parent]
        idx_focal_in_compose_af_element, pos_generated_af_element = idx_focal_in_compose[idx_parent], pos_generated[idx_parent] 
        # current_wids = added_data['current_wid'].repeat(len(idx_focal_in_compose_af_element)) # Logic error
        clique_dict = elem2frag(added_data['ligand_mol'], frag_base)
        mapped_frag_idx = [clique_dict[idx.item()] for idx in idx_focal_in_compose_af_element]
        current_wids = torch.tensor(mapped_frag_idx, device=idx_focal_in_compose_af_element.device)
        
        # global frag_mask 
        # global bonded_fail_mask_in_frag_mask

        frag_mask = (element_pred>7) # neglect the single-atom
        if frag_mask.any():
            next_mols, attach_points = sample_attach_point(model, current_wids, element_pred, h_compose, idx_focal_in_compose_af_element, frag_base)
        else:
            next_smis = [frag_base['data_base_smiles'][i].item().decode() for i in element_pred]
            next_mols = [Chem.MolFromSmiles(smi) for smi in next_smis]
            attach_points = torch.zeros(len(element_pred), dtype=torch.int64)
            
        bond_prediction = sample_bond(model, added_data, h_compose, compose_pos, idx_focal_in_compose_af_element, pos_generated_af_element, current_wids, element_pred, next_mols, frag_mask, attach_points)

        # current_frag = gen_pos(Chem.MolFromSmiles(frag_base['data_base_smiles'][added_data['current_wid']].item().decode()))
        # current_frag = set_mol_position(current_frag, added_data['current_motif_pos'])

        current_mol = copy.deepcopy(added_data['ligand_mol'])
        current_mols_4_frag = [current_mol] * frag_mask.sum()
        current_mols_4_atom = [current_mol] * (~frag_mask).sum()

        next_frags = [x for x,m in zip(next_mols, frag_mask) if m]
        next_frags = [gen_pos(mol) for mol in next_frags]

        rdkit_bond = [type_to_bond[bond]  for bond in bond_prediction.tolist()]
        rdkit_frag_bond = [x for x,m in zip(rdkit_bond, frag_mask) if m]
        rdkit_atom_bond = [x for x,m in zip(rdkit_bond, ~frag_mask) if m]

        bonded_confs, bonded_fail_mask_in_frag_mask, bonded_mol_list = mol_bonding(current_mols_4_frag, next_frags,idx_focal=idx_focal_in_compose_af_element[frag_mask], \
                idx_next_attach=attach_points[frag_mask], focal_pos=compose_pos[idx_focal_in_compose_af_element[frag_mask]], \
                next_attach_pos = pos_generated_af_element[frag_mask], bond_types=rdkit_frag_bond, align_first_attach=True) # since the covalent bond is relatively short, 
                                                                                                                            # so I believe there is little difference between align_first_attach=True and False

        next_frag_tobe_rotated_atom_num = [x.GetNumAtoms() for x,m in zip(next_frags, bonded_fail_mask_in_frag_mask) if m] # only provide 
        if model.pos_pred_type == 'dihedral':
            updated_pos = sample_pos_dihedral(model, bonded_mol_list, added_data, idx_focal_in_compose_af_element[frag_mask][bonded_fail_mask_in_frag_mask], \
                                            attach_points[frag_mask][bonded_fail_mask_in_frag_mask], 
                                            next_frag_tobe_rotated_atom_num,
                                            transform_ligand,
                                            protein_initial=False,
                                            batch_size=8) # if raise CUDA Memory Error, reduce the batch_size
            mol_frags_predicted = [set_mol_position(bonded_mol_list[i], updated_pos[i].detach().cpu()) for i in range(len(updated_pos))]
        elif model.pos_pred_type == 'cartesian':
            updated_pos = sample_pos_cartesian(model, bonded_mol_list, added_data, idx_focal_in_compose_af_element[frag_mask][bonded_fail_mask_in_frag_mask], \
                                            attach_points[frag_mask][bonded_fail_mask_in_frag_mask], 
                                            next_frag_tobe_rotated_atom_num,
                                            transform_ligand,
                                            protein_initial=False,
                                            batch_size=8) # if raise CUDA Memory Error, reduce the batch_size
            mol_frags_predicted = [set_mol_position(bonded_mol_list[i], updated_pos[i].detach().cpu()) for i in range(len(updated_pos))]
        else:
            mol_frags_predicted = sample_pos_geomopt(model, bonded_mol_list, added_data, idx_focal_in_compose_af_element[frag_mask][bonded_fail_mask_in_frag_mask], \
                                            attach_points[frag_mask][bonded_fail_mask_in_frag_mask], 
                                            next_frag_tobe_rotated_atom_num,
                                            transform_ligand,
                                            protein_initial=False,
                                            batch_size=8) # if raise CUDA Memory Error, reduce the batch_size
            # raise NotImplementedError('pos_pred_type should be dihedral or cartesian')
        

        next_atoms = [x for x,m in zip(next_mols, ~frag_mask) if m]
        next_atoms = [gen_pos(mol) for mol in next_atoms]
        next_atoms = [set_mol_position(next_atoms[i], pos_generated_af_element[~frag_mask][i].detach().cpu().reshape(-1,3)) for i in range(len(pos_generated_af_element[~frag_mask]))]
        mols_atom_predicted = [bond_mols(current_mols_4_atom[i], next_atoms[i], idx_focal_in_compose_af_element[~frag_mask][i].tolist(), 0, bond_type=rdkit_atom_bond[i]) for i in range(len(current_mols_4_atom))]

        
        success_frag_mask = double_masking(frag_mask, bonded_fail_mask_in_frag_mask)

        generated_mols = combine_atom_frag_list(mols_atom_predicted, mol_frags_predicted, ~frag_mask, success_frag_mask)
        
        # _ = [Chem.SanitizeMol(x) for x in generated_mols if x is not None]

        data_list = []
        for i in range(len(generated_mols)):
            if generated_mols[i] is not None:
                try:
                    Chem.SanitizeMol(generated_mols[i])
                except Exception as e:
                    pass
                    # print('sample_next_state error: ',e)
                new_data = {}
                new_data['ligand_mol'] = generated_mols[i]
                new_data['p_focal'] = p_focal[i].detach().cpu()
                new_data['p_pos'] = pdf_pos[i].detach().cpu()
                new_data['p_element'] = element_prob[i].detach().cpu()
                new_data['p_all'] = new_data['p_focal'] * new_data['p_pos'] * new_data['p_element']
                new_data['is_fragment'] = (element_pred[i]>7).detach().cpu()
                new_data['current_wid'] = element_pred[i].detach().cpu()
                new_data['status'] = 'running'
                data_list.append(ComplexData(**new_data))
    else:
        new_data = {}
        new_data['ligand_mol'] = added_data['ligand_mol']
        new_data['current_wid'] = added_data['current_wid'].detach().cpu()
        new_data['status'] = 'finished'
        new_data['p_focal'] = torch.tensor(1)
        new_data['p_pos'] = torch.tensor(1)
        new_data['p_element'] = torch.tensor(1)
        new_data['p_all'] = torch.tensor(3)
        new_data['is_fragment'] = torch.tensor(0)
        new_data = ComplexData(**new_data)
        return [new_data]
    
    return data_list


def get_init(model, data, frag_base, transform_ligand, threshold_dict):

    while True:
        data_list = sample_initial(model, data, frag_base, transform_ligand)
        for data in data_list:
            is_high_prob = ((data['p_focal'] > threshold_dict.focal_threshold) & \
                            (data['p_pos'] > threshold_dict.pos_threshold) & \
                            (data['p_element'] > threshold_dict.element_threshold) )
            data.is_high_prob = is_high_prob
    
        data_next_list = [data for data in data_list if data.is_high_prob]

        if len(data_next_list) == 0:
            p_focals = torch.tensor([data['p_focal'] for data in data_list])
            p_poses = torch.tensor([data['p_pos'] for data in data_list])
            p_elements = torch.tensor([data['p_element'] for data in data_list])

            if torch.all(p_poses < threshold_dict.pos_threshold):
                threshold_dict.pos_threshold = threshold_dict.pos_threshold / 2
                print('Positional probability threshold is too high. Change to %f' % threshold_dict.pos_threshold)
            elif torch.all(p_focals < threshold_dict.focal_threshold):
                threshold_dict.focal_threshold = threshold_dict.focal_threshold / 2
                print('Focal probability threshold is too high. Change to %f' % threshold_dict.focal_threshold)
            elif torch.all(p_elements < threshold_dict.element_threshold):
                threshold_dict.element_threshold = threshold_dict.element_threshold / 2
                print('Element probability threshold is too high. Change to %f' % threshold_dict.element_threshold)
            else:
                print('Initialization failed. This problem is caused by get_init function')
                break
        else:
            return data_next_list

def get_init_only_frag(model, data, frag_base, transform_ligand):
    try_cnt = 0 
    while True:
        
        try_cnt += 1
        if try_cnt > 5:
            print('Initialization failed. This problem is caused by get_init_only_frag function')
            return []
        
        data_list = sample_initial(model, data, frag_base, transform_ligand)
        data_next_list = [data for data in data_list if data['is_fragment']]
        if len(data_next_list) == 0:
            print(f'Try sample initial state again, count: {try_cnt}')
            continue
        else:
            return data_next_list
        
def get_next(model, data, frag_base, transform_ligand, threshold_dict = None, force_frontier=-1):
    
    data_next_list = sample_next_state(model, data, frag_base, transform_ligand, force_frontier=force_frontier)
    
    if threshold_dict is not None:
        for data in data_next_list:
            is_high_prob = ((data['p_focal'] > threshold_dict.focal_threshold) & \
                            (data['p_pos'] > threshold_dict.pos_threshold) & \
                            (data['p_element'] > threshold_dict.element_threshold) )
            data.is_high_prob = is_high_prob
        data_next_list = [data for data in data_next_list if data.is_high_prob]
        
    return data_next_list