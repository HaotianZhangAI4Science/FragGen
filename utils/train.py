import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
import math
import os
import time
import logging
from torch import nn
import numpy as np
import torch
import random
from scipy.special import i0

from utils.geom_utils import rotate2x_axis, rotate_around_x_axis, batch_dihedrals

def get_model_loss(model, batch):
    compose_feature = batch['compose_feature'].float()
    compose_pos = batch['compose_pos'].to(torch.float32)
    idx_ligand = batch['idx_ligand_ctx_in_compose']
    idx_protein = batch['idx_protein_in_compose']
    # interaction
    compose_knn_edge_feature = batch['compose_knn_edge_feature']
    compose_knn_edge_index = batch['compose_knn_edge_index']
    # the first attach
    idx_protein_attch_mask = batch['idx_protein_attch_mask']
    # the position next possible growing point
    idx_focal = batch['focal_id_in_context']   # focal in context is 
    # the next atom/fragment
    pos_subpocket = batch['next_site_attach_pos']
    edge_index_q_cps_knn = batch['edge_new_site_knn_index']
    edge_index_q_cps_knn_batch = batch['edge_new_site_knn_index_batch']
    # the next attach point
    node_feat_frags = batch['node_feat_frags']
    edge_index_frags= batch['edge_index_frags']
    edge_features_frags = batch['edge_features_frags']
    current_wid = batch['current_wid']
    next_motif_wid = batch['next_motif_wid']
    node_batch_frags = batch['node_feat_frags_batch']
    # the bond prediction
    bonded_a_nei_edge_features = batch['bonded_a_nei_edge_features'].float()
    bonded_b_nei_edge_features = batch['bonded_b_nei_edge_features'].float()
    next_motif_bonded_atom_feature = batch['next_motif_bonded_atom_feature'].float()
    # the position predition
    compose_next_feature = batch['compose_next_feature'].float()
    compose_pos_next = batch['compose_with_next_pos']
    idx_ligand_next = batch['idx_ligand_ctx_next_in_compose']
    idx_protein_next = batch['idx_protein_in_compose_with_next']
    edge_feature_pos_pred = batch['compose_next_knn_edge_feature']
    edge_index_pos_pred = batch['compose_next_knn_edge_index']
    ligand_pos_mask_idx = batch['ligand_pos_mask_idx']
    a = batch['a']
    b = batch['b']
    
    ligand_idx = batch['idx_ligand_ctx_next_in_compose']
    b_next = batch['idx_ligand_ctx_next_in_compose'][batch['ligand_pos_mask']]
    batch_b_next = batch['ligand_pos_mask_batch'][batch['ligand_pos_mask']]
    batch_mol = batch['idx_ligand_ctx_next_in_compose_batch']

    y_protein_frontier_pred, y_frontier, abs_pos_mu, pos_sigma, pos_pi,\
        y_type_pred, frag_node_2d, bond_pred,alpha, updated_pos = model(
            compose_feature=compose_feature,
            compose_pos=compose_pos,
            idx_ligand=idx_ligand,
            idx_protein=idx_protein,
            compose_knn_edge_index=compose_knn_edge_index,
            compose_knn_edge_feature=compose_knn_edge_feature,
            idx_protein_attch_mask=idx_protein_attch_mask,
            idx_focal=idx_focal,
            pos_subpocket=pos_subpocket,
            edge_index_q_cps_knn=edge_index_q_cps_knn,
            edge_index_q_cps_knn_batch=edge_index_q_cps_knn_batch,
            node_feat_frags = node_feat_frags,
            edge_index_frags= edge_index_frags,
            edge_features_frags = edge_features_frags,
            current_wid=current_wid,
            next_motif_wid=next_motif_wid,
            node_batch_frags = node_batch_frags,
            bonded_a_nei_edge_features=bonded_a_nei_edge_features,
            bonded_b_nei_edge_features=bonded_b_nei_edge_features,
            next_motif_bonded_atom_feature=next_motif_bonded_atom_feature,
            compose_next_feature=compose_next_feature,
            compose_pos_next=compose_pos_next,
            idx_ligand_next = idx_ligand_next,
            idx_protein_next = idx_protein_next,
            edge_feature_pos_pred=edge_feature_pos_pred,
            edge_index_pos_pred=edge_index_pos_pred,
            a=a,
            b=b,
            ligand_idx = ligand_idx,
            b_next=b_next,
            batch_b_next=batch_b_next,
            batch_mol=batch_mol,
            ligand_pos_mask_idx=ligand_pos_mask_idx)
    
    loss_protein_frontier, loss_frontier, loss_cav, \
        loss_class, loss_nx_attch, loss_bond, loss_pos = \
            get_loss(y_protein_frontier_pred, y_frontier, abs_pos_mu, pos_sigma, pos_pi,\
                                        y_type_pred, frag_node_2d, bond_pred, alpha, updated_pos, batch, verbose=False)
    # torch.nan_to_num(loss_protein_frontier)
    loss = (torch.nan_to_num(loss_frontier)
            + torch.nan_to_num(loss_cav)
            + torch.nan_to_num(loss_class)
            + torch.nan_to_num(loss_nx_attch)
            + torch.nan_to_num(loss_bond)
            + torch.nan_to_num(loss_pos)
            )
    
    return loss, loss_protein_frontier, loss_frontier, loss_cav, loss_class, loss_nx_attch, loss_bond, loss_pos

def get_loss(y_protein_frontier_pred, y_frontier,
            abs_pos_mu, pos_sigma, pos_pi,
            y_type_pred,
            frag_node_2d,
            bond_pred,
            alpha, updated_pos,
            batch, verbose=False):
    device = y_protein_frontier_pred.device
    # frontier prediction
    loss_protein_frontier = F.binary_cross_entropy_with_logits(
                input=y_protein_frontier_pred,
                target=batch['y_protein_attach'].view(-1, 1).float()
            ).clamp_max(10.)
    context_index = torch.zeros(batch['ligand_context_element'].shape[0]).to(device)
    context_index[batch['focal_id_ligand']] = 1
    loss_frontier = F.binary_cross_entropy_with_logits(
        input = y_frontier,
        target = context_index.view(-1, 1).float()
    ).clamp_max(10.)

    # cavity detection
    next_cavity = batch['next_site_attach_pos']
    loss_cav = -torch.log(
        get_mdn_loss(abs_pos_mu, pos_sigma, pos_pi, next_cavity) + 1e-16
    ).mean().clamp_max(10.)

    # class prediction
    # smooth_cross_entropy = SmoothCrossEntropyLoss(reduction='mean', smoothing=0.1)
    # the one-hot of bath['next_motif_wid'] is done in the smooth_cross_entropy
    # loss_class = smooth_cross_entropy(y_type_pred,batch['next_motif_wid'])

    criterion = nn.CrossEntropyLoss()
    loss_class = criterion(y_type_pred, batch['next_motif_wid']).clamp_max(10.)
    # frag_attach prediction
    if torch.isnan(frag_node_2d).any():
        loss_nx_attch = torch.tensor(float('nan'))
    else:
        # next_attach = torch.arange(batch['next_motif_pos'].shape[0])==batch['next_site_attach'] #convert to the one-hot vector
        next_index = torch.zeros(batch['node_feat_frags'].shape[0]).to(device)
        next_index[batch['next_site_attach']] = 1
        loss_nx_attch = F.binary_cross_entropy_with_logits(
            input = frag_node_2d,
            target = next_index.view(-1, 1).float()
        ).clamp_max(10.)

    # bond prediction
    if torch.isnan(bond_pred).any():
        loss_bond = torch.tensor(float('nan'))
    else:
        loss_bond = F.cross_entropy(
            input = bond_pred,
            target =F.one_hot(batch['next_bond'].to(torch.long), num_classes=5).squeeze(1).float().to(device)
        ).clamp_max(10.)
        
    if updated_pos is not None:
        loss_pos = rmsd_loss(updated_pos[batch['ligand_pos_mask_idx']] - batch['compose_with_next_pos_target'][batch['ligand_pos_mask_idx']])
    elif alpha is not None:
        pred_sin, pred_cos = rotate_alpha_angle(alpha, batch)
        torsion_loss = (cossin_loss(batch['true_cos'], pred_cos.reshape(-1), batch['true_sin'], pred_cos.reshape(-1))[batch['dihedral_mask']]).mean().clamp_max(10.)
        loss_pos = torsion_loss
    else:
        loss_pos = torch.tensor(float('nan'))

    if verbose:
        print('loss_protein_frontier', loss_protein_frontier)
        print('loss_frontier', loss_frontier)
        print('loss_cav', loss_cav)
        print('loss_class', loss_class)
        print('loss_nx_attch', loss_nx_attch)
        print('loss_bond', loss_bond)
        print('loss_pos', loss_pos)

    return loss_protein_frontier, loss_frontier, loss_cav, loss_class, loss_nx_attch, loss_bond, loss_pos

def rotate_alpha_angle(alpha, batch, verbose=False):
    # loss_pos = rmsd_loss(updated_pos[batch['ligand_pos_mask_idx']] - batch['compose_with_next_pos_target'][batch['ligand_pos_mask_idx']])
    Hx = rotate2x_axis(batch['y_pos']) #y_pos is in x_axis
    xn_pos = torch.matmul(Hx, batch['xn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
    yn_pos = torch.matmul(Hx, batch['yn_pos'].permute(0, 2, 1)).permute(0, 2, 1)
    y_pos = torch.matmul(Hx, batch['y_pos'].unsqueeze(1).permute(0, 2, 1)).squeeze(-1)
    R_alpha = rotate_around_x_axis(torch.sin(alpha).squeeze(-1), torch.cos(alpha).squeeze(-1))
    xn_pos = torch.matmul(R_alpha, xn_pos.permute(0, 2, 1)).permute(0, 2, 1)

    p_idx, q_idx = torch.cartesian_prod(torch.arange(3), torch.arange(3)).chunk(2, dim=-1)
    p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
    pred_sin, pred_cos = batch_dihedrals(xn_pos[:, p_idx],
                                            torch.zeros_like(y_pos).unsqueeze(1).repeat(1, 9, 1),
                                            y_pos.unsqueeze(1).repeat(1, 9, 1),
                                            yn_pos[:, q_idx])
    return pred_sin, pred_cos

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss
    
def rmsd_loss(tensor):
    return torch.sqrt(torch.mean(torch.square(tensor)))

def get_mdn_loss(mu, sigma, pi, pos_target):
    prob_gauss = _get_gaussian_probability(mu, sigma, pos_target)
    prob_mdn = pi * prob_gauss
    prob_mdn = torch.sum(prob_mdn, dim=1)
    return prob_mdn

def _get_gaussian_probability(mu, sigma, pos_target):
    """
    mu - (N, n_component, 3)
    sigma - (N, n_component, 3)
    pos_target - (N, 3)
    """
    GAUSSIAN_COEF = 1.0 / math.sqrt(2 * math.pi)
    target = pos_target.unsqueeze(1).expand_as(mu)
    errors = target - mu
    sigma = sigma + 1e-16
    p = GAUSSIAN_COEF * torch.exp(- 0.5 * (errors / sigma)**2) / sigma
    p = torch.prod(p, dim=2)
    return p # (N, n_component)

def get_mdn_probability(mu, sigma, pi, pos_target):
    prob_gauss = _get_gaussian_probability(mu, sigma, pos_target)
    prob_mdn = pi * prob_gauss
    prob_mdn = torch.sum(prob_mdn, dim=1)
    return prob_mdn

def cossin_loss_from_angle(predicted_angle, reference_angle):
    """
    negative similarity of two angles
    """
    a_cos = torch.cos(predicted_angle)
    b_cos = torch.cos(reference_angle)
    a_sin = torch.sin(predicted_angle)
    b_sin = torch.sin(reference_angle)
    sim = a_cos * b_cos + a_sin * b_sin
    return -sim

def cossin_loss(a_cos, b_cos, a_sin=None, b_sin=None):
    """
    :param a_cos: cos of first angle
    :param b_cos: cos of second angle
    :return: difference of cosines
    """
    if torch.is_tensor(a_sin):
        angle_sim = a_cos * b_cos + a_sin * b_sin
    else:
        angle_sim = a_cos * b_cos + torch.sqrt(1-a_cos**2 + 1e-5) * torch.sqrt(1-a_cos**2 + 1e-5)
    return -angle_sim

def von_mises_loss(predicted_angle, reference_angle, kappa):
    """
    Compute the von Mises loss.
    L(\theta, \mu, \kappa) = -\log \left( \frac{e^{\kappa \cos(\theta - \mu)}}{2\pi I_0(\kappa)} \right)
    
    Parameters:
    - predict_angle: reference angles (in radiansed)
    - reference_angle: reference angles (in radians)
    - kappa: concentration parameter, a higher value for kappa means a more peaked distribution (hyperparameter)
    - I_0: modified Bessel function of order 0

    Returns:
    - Loss value
    """
    nll = - (kappa * torch.cos(predicted_angle - reference_angle) - torch.log(2 * torch.pi * i0(kappa)))
    return nll

def filter_fragment_mask(next_frag_batch):
    unique_elements, counts = torch.unique(next_frag_batch, return_counts=True)
    elements_with_count_gt_one = unique_elements[counts > 1]
    
    mask = torch.tensor([item in elements_with_count_gt_one for item in next_frag_batch])
    
    return elements_with_count_gt_one, mask

def normalize_angle(angle):
    return angle % (2 * np.pi) 


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr
        )
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)

# Below the logger part
class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self

def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    # Dihedral loss example
    theta = torch.tensor([1.5, 0.5])  # observed angles
    mu = torch.tensor([1.6, 0.4])     # mean angles
    kappa = torch.tensor([2.0])       # concentration parameter

    loss = von_mises_loss(theta, mu, kappa)
    print(loss)

    # Case 1: a and b are the same
    a = torch.tensor([0.5])
    b = torch.tensor([0.5])
    print("Case 1:", von_mises_loss(a, b, kappa).item())

    # Case 2: a and b have a small difference
    a = torch.tensor([0.5])
    b = torch.tensor([0.6])
    print("Case 2:", von_mises_loss(a, b, kappa).item())

    # Case 3: a and b are opposite
    a = torch.tensor([0.5])
    b = torch.tensor([-0.5])
    print("Case 3:", von_mises_loss(a, b, kappa).item())

    # Case 4: a and b have a larger difference (but not opposite)
    a = torch.tensor([0.5])
    b = torch.tensor([0.9])
    print("Case 4:", von_mises_loss(a, b, kappa).item())

    # Case 5: a and b are the same, but using sin values directly
    a = torch.tensor([0.5])
    b = torch.tensor([0.5])
    a_sin = torch.tensor([0.8660254])  # sin(60 degrees)
    b_sin = torch.tensor([0.8660254])
    print("Case 5:", von_mises_loss(a, b, kappa).item())


    # y_protein_frontier_pred, y_frontier, abs_pos_mu, pos_sigma, pos_pi,\
    #     y_type_pred, frag_node_2d, bond_pred,alpha = model(
        
    #     compose_feature = batch['compose_feature'].float(),
    #     compose_pos = batch['compose_pos'].to(torch.float32),
    #     idx_ligand = batch['idx_ligand_ctx_in_compose'],
    #     idx_protein = batch['idx_protein_in_compose'],
    #     compose_knn_edge_index = batch['compose_knn_edge_feature'],
    #     compose_knn_edge_feature = batch['compose_knn_edge_index'],
    #     idx_protein_attch_mask = batch['idx_protein_attch_mask'],
    #     ligand_context_pos_batch = batch['ligand_context_pos_batch'],
    #     idx_focal = batch['focal_id_in_context'],
    #     pos_subpocket = batch['next_site_attach_pos'],
    #     edge_index_q_cps_knn = batch['edge_new_site_knn_index'],
    #     edge_index_q_cps_knn_batch = batch['edge_new_site_knn_index_batch'],
    #     node_feat_frags = batch['node_feat_frags'],
    #     edge_index_frags = batch['edge_index_frags'],
    #     edge_features_frags = batch['edge_features_frags'],
    #     current_wid = batch['current_wid'],
    #     next_motif_wid = batch['next_motif_wid'],
    #     node_batch_frags = batch['node_feat_frags_batch'],
    #     context_next_node_feature = batch['ligand_context_next_feature_full'].float(),
    #     context_node_feature = batch['ligand_context_feature_full'],
    #     ligand_context_pos = batch['ligand_context_pos'],
    #     bonded_a_nei_edge_features = batch['bonded_a_nei_edge_features'].float(),
    #     bonded_b_nei_edge_features = batch['bonded_b_nei_edge_features'].float(),
    #     query_feature = batch['next_motif_bonded_atom_feature'].float(),
    #     compose_next_feature = batch['compose_next_feature'].float(),
    #     compose_pos_next = batch['compose_with_next_pos'],
    #     idx_ligand_next = batch['idx_ligand_ctx_next_in_compose'],
    #     idx_protein_next = batch['idx_protein_in_compose_with_next'],
    #     edge_feature_pos_pred = batch['compose_next_knn_edge_feature'],
    #     edge_index_pos_pred = batch['compose_next_knn_edge_index'],
    #     a = batch['a'],
    #     b = batch['b'],
    #     ligand_idx = batch['idx_ligand_ctx_next_in_compose'],
    #     b_next = batch['idx_ligand_ctx_next_in_compose'][batch['ligand_pos_mask']],
    #     batch_b_next = batch['ligand_pos_mask_batch'][batch['ligand_pos_mask']],
    #     batch_mol = batch['idx_ligand_ctx_next_in_compose_batch']
    #     )
    
    # loss_protein_frontier, loss_frontier, loss_cav, \
    #     loss_class, loss_nx_attch, loss_bond, loss_pos = \
    #         get_loss(y_protein_frontier_pred, y_frontier, abs_pos_mu, pos_sigma, pos_pi,\
    #                                     y_type_pred, frag_node_2d, bond_pred, alpha, batch, verbose=False)