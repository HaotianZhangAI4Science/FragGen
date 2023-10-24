from rdkit import Chem
from rdkit.Chem import AllChem
from copy import deepcopy
import numpy as np
import torch
from scipy.spatial.transform import Rotation 
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdmt

def batch_dihedrals(p0, p1, p2, p3, angle=False):
    '''
    Compute the dihedral angle between two planes defined by four points, i.e., plane 1 is defined by p0, p1, p2 and plane 2 is defined by p1, p2, p3.W
    Returns:
        sin_dihedral, cos_dihedral
        or
        dihedral angle in radians
    '''
    s1 = p1 - p0
    s2 = p2 - p1
    s3 = p3 - p2

    sin_d_ = torch.linalg.norm(s2, dim=-1) * torch.sum(s1 * torch.cross(s2, s3, dim=-1), dim=-1)
    cos_d_ = torch.sum(torch.cross(s1, s2, dim=-1) * torch.cross(s2, s3, dim=-1), dim=-1)

    if angle:
        return torch.atan2(sin_d_, cos_d_ + 1e-10)

    else:
        den = torch.linalg.norm(torch.cross(s1, s2, dim=-1), dim=-1) * torch.linalg.norm(torch.cross(s2, s3, dim=-1), dim=-1) + 1e-10
        return sin_d_/den, cos_d_/den

def rand_rotate_around_axis(vec, ref, pos, alpha=None):
    #vec = vec/torch.norm(vec)
    if alpha is None:
        alpha = torch.randn(1)
    n_pos = pos.shape[0]
    sin, cos = torch.sin(alpha), torch.cos(alpha)
    K = 1 - cos
    M = torch.dot(vec, ref)
    nx, ny, nz = vec[0], vec[1], vec[2]
    x0, y0, z0 = ref[0], ref[1], ref[2]
    T = torch.tensor([nx ** 2 * K + cos, nx * ny * K - nz * sin, nx * nz * K + ny * sin,
         (x0 - nx * M) * K + (nz * y0 - ny * z0) * sin,
         nx * ny * K + nz * sin, ny ** 2 * K + cos, ny * nz * K - nx * sin,
         (y0 - ny * M) * K + (nx * z0 - nz * x0) * sin,
         nx * nz * K - ny * sin, ny * nz * K + nx * sin, nz ** 2 * K + cos,
         (z0 - nz * M) * K + (ny * x0 - nx * y0) * sin,
         0, 0, 0, 1]).reshape(4, 4)
    pos = torch.cat([pos.t(), torch.ones(n_pos).unsqueeze(0)], dim=0)
    rotated_pos = torch.mm(T, pos)[:3]
    return rotated_pos.t()

def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = deepcopy(rdkit_mol)
    set_rdmol_positions_(mol, pos)
    return mol


def set_rdmol_positions_(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol

def rotate_matrix_around_axis(axis, angle):
    """
    Compute rotation matrices for a given axis and angle.
    :param axis: torch.Tensor of shape [1, 3] (or [batch_size, 3] if each rotation has a different axis)
    :param angle: Rotation angle in radians, scalar
    :return: Rotation matrices, torch.Tensor of shape [batch_size, 3, 3]
    
    \[
    R(\mathbf{u}, \theta) = 
    \begin{bmatrix}
    \cos(\theta) + u_x^2(1-\cos(\theta)) & u_x u_y (1-\cos(\theta)) - u_z \sin(\theta) & u_x u_z (1-\cos(\theta)) + u_y \sin(\theta) \\
    u_y u_x (1-\cos(\theta)) + u_z \sin(\theta) & \cos(\theta) + u_y^2(1-\cos(\theta)) & u_y u_z (1-\cos(\theta)) - u_x \sin(\theta) \\
    u_z u_x (1-\cos(\theta)) - u_y \sin(\theta) & u_z u_y (1-\cos(\theta)) + u_x \sin(\theta) & \cos(\theta) + u_z^2(1-\cos(\theta))
    \end{bmatrix}
    \]
    """
    
    # Normalize the axis
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    
    # Create the rotation matrix
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    outer_product = torch.ger(axis.squeeze(), axis.squeeze())
    cross_product_matrix = torch.tensor([
        [0, -axis[0, 2], axis[0, 1]],
        [axis[0, 2], 0, -axis[0, 0]],
        [-axis[0, 1], axis[0, 0], 0]
    ]).to(axis.device)
    
    R = cos_angle * torch.eye(3).to(axis.device) + (1 - cos_angle) * outer_product - sin_angle * cross_product_matrix
    
    return R

def rotate_axis_w_centered_point(R, centered, points):
    """
    Input:
        R matric represents the rotation matrix, typically representing the rotation around an axis with an angle
        centered is the center point of the rotation
        points is the set of points to be rotated
        
    Output:
        rotated_points
    """
    points = points - centered
    points_t = points.T
    rotated_points_t = torch.mm(R, points_t)
    rotated_points = rotated_points_t.T

    return rotated_points + centered

def rotate_batch_matrix_around_axis(axis, angle):
    """
    Compute rotation matrices for a given axis and angle for batches.
    :param axis: torch.Tensor of shape [batch_size, 3]
    :param angle: Rotation angle in radians, torch.Tensor of shape [batch_size]
    :return: Rotation matrices, torch.Tensor of shape [batch_size, 3, 3]
    """
    batch_size = axis.shape[0]
    
    # Normalize the axis
    axis = axis / torch.norm(axis, dim=-1, keepdim=True)
    
    cos_angle = torch.cos(angle).view(batch_size, 1, 1)
    sin_angle = torch.sin(angle).view(batch_size, 1, 1)
    
    identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(axis.device)
    
    outer_product = torch.bmm(axis.unsqueeze(-1), axis.unsqueeze(1))
    
    cross_product_matrix = torch.stack([
        torch.zeros(batch_size, device=axis.device),
        -axis[:, 2], 
        axis[:, 1],
        axis[:, 2], 
        torch.zeros(batch_size, device=axis.device), 
        -axis[:, 0],
        -axis[:, 1], 
        axis[:, 0], 
        torch.zeros(batch_size, device=axis.device)
    ], dim=1).view(batch_size, 3, 3)
    
    R = cos_angle * identity + (1 - cos_angle) * outer_product - sin_angle * cross_product_matrix
    return R

def batched_rotate_around_center(rotate_matrix, centered, positions, batch_trace):
    rotated_positions = []

    for pos, idx in zip(positions, batch_trace):
        centered_pos = pos - centered[idx]
        rotated_pos = torch.matmul(rotate_matrix[idx], centered_pos) + centered[idx]
        rotated_positions.append(rotated_pos)

    return torch.stack(rotated_positions)

def rotate2x_axis(neighbor_coords):
    """
    Given predicted neighbor coordinates from model, return rotation matrix
    :param neighbor_coords: y or x coordinates for the x or y center node
        (n_dihedral_pairs, 3)
    :return: rotation matrix (n_dihedral_pairs, 3, 3)
    # h1 is along the py , h2 and h3 represent the perpendicular plane
    """

    p_Y = neighbor_coords

    eta_1 = torch.rand_like(p_Y)
    eta_2 = eta_1 - torch.sum(eta_1 * p_Y, dim=-1, keepdim=True) / (torch.linalg.norm(p_Y, dim=-1, keepdim=True)**2 + 1e-10) * p_Y
    eta = eta_2 / torch.linalg.norm(eta_2, dim=-1, keepdim=True)

    h1 = p_Y / (torch.linalg.norm(p_Y, dim=-1, keepdim=True) + 1e-10)  # (n_dihedral_pairs, n_model_confs, 10)

    h3_1 = torch.cross(p_Y, eta, dim=-1)
    h3 = h3_1 / (torch.linalg.norm(h3_1, dim=-1, keepdim=True) + 1e-10)  # (n_dihedral_pairs, n_model_confs, 10)

    h2 = -torch.cross(h1, h3, dim=-1)  # (n_dihedral_pairs, n_model_confs, 10)

    H = torch.cat([h1.unsqueeze(-2),
                   h2.unsqueeze(-2),
                   h3.unsqueeze(-2)], dim=-2)

    return H

def rotate_around_x_axis(alpha, alpha_cos=None):
    """
    Builds the alpha rotation matrix

    :param alpha: predicted values of torsion parameter alpha (n_dihedral_pairs)
    :return: alpha rotation matrix (n_dihedral_pairs, 3, 3)
    """
    H_alpha = torch.FloatTensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]).repeat(alpha.shape[0], 1, 1).to(alpha.device)

    if torch.is_tensor(alpha_cos):
        H_alpha[:, 1, 1] = alpha_cos
        H_alpha[:, 1, 2] = -alpha
        H_alpha[:, 2, 1] = alpha
        H_alpha[:, 2, 2] = alpha_cos
    else:
        H_alpha[:, 1, 1] = torch.cos(alpha)
        H_alpha[:, 1, 2] = -torch.sin(alpha)
        H_alpha[:, 2, 1] = torch.sin(alpha)
        H_alpha[:, 2, 2] = torch.cos(alpha)

    return H_alpha

def trans_frag(mol, frag, attch_point_mol, attch_point_frag, vector, return_combine=False):
    '''
    Suppose: Attachment point of mol is r_m; Attachment point of frag is r_f
    vector = rf - r_m  
    \tilde{r_f} = r_m + vector
    r_trans = \tilde{r_f} - r_m
    '''
    next_frag = deepcopy(frag)
    mol_attach_pos = np.array(mol.GetConformer().GetAtomPosition(attch_point_mol))
    frag_attach_pos = np.array(frag.GetConformer().GetAtomPosition(attch_point_frag))
    target_attch_frag = mol_attach_pos + vector
    next_frag_pos = next_frag.GetConformer().GetPositions() + target_attch_frag - next_frag.GetConformer().GetAtomPosition(attch_point_frag)
    next_frag_trans = set_rdmol_positions(next_frag,next_frag_pos)
    if return_combine:
        return Chem.CombineMols(mol, next_frag_trans)
    else:
        return next_frag_trans
    
def bond_mols(mol, frag, attch_point_mol, attch_point_frag, bond_type=Chem.rdchem.BondType.SINGLE):

    # Create an editable version of mol1
    combine_mol = Chem.CombineMols(mol, frag)

    # Record the number of atoms in mol1 for later use
    mol_n_atoms = mol.GetNumAtoms()
    combine_mol_edit = Chem.EditableMol(combine_mol)
    combine_mol_edit.AddBond(attch_point_mol, mol_n_atoms + attch_point_frag, order=bond_type)
    bonded_mol = combine_mol_edit.GetMol()

    # Return the new molecule
    return bonded_mol



def rotate_matrix(theta=None):
    if theta is None:
        theta = np.random.rand(3) * 2 * np.pi  # Random rotation angles
    rot_matrix = np.array([
        [np.cos(theta[0]), -np.sin(theta[0]), 0],
        [np.sin(theta[0]), np.cos(theta[0]), 0],
        [0, 0, 1]
    ])
    return rot_matrix

def rotate_fragment(mol, center, r_matrix):
    conf = mol.GetConformer()
    center = np.array(conf.GetAtomPosition(center))
    for atom_idx in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(atom_idx)) - center
        new_pos = np.dot(r_matrix, pos) + center
        conf.SetAtomPosition(atom_idx, new_pos)
    return mol

def uff_constraint(mol,lig_constraint,n_iters=200,n_tries=2):

    if type(lig_constraint) == Chem.rdchem.Mol:
        lig_constraint = mol.GetSubstructMatch(lig_constraint)
    else:
        lig_constraint = lig_constraint
    uff = AllChem.UFFGetMoleculeForceField(mol, confId=0, ignoreInterfragInteractions=False)
    uff.Initialize()
    for i in lig_constraint:
        uff.AddFixedPoint(i)
    converged = False
    while n_tries > 0 and not converged:
        converged = not uff.Minimize(maxIts=n_iters)
        n_tries -= 1
    return mol

def uffopt_rigidpkt(rd_mol,pkt_mol,lig_constraint=None,n_iters=200,n_tries=2, lig_h=True, pkt_h=False):
    '''
    A function to perform UFF optimization with the binding site fixed, it consums about 1s per pocket-ligand pair (10A)
    Input:
        rd_mol: the receptor molecule
        pkt_mol: the pocket molecule. For instance, pkt_mol = Chem.MolFromPDBFile('1a07_ligand.pdb')
        lig_constraint: the indices of the ligand atoms to be fixed
        n_iters: the number of iterations
        n_tries: the number of tries
        lig_h: whether to add Hs to the ligand
        pkt_h: whether to add Hs to the pocket
    '''
    if lig_h:
        rd_mol = Chem.AddHs(rd_mol,addCoords=True) # hydrogen atoms are appended to the end of the molecule defaultly
    if pkt_h:
        pkt_mol = Chem.AddHs(pkt_mol,addCoords=True)

    rd_mol = Chem.RWMol(rd_mol)
    uff_mol = Chem.CombineMols(pkt_mol, rd_mol) # Combine the pkt and lig, the ligand is appended to the end of the complex

    try:
        Chem.SanitizeMol(uff_mol)
    except Chem.AtomValenceException:
        print('Invalid valence')
    except (Chem.AtomKekulizeException, Chem.KekulizeException):
        print('Failed to kekulize')
    try:
        uff = AllChem.UFFGetMoleculeForceField(
                        uff_mol, confId=0, ignoreInterfragInteractions=False
                    )
        uff.Initialize()
        for i in range(pkt_mol.GetNumAtoms()): # Fix the rec atoms
            uff.AddFixedPoint(i)
        if lig_constraint is not None:
            for i in lig_constraint:
                uff.AddFixedPoint(pkt_mol.GetNumAtoms()+i) # Fix the specified lig atoms

        converged = False
        n_iters=n_iters
        n_tries=n_tries
        while n_tries > 0 and not converged:
            print('.', end='', flush=True)
            converged = not uff.Minimize(maxIts=n_iters)
            n_tries -= 1
        print(flush=True)
        print("Performed UFF with binding site...")
    except:
        print('Skip UFF...')

    frags = Chem.GetMolFrags(uff_mol, asMols=True)
    updated_rd_mol = frags[-1]
    return updated_rd_mol

def rot_mat(center=None,angle=None, axis=None):
    '''
    Create a 4 \times 4 rotation matrix, determined by the center, angle and axis
    Principly, it is more suitable to use from the uniformity consideration
    But I stick to the rotate_matrix_around_axis and its associated functions, which is clearer for freshmen. 
    I'm glad if someone come across this and give me some feedbacks. Have fun!
    '''
    if angle is None:
        rotation_angle = random.uniform(0, 2*np.pi)
    if axis is None:
        phi = np.random.uniform(0, np.pi*2)
        theta = np.random.uniform(0, np.pi)
        rotation_axis = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    # Define rotation matrix
    r = Rotation.from_rotvec(rotation_angle * rotation_axis)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = r.as_matrix()

    translation_matrix_to_origin = np.eye(4)
    translation_matrix_to_original = np.eye(4)
    if center is not None:
        # Define translation matrix to origin
        translation_matrix_to_origin[:3, 3] = -center
        # Define translation matrix back to the original position
        translation_matrix_to_original[:3, 3] = center

    total_transformation = translation_matrix_to_original @ rotation_matrix @ translation_matrix_to_origin
    return total_transformation

def normalize_angle(angle):
    return angle % (2 * np.pi) 

if __name__ == '__main__':
    from .chem import read_sdf
    old_existing = read_sdf('./mols/existing.sdf')[0]
    old_next_frag = read_sdf('./mols/next_frag.sdf')[0]
    next_frag = deepcopy(old_next_frag)
    existing = deepcopy(old_existing)
    next_frag.RemoveAllConformers()
    AllChem.EmbedMolecule(next_frag)

    existing_attach = 4
    next_frag_attach = 0
    existing_attach_pos = old_existing.GetConformer().GetAtomPosition(existing_attach)
    next_frag_attach_pos = old_next_frag.GetConformer().GetAtomPosition(next_frag_attach)

    # suppose we have predicted the bond vector
    noise = np.random.randn(1) * 0.1
    bond_vector = next_frag_attach_pos - existing_attach_pos
    pred_bond_vectot = bond_vector + noise

    # bond next fragment to the exisitng molecule
    next_frag_trans = trans_frag(old_existing, next_frag, existing_attach, next_frag_attach, pred_bond_vectot)
    bond_mols(existing, next_frag_trans, existing_attach, next_frag_attach)

    # suppose we have predicted the rotation angle
    r_matrix = rotate_matrix()
    next_frag_rotated = rotate_fragment(next_frag_trans, next_frag_attach, r_matrix)
    trans_frag(old_existing, next_frag_rotated, existing_attach, next_frag_attach, bond_vector, return_combine=True)



    ### another approach: use the FF to set the initial position of the next fragment
    r_matrix = rotate_matrix()
    frag_trans = rotate_fragment(next_frag_trans, next_frag_attach, r_matrix)
    mol_bond = bond_mols(existing, frag_trans, existing_attach, next_frag_attach)
    opt_mol = uff_constraint(mol_bond, existing)

    # Using EGNN to refine the next_frag geometries 
    
