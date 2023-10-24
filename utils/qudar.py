import torch
import torch.nn as nn

class QuaternionPredictor(nn.Module):
    def __init__(self, input_dim):
        super(QuaternionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 4)  # Output is 4D quaternion

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)  # Normalize to unit length
        return x
        
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from pyquaternion import Quaternion

# Load your molecule
mol = Chem.MolFromSmiles('CCO')
# Get conformer
AllChem.EmbedMolecule(mol)
conf = mol.GetConformer()


# Define the atoms in the fragment you want to rotate
# Here we'll just rotate the whole molecule
rotatable_atoms = list(range(mol.GetNumAtoms()))

# Choose a point to rotate around - here we'll use the first atom in the fragment
rotation_point = np.array(conf.GetAtomPosition(rotatable_atoms[0]))

# Generate a random rotation axis
phi = np.random.uniform(0, np.pi*2)
theta = np.random.uniform(0, np.pi)
rotation_axis = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

# Generate a random rotation angle
rotation_angle = np.random.uniform(0, np.pi*2)

# Convert to a quaternion
rotation_quaternion = Quaternion(axis=rotation_axis, angle=rotation_angle)

# Perform the rotation
for atom_idx in rotatable_atoms:
    # Get current atom position
    atom_pos = np.array(conf.GetAtomPosition(atom_idx))
    # Translate to origin
    atom_pos -= rotation_point
    # Convert to quaternion
    atom_pos_quat = Quaternion(0, atom_pos[0], atom_pos[1], atom_pos[2])
    # Rotate and convert back to Cartesian coordinates
    atom_pos = (rotation_quaternion * atom_pos_quat * rotation_quaternion.conjugate).vector
    # Translate back
    atom_pos += rotation_point
    # Update atom position
    conf.SetAtomPosition(atom_idx, atom_pos)
