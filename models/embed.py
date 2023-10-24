from torch.nn import Module, Linear
import torch

class AtomEmbedding(Module):
    def __init__(self, in_scalar, in_vector,
                 out_scalar, out_vector, vector_normalizer=20.):
        super().__init__()
        assert in_vector == 1
        self.in_scalar = in_scalar
        self.vector_normalizer = vector_normalizer
        self.emb_sca = Linear(in_scalar, out_scalar)
        self.emb_vec = Linear(in_vector, out_vector)

    def forward(self, scalar_input, vector_input):
        vector_input = vector_input / self.vector_normalizer
        assert vector_input.shape[1:] == (3, ), 'Not support. Only one vector can be input'
        sca_emb = self.emb_sca(scalar_input[:, :self.in_scalar])  # b, f -> b, f'
        vec_emb = vector_input.unsqueeze(-1)  # b, 3 -> b, 3, 1
        vec_emb = self.emb_vec(vec_emb).transpose(1, -1)  # b, 1, 3 -> b, f', 3
        return sca_emb, vec_emb

def embed_compose(compose_feature, compose_pos, idx_ligand, idx_protein,
                                      ligand_atom_emb, protein_atom_emb,
                                      emb_dim):

    h_ligand = ligand_atom_emb(compose_feature[idx_ligand], compose_pos[idx_ligand])
    h_protein = protein_atom_emb(compose_feature[idx_protein], compose_pos[idx_protein])
    
    h_sca = torch.zeros([len(compose_pos), emb_dim[0]],).to(h_ligand[0])
    h_vec = torch.zeros([len(compose_pos), emb_dim[1], 3],).to(h_ligand[1])
    h_sca[idx_ligand], h_sca[idx_protein] = h_ligand[0], h_protein[0]
    h_vec[idx_ligand], h_vec[idx_protein] = h_ligand[1], h_protein[1]
    return [h_sca, h_vec]
