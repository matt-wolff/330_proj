import torch
import esm
import torch.nn as nn

# sequences: list of protein sequences

class ESMEmbedder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

    def representations(self, data):
        self.model.eval()

        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(self.device)
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12], return_contacts=False)
        token_representations = results["representations"][12]

        return token_representations, batch_lens

    def Embeddings2D(self, sequences):
        data = [(f'protein{k}', sequences[k]) for k in range(len(sequences))]
        token_representations, batch_lens = self.representations(data)
        SeqReps2D = []
        for i, tokens_len in enumerate(batch_lens):
            SeqReps2D.append(token_representations[i, 1 : tokens_len - 1])
        return SeqReps2D

    def Embeddings1D(self, sequences):
        data = [(f'protein{k}', sequences[k]) for k in range(len(sequences))]
        token_representations, batch_lens = self.representations(data)
        SeqReps1D = []
        for i, tokens_len in enumerate(batch_lens):
            SeqReps1D.append(token_representations[i, 1 : tokens_len - 1].mean(0))
        return SeqReps1D
