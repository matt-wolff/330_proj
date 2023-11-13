import torch
import esm

# sequences: list of protein sequences
def representations(data):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    return token_representations, batch_lens

def Embeddings2D(sequences):
    data = {f'protein{k}':sequences[k] for k in range(len(sequences))}
    token_representations, batch_lens = representations(data)
    SeqReps2D = []
    for i, tokens_len in enumerate(batch_lens):
        SeqReps2D.append(token_representations[i, 1 : tokens_len - 1])
    return SeqReps2D

def Embeddings1D(sequences):
    data = {f'protein{k}':sequences[k] for k in range(len(sequences))}
    token_representations, batch_lens = representations(data)
    SeqReps1D = []
    for i, tokens_len in enumerate(batch_lens):
        SeqReps1D.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    return SeqReps1D