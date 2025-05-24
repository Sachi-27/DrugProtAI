import os 
import json
import pandas as pd
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from transformers import T5EncoderModel, T5Tokenizer
from transformers import EsmTokenizer, EsmModel
import torch 
from tqdm import tqdm
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer

def get_esm_model():
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", do_lower_case=False)

    return model, tokenizer

def extract_seq_data():
    with open("protein_props.json") as f:
        protein_data = json.load(f)
    protein_data = {k:v["Sequence"] for k,v in protein_data.items()}
    return protein_data

MODEL_TO_CONSIDER = "ProtTrans" # Use either "ESM" or "ProtTrans"


if MODEL_TO_CONSIDER == "ESM": model, tokenizer = get_esm_model()
elif MODEL_TO_CONSIDER == "ProtTrans": model, tokenizer = get_T5_model()
else: raise ValueError("No such model implemented")

protein_sequences = extract_seq_data()

# Save protein embeddings into a csv file, with protein_id as index and embedding as columns
embeddings = []
protein_ids_considered = []
for protein_id, protein_sequence in tqdm(protein_sequences.items()):
    # print(f"Processing {protein_id}...", f"completed {len(embeddings)} out of {len(protein_sequences)}")
    if len(protein_sequence) > 3000: continue # skip long sequences
    token_encoding = tokenizer(protein_sequence, return_tensors="pt", padding="longest")
    input_ids = token_encoding["input_ids"].to(device)
    attention_mask = token_encoding["attention_mask"].to(device)
    if MODEL_TO_CONSIDER == "ProtTrans":
        # Global pooling across protein residues
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, output_hidden_states=True)
        emb = embedding_repr.last_hidden_state[0] # N x 1024 where N is number of residues
        avg_pool_emb = torch.mean(emb, dim=0) # 1024
        # embeddings.append(avg_pool_emb.cpu().detach().numpy())
        embeddings.append(emb[-1].cpu().numpy()) # Use the last token's embedding as a global representation

        del embedding_repr, emb, avg_pool_emb
    elif MODEL_TO_CONSIDER == "ESM":
        # Use ESM’s CLS token (index 0) embedding instead of average pooling. That's the intended global representation.
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # CLS token is at position 0
            cls_embedding = outputs.last_hidden_state[0, 0, :]  # Shape: (1280,)
        embeddings.append(cls_embedding.cpu().numpy())

        del outputs, cls_embedding

    protein_ids_considered.append(protein_id)

    # Clean up GPU memory
    del token_encoding, input_ids, attention_mask
    torch.cuda.empty_cache()


# Save the embeddings to a csv file
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.index = protein_ids_considered
embeddings_df.to_csv("protein_embeddings_ProtTransT5.csv")
