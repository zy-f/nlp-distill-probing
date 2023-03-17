from transformers import AutoModelForSequenceClassification
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import current_pst_time
import json
from tqdm import tqdm
import os

# === DATASETS ===
class DistilBertExtractionDataset(Dataset):
    def __init__(self, tok_dset):
        self.input_ids = tok_dset['input_ids']
        self.attention_mask = tok_dset['attention_mask']
        self.pos_labels = tok_dset['pos_labels']
        self.dep_labels = tok_dset['dep_labels']
        self.label_endpoints = tok_dset['label_endpoints']

    def __len__(self):
        return len(self.label_endpoints)

    def __getitem__(self, idx):
        inp = torch.LongTensor(self.input_ids[idx])
        am = torch.FloatTensor(self.attention_mask[idx])
        start, end = self.label_endpoints[idx]
        pos_label = torch.LongTensor([-1]*start + self.pos_labels[idx] + [-1]*(len(am) - end))
        dep_label = torch.LongTensor([-1]*start + self.dep_labels[idx] + [-1]*(len(am) - end))
        lbl_ep = torch.IntTensor(self.label_endpoints[idx])
        bert_inp = {'input_ids': inp, 'attention_mask': am}
        return bert_inp, (pos_label, dep_label, lbl_ep)

class BertExtractionDataset(DistilBertExtractionDataset):
    '''
    It's literally the same except add token type IDs
    '''
    def __init__(self, tok_dset):
        super().__init__(tok_dset)
        self.token_type_ids = tok_dset['token_type_ids']

    def __getitem__(self, idx):
        bert_inp, bert_out = super().__getitem__(idx)
        bert_inp['token_type_ids'] = torch.LongTensor(self.token_type_ids[idx])
        return bert_inp, bert_out

def main():
    # === CONSTANTS (can turn into command-line args later) ===
    TOKEN_TYPE = 'bert_subwordtok_redo'
    DATASET_CLASS = BertExtractionDataset
    MODEL_TYPE = 'bert-base-uncased'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels=3)
    MODEL_WEIGHTS = None#'checkpoints/bert_finetune_initial-2023_02_28-14_18.pth'
    bsz = 64
    device = 'cuda:0'
    SAVE_PATH = 'data/probe/bert_mlm_none-extracts.ptdata'
    if os.path.exists(SAVE_PATH):
        print(f'WARNING: {SAVE_PATH} already exists. Are you sure you want to overwrite it?')
        if not input('[Y/N] > ').lower().startswith('y'):
            raise OSError('Path already exists, not overwriting.')
    # === END CONSTANTS ===

    # === SCRIPT ===
    if MODEL_TYPE.startswith('distilbert-base'):
        layers_to_hook = [
            (model.distilbert.embeddings.LayerNorm, 'layer0'),
            (model.distilbert.transformer.layer[0].output_layer_norm, 'layer1'),
            (model.distilbert.transformer.layer[3].output_layer_norm, 'layer4'),
            (model.distilbert.transformer.layer[4].output_layer_norm, 'layer5'),
            (model.distilbert.transformer.layer[5].output_layer_norm, 'layer6')
        ]
    elif MODEL_TYPE.startswith('bert-base'):
        layers_to_hook = [
            (model.bert.embeddings.LayerNorm, 'layer0'),
            (model.bert.encoder.layer[0].output.LayerNorm, 'layer1'),
            # (model.bert.encoder.layer[3].output.LayerNorm, 'layer4'),
            # (model.bert.encoder.layer[4].output.LayerNorm, 'layer5'),
            (model.bert.encoder.layer[5].output.LayerNorm, 'layer6'),
            (model.bert.encoder.layer[9].output.LayerNorm, 'layer10'),
            (model.bert.encoder.layer[10].output.LayerNorm, 'layer11'),
            (model.bert.encoder.layer[11].output.LayerNorm, 'layer12')
        ]

    with open(f'data/UD_English-EWT/en_ewt-ud-train-{TOKEN_TYPE}.json', 'r') as f:
        train_tok = json.load(f)

    with open(f'data/UD_English-EWT/en_ewt-ud-dev-{TOKEN_TYPE}.json', 'r') as f:
        val_tok = json.load(f)
    train_dset = DATASET_CLASS(train_tok)
    val_dset = DATASET_CLASS(val_tok)
    train_dl = DataLoader(train_dset, batch_size=bsz, shuffle=True)
    val_dl = DataLoader(val_dset, batch_size=bsz, shuffle=False)

    if MODEL_WEIGHTS is not None:
        model.load_state_dict(torch.load(MODEL_WEIGHTS))

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().cpu()
        return hook

    for layer, name in layers_to_hook:
        layer.register_forward_hook(get_activation(name))

    model.eval()
    model.to(device)
    final_extracts = {}
    for dl_name, dl in [('train', train_dl), ('val', val_dl)]:
        extracts = {name:[] for _, name in layers_to_hook}
        extracts['pos_labels'] = []
        extracts['dep_labels'] = []
        with torch.no_grad():
            for bert_inp, (pos_lbl, dep_lbl, lbl_ep) in tqdm(dl, desc=dl_name):
                bert_inp = {name: t.to(device) for name, t in bert_inp.items()}
                model(**bert_inp)
                for name, layer_output in activation.items():
                    for i in range(len(lbl_ep)):
                        start_idx = lbl_ep[i,0]
                        end_idx = lbl_ep[i,1]
                        extracts[name].append(layer_output[i,start_idx:end_idx,:])
                for i in range(len(lbl_ep)):
                    start_idx = lbl_ep[i,0]
                    end_idx = lbl_ep[i,1]
                    extracts['pos_labels'].append(pos_lbl[i,start_idx:end_idx])
                    extracts['dep_labels'].append(dep_lbl[i,start_idx:end_idx])
        flattened_extracts = {k: torch.cat(tensor_list, dim=0) for k,tensor_list in extracts.items()}
        final_extracts[dl_name] = flattened_extracts
    torch.save(final_extracts, SAVE_PATH)
    print(f'Extracts saved to {SAVE_PATH}')

if __name__ == '__main__':
    main()
