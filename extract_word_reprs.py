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
        self.label_endpoints = tok_dset['label_endpoints']

    def __len__(self):
        return len(self.label_endpoints)

    def __getitem__(self, idx):
        inp = torch.LongTensor(self.input_ids[idx])
        am = torch.FloatTensor(self.attention_mask[idx])
        lbl_ep = torch.IntTensor(self.label_endpoints[idx])
        bert_inp = {'input_ids': inp, 'attention_mask': am}
        return bert_inp, lbl_ep

class BertExtractionDataset(DistilBertExtractionDataset):
    '''
    It's literally the same except add token type IDs
    '''
    def __init__(self, tok_dset):
        super().__init__(tok_dset)
        self.token_type_ids = tok_dset['token_type_ids']

    def __getitem__(self, idx):
        bert_inp, lbl_ep = super().__getitem__(idx)
        bert_inp['token_type_ids'] = torch.LongTensor(self.token_type_ids[idx])
        return bert_inp, lbl_ep

def main(\
    DATASET_CLASS = DistilBertExtractionDataset,
    DATASET_BASE = 'data/ontonotes/ontonotes',
    MODEL_TYPE = 'distilbert-base-uncased',
    MODEL_WEIGHTS = 'checkpoints/distilbert_distfinetune_strongDistLoss.pth',
    SAVE_BASE = 'data/probe/ontoV4/distilbert_mlm_mixedmnli',
    overwrite=False
    ):
    # === CONSTANTS (can turn into command-line args later) ===
    TOKEN_TYPE = 'bert_subwordtok_redo'
    bsz = 64
    device = 'cuda:0'
    # === END CONSTANTS ===

    # === SCRIPT ===
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels=3)
    
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
            (model.bert.encoder.layer[5].output.LayerNorm, 'layer6'),
            (model.bert.encoder.layer[9].output.LayerNorm, 'layer10'),
            (model.bert.encoder.layer[10].output.LayerNorm, 'layer11'),
            (model.bert.encoder.layer[11].output.LayerNorm, 'layer12')
        ]

    with open(f'{DATASET_BASE}-train-{TOKEN_TYPE}.json', 'r') as f:
        train_tok = json.load(f)

    with open(f'{DATASET_BASE}-val-{TOKEN_TYPE}.json', 'r') as f:
        val_tok = json.load(f)

    for key in train_tok.keys():
        if 'labels' in key:
            SAVE_PATH = f"{SAVE_BASE[:SAVE_BASE.rfind('/')]}/{key}.ptdata"
            if not os.path.exists(SAVE_PATH):
                label_data = {}
                label_data['train'] = torch.LongTensor([l for sent in train_tok[key] for l in sent])
                label_data['val'] = torch.LongTensor([l for sent in val_tok[key] for l in sent])
                torch.save(label_data, SAVE_PATH)
                print(f'{key} data saved to {SAVE_PATH}')
            else:
                print(f'{key} data already exists at {SAVE_PATH}')

    train_dset = DATASET_CLASS(train_tok)
    val_dset = DATASET_CLASS(val_tok)
    train_dl = DataLoader(train_dset, batch_size=bsz, shuffle=False)
    val_dl = DataLoader(val_dset, batch_size=bsz, shuffle=False)

    if MODEL_WEIGHTS is not None:
        model.load_state_dict(torch.load(MODEL_WEIGHTS))

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().cpu()
        return hook
    
    for layer, name in layers_to_hook:
        activation = {}
        SAVE_PATH = f'{SAVE_BASE}-{name}.ptdata'
        if os.path.exists(SAVE_PATH):
            if not overwrite:
                print(f'skipping {name} because it exists')
                continue
            else:
                print(f'WARNING: overwriting {SAVE_PATH}')

        handle = layer.register_forward_hook(get_activation(name))
        model.eval()
        model.to(device)
        final_extracts = {}
        for dl_name, dl in [('train', train_dl), ('val', val_dl)]:
            extracts = []
            with torch.no_grad():
                for bert_inp, lbl_ep in tqdm(dl, desc=dl_name):
                    bert_inp = {name: t.to(device) for name, t in bert_inp.items()}
                    model(**bert_inp)
                    for _, layer_output in activation.items():
                        for i in range(len(lbl_ep)):
                            start_idx = lbl_ep[i,0]
                            end_idx = lbl_ep[i,1]
                            extracts.append(layer_output[i,start_idx:end_idx,:])
            flattened_extracts = torch.cat(extracts, dim=0)
            final_extracts[dl_name] = flattened_extracts
        handle.remove()
        torch.save(final_extracts, SAVE_PATH)
        print(f'Extracts saved to {SAVE_PATH}')

if __name__ == '__main__':
    kwargs_sets = [
        # dict(
        #     DATASET_CLASS = BertExtractionDataset,
        #     MODEL_TYPE = 'bert-base-uncased',
        #     MODEL_WEIGHTS = None,
        #     SAVE_BASE = 'data/probe/ontoV4/bert_mlm_none'
        # ),
        dict(
            DATASET_CLASS = BertExtractionDataset,
            MODEL_TYPE = 'bert-base-uncased',
            MODEL_WEIGHTS = 'checkpoints/bert_finetune_old.pth',
            SAVE_BASE = 'data/probe/ontoV4/bert_mlm_mnli'
        ),
        # dict(
        #     DATASET_CLASS = DistilBertExtractionDataset,
        #     MODEL_TYPE = 'distilbert-base-uncased',
        #     MODEL_WEIGHTS = None,
        #     SAVE_BASE = 'data/probe/ontoV4/distilbert_mlm_none'
        # ),
        dict(
            DATASET_CLASS = DistilBertExtractionDataset,
            MODEL_TYPE = 'distilbert-base-uncased',
            MODEL_WEIGHTS = 'checkpoints/distilbert_finetune_old.pth',
            SAVE_BASE = 'data/probe/ontoV4/distilbert_mlm_mnli'
        ),
        dict(
            DATASET_CLASS = DistilBertExtractionDataset,
            MODEL_TYPE = 'distilbert-base-uncased',
            MODEL_WEIGHTS = 'checkpoints/distilbert_distfinetune_strongDistLoss_old.pth',
            SAVE_BASE = 'data/probe/ontoV4/distilbert_mlm_mixedmnli'
        ),
    ]
    for kwargs in kwargs_sets:
        main(**kwargs, DATASET_BASE='data/ontonotes/ontonotes', overwrite=False)
