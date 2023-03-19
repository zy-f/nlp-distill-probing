from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, concatenate_datasets
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import log_generalization_output, acc_func, get_nli_tok_func, NLI_LABEL_MAP
from tqdm import tqdm, trange
from bert_finetune import BERTLikeDataset, bert_epoch_loop
from distilbert_finetune import DistilBERTLikeDataset, distilbert_epoch_loop
import json

MODEL_INFO = {
    'bert_finetune_old': {
        'm_arch': 'bert',
        'm_pretrain': 'mlm',
        'm_finetune': 'mnli'
    },

    'distilbert_finetune_old': {
        'm_arch': 'distilbert',
        'm_pretrain': 'mlm',
        'm_finetune': 'mnli'
    },

    # 'distilbert_distfinetune_onlyDistLoss': {
    #     'm_arch': 'distilbert',
    #     'm_pretrain': 'mlm',
    #     'm_finetune': 'distmnli'
    # },

    'distilbert_distfinetune_strongDistLoss_old': {
        'm_arch': 'distilbert',
        'm_pretrain': 'mlm',
        'm_finetune': 'mixedmnli'
    },
}

def get_dataset(dset):
    # load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok_func = get_nli_tok_func(tokenizer)

    if dset == 'jam':
        jamnli_raw = load_dataset("Ruth-Ann/jampatoisnli")
        jamnli_tok = jamnli_raw.map(tok_func, batched=True)
        def numerically_label(ex):
            ex['label'] = NLI_LABEL_MAP[ex['label']]
            return ex
        jamnli_tok = jamnli_tok.map(numerically_label)
        full_data_tok = jamnli_tok['train']
    elif dset in 'snli':
        snli_raw = load_dataset("snli")
        snli_tok = snli_raw.map(tok_func, batched=True)
        full_data_tok = concatenate_datasets([snli_tok['train'], snli_tok['validation'], snli_tok['test']])
        full_data_tok = full_data_tok.filter(lambda ex: ex['label'] != -1) # remove weird label cases
    elif dset == 'anli':
        anli_raw = load_dataset("anli")
        anli_tok = anli_raw.map(tok_func, batched=True)
        full_data_tok = concatenate_datasets(list(anli_tok.values()))
    elif dset.startswith('dnc'):
        dnc_task = dset.split('-')[-1]
        with open(f'data/DNC/function_words/NLI/nli-{dnc_task}_data.json', 'r') as f:
            data_samples = json.load(f)
        full_data_tok = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'label': []
        }
        for ex in data_samples:
            ex['premise'] = ex.pop('context')
            full_data_tok['label'].append(NLI_LABEL_MAP[ex['label']])
            for k, v in tok_func(ex).items():
                full_data_tok[k].append(v)
    elif dset == 'hans':
        hans_raw = load_dataset("hans")
        hans_tok = hans_raw.map(tok_func, batched=True)
        full_data_tok = concatenate_datasets(list(hans_tok.values()))
    elif dset == 'mnli_m':
        raw = load_dataset('multi_nli')
        full_data_tok = raw.map(tok_func, batched=True)['validation_matched']
    elif dset == 'mnli_mm':
        raw = load_dataset('multi_nli')
        full_data_tok = raw.map(tok_func, batched=True)['validation_mismatched']
    else:
        raise NotImplementedError
    return full_data_tok

def evaluate_on(full_data_tok, model_file, log_info):
    MODEL_WEIGHTS = f'checkpoints/{model_file}.pth'
    
    model_arch = log_info['m_arch']
    MODEL_TYPE = f'{model_arch}-base-uncased'

    DatasetClass = BERTLikeDataset if model_arch == 'bert' else DistilBERTLikeDataset
    epoch_loop = bert_epoch_loop if model_arch == 'bert' else distilbert_epoch_loop

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels=3)
    if MODEL_WEIGHTS is not None:
        model.load_state_dict(torch.load(MODEL_WEIGHTS))
        print(f'weights from {MODEL_WEIGHTS} loaded')

    print('Writing tok data to dataset class...', end='', flush=True)
    full_data = DatasetClass(full_data_tok)
    print('Loaded to dataset.')
    # HYPERPARAMETERS
    bsz = 256
    device = 'cuda:0'
    loss_func = nn.CrossEntropyLoss()
    
    # create dataloaders (batches and randomizes samples)
    full_dl = DataLoader(full_data, batch_size=bsz, shuffle=False)

    model.to(device)
    with torch.no_grad():
        loss, pred, y = epoch_loop(model, full_dl, loss_func, device=device)
        acc = acc_func(pred, y)
        log_info['loss'] = loss
        log_info['acc'] = acc
        print(f"full_data: loss={loss}, acc={acc}")
    log_generalization_output(log_info)
    

def main(dset='anli'):
    full_data_tok = get_dataset(dset)
    models = [
        'bert_finetune_old',
        'distilbert_finetune_old',
        'distilbert_distfinetune_strongDistLoss_old'
    ]
    for model_file in models:
        log_info = {'dset': dset, **MODEL_INFO[model_file]}
        evaluate_on(full_data_tok, model_file, log_info)

if __name__ == '__main__':
    dsets = [
        'dnc-negation'
        # 'dnc-comparatives', 
        # 'dnc-prepositions',
        # 'dnc-quantification',
        # 'dnc-spatial',
        # 'jam',
        # 'snli',
        # 'anli',
        # 'hans',
        # 'mnli_m',
        # 'mnli_mm'
    ]
    for d in dsets:
        main(dset=d)