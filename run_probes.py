from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import log_metrics, current_pst_time, acc_func, log_probe_outputs, \
                  UPOS_TAGS, DEP_TAGS, NER_TAGS
from tqdm import tqdm
from contextlib import nullcontext

# === LARGE CODE BLOCKS ===
# probe models
class TwoLayerNN(nn.Module):
    def __init__(self, emb_size, hidden_size, n_classes):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# dataset stuff
class ConditionalProbeDataset(Dataset):
    def __init__(self, base_extract, deep_extract, labels):
        self.base_extract = base_extract
        self.deep_extract = deep_extract
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        inp = torch.cat((self.base_extract[i], self.deep_extract[i]), dim=-1)
        lbl = self.labels[i]
        return inp, lbl

def get_conditional_datasets(model_name, layer, task='pos'):
    '''
    task = {pos, dep, ner}
    '''
    if task in ['ner']:
        base_path = 'data/probe/ontoV4'
    elif task in ['pos', 'dep']:
        base_path = 'data/probe/ewt'
    else:
        raise NotImplementedError
    labels = torch.load(f'{base_path}/{task}_labels.ptdata')
    base_layer = torch.load(f'{base_path}/{model_name}-layer0.ptdata')
    if layer == 'layer0':
        deep_layer = {'train': torch.zeros_like(base_layer['train']),
                      'val': torch.zeros_like(base_layer['val'])
        }
    else:
        deep_layer = torch.load(f'{base_path}/{model_name}-{layer}.ptdata')
    train_dset = ConditionalProbeDataset(base_layer['train'], deep_layer['train'], labels['train'])
    val_dset = ConditionalProbeDataset(base_layer['val'], deep_layer['val'], labels['val'])
    return train_dset, val_dset

def get_individual_datasets(model_name, layer, task='pos'):
    '''
    task = {pos, dep, ner}
    '''
    if task == 'ner':
        base_path = 'data/probe/ontoV4'
    elif task in ['pos', 'dep']:
        base_path = 'data/probe/ewt'
    else:
        raise NotImplementedError
    labels = torch.load(f'{base_path}/{task}_labels.ptdata')
    layer_data = torch.load(f'{base_path}/{model_name}-{layer}.ptdata')
    train_dset = torch.utils.data.TensorDataset(layer_data['train'], labels['train'])
    val_dset = torch.utils.data.TensorDataset(layer_data['val'], labels['val'])
    return train_dset, val_dset

def probe_epoch_loop(model, dl, loss_func, pbar=None, device='cpu', optimizer=None):
    # set training or validation mode
    do_train = optimizer is not None
    model.train() if do_train else model.eval()
    epoch_preds = []
    labels = []
    accum_loss = 0
    for i, batch in enumerate(dl):
        (inp, lbl) = (t.to(device) for t in batch)
        out = model(inp)
        loss = loss_func(out, lbl)
        if do_train:
            optimizer.zero_grad() # at every batch
            loss.backward()
            optimizer.step()
        lbl_pred = torch.argmax(out.detach(), dim=1)
        epoch_preds += lbl_pred.cpu().tolist()
        labels += lbl.cpu().tolist()
        accum_loss += loss.item()
        if pbar is not None:
            pbar.update(1)
    flat_epoch_preds = np.array(epoch_preds)
    flat_labels = np.array(labels)
    epoch_loss = accum_loss/len(flat_labels)
    return epoch_loss, flat_epoch_preds, flat_labels

def run_probe(train_dset, val_dset, unique_run_name=None, hidden_size=45, n_classes=17): # main program
    # create probe
    emb_size = val_dset[0][0].shape[-1]
    model = TwoLayerNN(emb_size, hidden_size, n_classes)

    # HYPERPARAMETERS
    lr = 1e-4
    n_epochs = 20
    bsz = 512
    device = 'cuda:0'

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    # create dataloaders (batches and randomizes samples)
    train_dl = DataLoader(train_dset, batch_size=bsz, shuffle=True)
    val_dl = DataLoader(val_dset, batch_size=bsz, shuffle=False)

    # ==== RUN TRAINING ====
    model.to(device)
    log_dir = f"./logs/{unique_run_name}"
    best_val_acc = 0
    TOTAL_STEPS = n_epochs * (len(train_dl) + len(val_dl))
    # bar_fmt = '{desc}{percentage:.0f}%|{bar}|{n}/{total_fmt}'
    with tqdm(total=TOTAL_STEPS, position=0, leave=True) as pbar:
        with SummaryWriter(log_dir=log_dir) \
            if unique_run_name is not None else nullcontext() \
        as log_writer:
            pbar.set_description(f'best={best_val_acc:.3f}')
            for epoch in range(n_epochs):
                tr_loss, tr_pred, tr_y = probe_epoch_loop(model, train_dl, loss_func, pbar=pbar, \
                                                        device=device, optimizer=optimizer)
                tr_metrics = [('loss', tr_loss), \
                            ('accuracy', acc_func(tr_pred, tr_y))]
                if log_writer is not None:
                    log_metrics(log_writer, tr_metrics, epoch, data_src='train')
                # print(f"Epoch {epoch} train: loss={tr_metrics[0][1]}, acc={tr_metrics[1][1]}")

                # run on val dataset
                with torch.no_grad():
                    v_loss, v_pred, v_y = probe_epoch_loop(model, val_dl, loss_func, pbar=pbar, device=device)
                    v_acc = acc_func(v_pred, v_y)
                    v_metrics = [('loss', v_loss), \
                                ('accuracy', v_acc)]
                    if log_writer is not None:
                        log_metrics(log_writer, v_metrics, epoch, data_src='val')
                    # print(f"Epoch {epoch} val: loss={v_metrics[0][1]}, acc={v_metrics[1][1]}")

                    if v_acc > best_val_acc:
                        best_val_acc = v_acc
                        pbar.set_description(f'best={best_val_acc:.3f}')
                        if log_writer is not None:
                            torch.save(model.state_dict(), f'checkpoints/{unique_run_name}.pth')
                if log_writer is not None:
                    log_writer.add_hparams({}, {'res/best_val_acc': best_val_acc})
    return best_val_acc
# ==== END CODE BLOCKS ====

# simple functions to run all probes
def probe_all_layers(layers, model_info, probe_info, probe_dims):
    probe_layers = [f'layer{i}' for i in layers]
    model_name = f"{model_info['m_arch']}_{model_info['m_pretrain']}_{model_info['m_finetune']}"
    print(f'PROBING LAYERS: {probe_layers}')
    if probe_info['p_type'] == 'indiv':
        get_dataset_func = get_individual_datasets
    elif probe_info['p_type'] == 'cond':
        get_dataset_func = get_conditional_datasets
    else:
        raise NotImplementedError
    layer_accs = {}
    for layer in probe_layers:
        train_dset, val_dset = get_dataset_func(model_name, layer, task=probe_info['p_task'])
        tstamp = current_pst_time().strftime('%Y_%m_%d-%H_%M')
        print(f"starting {layer} probe")
        best_val_acc = run_probe(train_dset, val_dset, **probe_dims)
        layer_accs[layer] = best_val_acc
        print('-'*20)
    log_probe_outputs(tstamp, model_info, probe_info, layer_accs)

'''
timestamp: set automatically
m_arch: model architecture -> {bert, distilbert}
m_pretrain: model pretraining info -> {mlm}
m_finetune: model finetuning info -> {none, mnli, distmnli, <other_nli>, dist<other_nli>}
p_type: type of probe -> {indiv, cond}
p_task: probe prediction task/label -> {pos, dep}
'''
probe_dims = {
    'hidden_size': 45,
    'n_classes': -1
}
model_info = {
    'm_arch': 'distilbert',
    'm_pretrain': 'mlm',
    'm_finetune': 'mixedmnli',
}

model_name = f"{model_info['m_arch']}_{model_info['m_pretrain']}_{model_info['m_finetune']}"
print('extracting for:', model_name)

layers = [0, 1, 6, 10, 11, 12] if model_info['m_arch'] == 'bert' else [0, 1, 4, 5, 6]

for p_task in ['ner']: #['pos', 'dep']
    for p_type in ['indiv', 'cond']:
        if p_task == 'pos':
            probe_dims['n_classes'] = len(UPOS_TAGS)
        elif p_task == 'dep':
            probe_dims['n_classes'] = len(DEP_TAGS)
        elif p_task == 'ner':
            probe_dims['n_classes'] = len(NER_TAGS) # from huggingface
        else:
            raise NotImplementedError
        probe_info = {'p_task': p_task, 'p_type': p_type}
        print(f'PROBING VIA {p_type.upper()} PROBE ON {p_task.upper()} LABEL')
        probe_all_layers(layers, model_info, probe_info, probe_dims)