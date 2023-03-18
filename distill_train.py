from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
                         DistilBertForSequenceClassification, DistilBertConfig
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import log_metrics, current_pst_time, acc_func, DotDict, get_nli_tok_func
from tqdm import tqdm, trange

from bert_distillation_outputs import get_bert_logits

# ==== BIG CODE BLOCKS ====

class JointDistillationLoss(nn.Module):
    def __init__(self, dist_ce_wt=5, hard_ce_wt=2, temp=1):
        # default weights from
        # https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation
        super().__init__()
        self.dist_ce_wt = dist_ce_wt
        self.hard_ce_wt = hard_ce_wt
        self.temp = temp

    def forward(self, s_logits, t_logits, true_labels, is_train=False):
        tau = self.temp if is_train else 1
        t_probs = F.softmax(t_logits/tau, dim=-1)
        dist_loss = F.cross_entropy(s_logits/tau, t_probs) * tau**2
        hard_loss = F.cross_entropy(s_logits, true_labels)
        return self.dist_ce_wt * dist_loss + self.hard_ce_wt * hard_loss


class DistillTuneDistilBERTDataset(Dataset): # dataset for finetuning distilbert with distillation
    def __init__(self, tok_dset, t_logits):
        self.input_ids = tok_dset['input_ids']
        self.attention_mask = tok_dset['attention_mask']
        self.labels = tok_dset['label']
        self.soft_labels = t_logits
        assert len(self.labels) == len(self.soft_labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inp = torch.LongTensor(self.input_ids[idx])
        am = torch.FloatTensor(self.attention_mask[idx])
        bert_logits = torch.FloatTensor(self.soft_labels[idx])
        fwd_inp = {'input_ids': inp, 'attention_mask': am}
        labels = {'t_logits': bert_logits, 'label': self.labels[idx]}
        return fwd_inp, labels

def distill_epoch_loop(model, dl, loss_func, device='cpu', optimizer=None):
    # set training or validation mode
    do_train = optimizer is not None
    model.train() if do_train else model.eval()

    epoch_preds = []
    epoch_labels = []
    accum_loss = 0
    for i, (inputs, labels) in enumerate(tqdm(dl, desc='train' if do_train else 'eval')):
        inputs = {name: t.to(device) for name, t in inputs.items()}
        labels = {name: t.to(device) for name, t in labels.items()}
        out = model(**inputs)
        loss = loss_func(out['logits'], labels['t_logits'], labels['label'], is_train=do_train)
        if do_train:
            optimizer.zero_grad() # at every batch
            loss.backward()
            optimizer.step()
        lbl_pred = torch.argmax(out['logits'].detach(), dim=1)
        epoch_preds.append(lbl_pred.cpu())
        epoch_labels.append(labels['label'].cpu())
        accum_loss += loss.item()
    flat_epoch_preds = torch.cat(epoch_preds, dim=0).numpy()
    flat_labels = torch.cat(epoch_labels, dim=0).numpy()
    epoch_loss = accum_loss/len(flat_labels)
    return epoch_loss, flat_epoch_preds, flat_labels

# ==== END CODE BLOCKS ====

def main():
    # CONSTANTS
    DSET_NAME = 'multi_nli'

    # load data
    mnli_raw = load_dataset(DSET_NAME)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok_func = get_nli_tok_func(tokenizer)
    mnli_tok = mnli_raw.map(tok_func, batched=True)
    bert_logit_data = get_bert_logits(DSET_NAME)
    # initialize datasets that contain (input, attention_mask, label)
    train_data = DistillTuneDistilBERTDataset(mnli_tok['train'], bert_logit_data['train'])
    val_m_data = DistillTuneDistilBERTDataset(mnli_tok['validation_matched'], bert_logit_data['validation_matched'])
    val_mm_data = DistillTuneDistilBERTDataset(mnli_tok['validation_mismatched'], bert_logit_data['validation_mismatched'])
    print('data loaded')

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    # model = DistilBertForSequenceClassification(DistilBertConfig(num_labels=3))
    model.load_state_dict(torch.load('checkpoints/distilbert_distfinetune_strongDistLoss_retry.pth'))
    print('model loaded')

    # HYPERPARAMETERS
    hparams = DotDict(
        lr = 1e-5,
        n_epochs = 5,
        bsz = 64,
        distloss__dist_ce_wt=5,
        distloss__hard_ce_wt=2,
        distloss__temp=2
    )

    device = 'cuda:0'
    # run_name = 'raw_distilbert_distfinetune_onlyDistLoss'
    run_name = 'distilbert_distfinetune_strongDistLoss_retry'

    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.lr)
    loss_func = JointDistillationLoss(dist_ce_wt=hparams.distloss__dist_ce_wt,
                                    hard_ce_wt=hparams.distloss__hard_ce_wt,
                                    temp=hparams.distloss__temp)

    # create dataloaders (batches and randomizes samples)
    train_dl = DataLoader(train_data, batch_size=hparams.bsz, shuffle=True)
    val_m_dl = DataLoader(val_m_data, batch_size=hparams.bsz, shuffle=False)
    val_mm_dl = DataLoader(val_mm_data, batch_size=hparams.bsz, shuffle=False)

    # ==== RUN TRAINING ====
    model.to(device)
    unique_run_name = f"{run_name}-{current_pst_time().strftime('%Y_%m_%d-%H_%M')}"
    log_dir = f"./logs/{unique_run_name}"
    best_val_acc = 0
    with SummaryWriter(log_dir=log_dir) as log_writer:
        for epoch in range(hparams.n_epochs):
            tr_loss, tr_pred, tr_y = distill_epoch_loop(model, train_dl, loss_func, device=device, optimizer=optimizer)
            tr_metrics = [('loss', tr_loss), \
                        ('accuracy', acc_func(tr_pred, tr_y))]
            log_metrics(log_writer, tr_metrics, epoch, data_src='train')
            print(f"Epoch {epoch} train: loss={tr_metrics[0][1]}, acc={tr_metrics[1][1]}")

            # run on val matched dataset
            with torch.no_grad():
                v_m_loss, v_m_pred, v_m_y = distill_epoch_loop(model, val_m_dl, loss_func, device=device)
                v_m_acc = acc_func(v_m_pred, v_m_y)
                v_m_metrics = [('loss', v_m_loss), \
                            ('accuracy', v_m_acc)]
                log_metrics(log_writer, v_m_metrics, epoch, data_src='val_match')
                print(f"Epoch {epoch} val_m: loss={v_m_metrics[0][1]}, acc={v_m_metrics[1][1]}")

                v_mm_loss, v_mm_pred, v_mm_y = distill_epoch_loop(model, val_mm_dl, loss_func, device=device)
                v_mm_acc = acc_func(v_mm_pred, v_mm_y)
                v_mm_metrics = [('loss', v_mm_loss), \
                            ('accuracy', v_mm_acc)]
                log_metrics(log_writer, v_mm_metrics, epoch, data_src='val_mismatch')
                print(f"Epoch {epoch} val_mm: loss={v_mm_metrics[0][1]}, acc={v_mm_metrics[1][1]}")
                val_acc = (v_m_acc + v_mm_acc)/2
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), f'checkpoints/{unique_run_name}.pth')
        log_writer.add_hparams(hparams._dict(), {'res/best_val_acc': best_val_acc})

if __name__ == '__main__':
    main()