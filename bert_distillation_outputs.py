from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from bert_finetune import BERTLikeDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
from utils import get_nli_tok_func

def get_bert_logits(dset_name='multi_nli'):
    '''
    utility function for other files to load cached BERT outputs
    '''
    load_path = f"data/distill/BERTbase_{dset_name}_finetune-outputs.npz"
    bert_outputs = np.load(load_path)
    return bert_outputs

def main():
    # load data
    data_name = "multi_nli"
    save_path = f"data/distill/BERTbase_{data_name}_finetune-outputs"
    dset_splits = ['train', 'validation_matched', 'validation_mismatched']
    n_classes = 3
    mnli_raw = load_dataset(data_name)

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok_func = get_nli_tok_func(tokenizer)
    mnli_tok = mnli_raw.map(tok_func, batched=True)
    
    bsz = 64
    device = 'cuda:0'
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.load_state_dict(torch.load('checkpoints/bert_finetune_retry.pth'))
    print('model finetune weights loaded')
    
    model.eval()
    model.to(device)
    bert_outputs_cache = {}
    
    for split in dset_splits:
        dset = BERTLikeDataset(mnli_tok[split])

        # create dataloaders (batches and randomizes samples)
        dl = DataLoader(dset, batch_size=bsz, shuffle=False, drop_last=False)
        model_outputs = []
        with torch.no_grad():
            for batch in tqdm(dl, desc=split):
                (inp, tti, am, _) = (t.to(device) for t in batch)
                out = model(input_ids=inp, attention_mask=am, token_type_ids=tti, return_dict=True)
                model_outputs.append(out.logits.cpu())
        bert_outputs_cache[split] = torch.cat(model_outputs, dim=0).numpy()
        assert len(bert_outputs_cache[split]) == mnli_tok[split].num_rows
    
    np.savez(save_path, **bert_outputs_cache)
    print(f"Outputs for splits {', '.join(dset_splits)} of dataset {data_name} saved to {save_path}")
    

if __name__ == '__main__':
    main()