from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, concatenate_datasets
import numpy as np
# from utils import current_pst_time, get_nli_tok_func, NLI_LABEL_MAP
import json
from tqdm import tqdm
from collections import Counter

# randomly sample both sets because they are too big
rng = np.random.default_rng(seed=224)
keep_rate = 0.3

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
raw = load_dataset('conll2012_ontonotesv5', 'english_v4')

for split in ['train', 'validation']:
    bert_usable_data = {'text_tokens':[],
                        'input_ids':[], 
                        'token_type_ids':[],
                        'attention_mask':[],
                        'ner_labels':[],
                        'label_endpoints':[]}
    base_data = raw[split]['sentences']
    for doc in tqdm(base_data, position=0, leave=True):
        for sent_data in doc:
            if rng.random() > keep_rate:
                continue
            word_tokens = sent_data['words']
            ner_tags = sent_data['named_entities']
            sent_tok_data = tokenizer(word_tokens, is_split_into_words=True, \
                                  return_token_type_ids=True, return_attention_mask=True,\
                                  truncation='longest_first', padding='max_length', max_length=80)
            sw_tok = tokenizer.convert_ids_to_tokens(sent_tok_data['input_ids']) 
            
            i_start, i_end = sw_tok.index('[CLS]')+1, sw_tok.index('[SEP]')
            subword_tokens = sw_tok[i_start:i_end] # remove padding and special tokens
            subword_ner_tags = []
            word_idx = -1
            sw_idx = 0
            for sw in subword_tokens:
                if (word_idx+1 < len(ner_tags)) and (sw[0] == word_tokens[word_idx+1][0].lower()):
                    word_idx += 1
                subword_ner_tags.append(ner_tags[word_idx])
            assert len(subword_ner_tags) == i_end - i_start
            bert_usable_data['text_tokens'].append(subword_tokens)
            bert_usable_data['input_ids'].append(sent_tok_data['input_ids'])
            bert_usable_data['token_type_ids'].append(sent_tok_data['token_type_ids'])
            bert_usable_data['attention_mask'].append(sent_tok_data['attention_mask'])
            bert_usable_data['ner_labels'].append(subword_ner_tags)
            bert_usable_data['label_endpoints'].append((i_start, i_end))
    if split == 'validation':
        split = 'val'
    savename = f'data/ontonotes/ontonotes-{split}-bert_subwordtok_redo.json'
    with open(savename, 'w') as f:
        json.dump(bert_usable_data, f, indent=2)
    print(f"Saved as: {savename}")
    print('\n'+'-'*20+'\n')