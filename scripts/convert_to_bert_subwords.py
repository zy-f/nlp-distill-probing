from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import json
from collections import OrderedDict
# access to parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bert_finetune import BERTLikeDataset
from utils import UPOS_TAGS, UPOS_TAG_TO_ID, DEP_TAGS, DEP_TAG_TO_ID

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
filepaths = [
    'data/UD_English-EWT/en_ewt-ud-train-wordtok_redo.json',
    'data/UD_English-EWT/en_ewt-ud-dev-wordtok_redo.json',
    'data/UD_English-EWT/en_ewt-ud-test-wordtok_redo.json',
]

for fp in filepaths:
    with open(fp, 'r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    bert_usable_data = {'text_tokens':[],
                        'input_ids':[], 
                        'token_type_ids':[],
                        'attention_mask':[],
                        'pos_labels':[],
                        'dep_labels':[],
                        'label_endpoints':[]}

    for sent_id, (raw_text, word_tokens, pos_tags, dep_tags) in data.items():
        sent_tok_data = tokenizer(word_tokens, is_split_into_words=True, \
                                  return_token_type_ids=True, return_attention_mask=True,\
                                  truncation='longest_first', padding='max_length', max_length=80)
        sw_tok = tokenizer.convert_ids_to_tokens(sent_tok_data['input_ids'])
        i_start, i_end = sw_tok.index('[CLS]')+1, sw_tok.index('[SEP]')
        subword_tokens = sw_tok[i_start:i_end] # remove padding and special tokens
        subword_tags = []
        subword_dep_tags = []
        pos_idx = -1
        sw_idx = 0
        for sw in subword_tokens:
            if (pos_idx+1 < len(pos_tags)) and (sw[0] == word_tokens[pos_idx+1][0].lower()):
                pos_idx += 1
            subword_tags.append(pos_tags[pos_idx])
            subword_dep_tags.append(dep_tags[pos_idx])
        subword_tag_ids = [UPOS_TAG_TO_ID[tag] for tag in subword_tags]
        subword_dep_tag_ids = [DEP_TAG_TO_ID[tag] for tag in subword_dep_tags]
        assert len(subword_tag_ids) == i_end - i_start
        assert len(subword_dep_tag_ids) == i_end - i_start
        bert_usable_data['text_tokens'].append(subword_tokens)
        bert_usable_data['input_ids'].append(sent_tok_data['input_ids'])
        bert_usable_data['token_type_ids'].append(sent_tok_data['token_type_ids'])
        bert_usable_data['attention_mask'].append(sent_tok_data['attention_mask'])
        bert_usable_data['pos_labels'].append(subword_tag_ids)
        bert_usable_data['dep_labels'].append(subword_dep_tag_ids)
        bert_usable_data['label_endpoints'].append((i_start, i_end))
    savename = fp[:fp.rfind('wordtok')] + 'bert_subwordtok_redo.json'
    with open(savename, 'w') as f:
        json.dump(bert_usable_data, f, indent=4)
    print(f"Saved as: {savename}")
    print('\n'+'-'*20+'\n')