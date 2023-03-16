import argparse
from collections import OrderedDict
import json
from tqdm import tqdm
from transformers import AutoTokenizer

files = [
    'data/UD_English-EWT/en_ewt-ud-train.conllu',
    'data/UD_English-EWT/en_ewt-ud-dev.conllu',
    'data/UD_English-EWT/en_ewt-ud-test.conllu'
]

for fp in files:
    dset_data = OrderedDict()
    with open(fp,'r') as file:
        raw_text = ""
        sent_id = ""
        word_tokens = []
        pos_tags = []
        dep_tags = []
        for line in tqdm(file):
            if line[0] == '#' or len(line.strip()) == 0:
                if line.startswith('# sent_id'):
                    if len(pos_tags) > 0:
                        dset_data[sent_id] = (raw_text, word_tokens, pos_tags, dep_tags)
                        word_tokens = []
                        pos_tags = []
                        dep_tags = []
                    sent_id = line[len('# sent_id = '):].strip()
                if line.startswith('# text'):
                    raw_text = line[len('# text = '):].strip()
                continue
            tsv_data = line.strip().split('\t')
            if '-' in tsv_data[0] or '.' in tsv_data[0]:
                continue
            assert len(tsv_data) == 10
            word_tokens.append(tsv_data[1])
            pos_tags.append(tsv_data[3])
            dep_tags.append(tsv_data[7])
        dset_data[sent_id] = (raw_text, word_tokens, pos_tags, dep_tags)
    print(f"FILE: {fp}")
    savename = fp[:fp.rfind('.')] + '-wordtok_redo.json'
    with open(savename, 'w') as f:
        json.dump(dset_data, f)
    print(f"Saved as: {savename}")