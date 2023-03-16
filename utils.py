import datetime
import pytz
import csv

# CONSTANTS
UPOS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
UPOS_TAG_TO_ID = {UPOS_TAGS[i] : i for i in range(len(UPOS_TAGS))}
DEP_TAGS = ['acl', 'acl:relcl', 'advcl', 'advcl:relcl', 'advmod', 'amod', 'appos', 'aux', 'aux:pass', 'case', 'cc', 'cc:preconj', 'ccomp', 'compound', 'compound:prt', 'conj', 'cop', 'csubj', 'csubj:outer', 'csubj:pass', 'dep', 'det', 'det:predet', 'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'flat:foreign', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nmod:npmod', 'nmod:poss', 'nmod:tmod', 'nsubj', 'nsubj:outer', 'nsubj:pass', 'nummod', 'obj', 'obl', 'obl:npmod', 'obl:tmod', 'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']
DEP_TAG_TO_ID = {DEP_TAGS[i] : i for i in range(len(DEP_TAGS))}

# METRICS
def acc_func(predictions, labels):
    acc = (predictions == labels).mean()
    return acc

# LOGGING
def log_metrics(writer, metrics, step, data_src=''):
    '''
    basic logging function. assumes scalar metrics to be logged independently
    writer: SummaryWriter to use for logging
    metrics: list of (<metric name>: <metric value>) tuples
    data_src: source data for metrics, if you want to group train and val metrics, for example
    '''
    for name, v in metrics:
        if data_src != '':
            name = f'{name}/{data_src}'
        writer.add_scalar(name, v, step)

# def log_probe_outputs(timestamp, model_architecture, pretraining, finetuning, probe_type, base_layer, layer1, layer4, layer6, layer12):
#     '''
#     model_architecture: bert, distilbert
#     pretraining: raw, mlm
#     finetuning: none, mnli, distmnli
#     probe_type: individual, conditional
#     '''
#     with open('log_probe_outputs.csv', 'a') as csvfile:
#         fieldnames = ['timestamp', 'model_architecture', 'pretraining', 'finetuning', 'probe_type', \
#         'base_layer', 'layer1', 'layer4', 'layer6', 'layer12']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writerow({'timestamp': timestamp, 'model_architecture': model_architecture, 'pretraining': pretraining, \
#         'finetuning': finetuning, 'probe_type': probe_type, 'base_layer': base_layer, 'layer1': layer1, 'layer4': layer4, \
#         'layer6': layer6, 'layer12': layer12})

def log_probe_outputs(timestamp, model_info, probe_info, layer_data):
    '''
    timestamp: set automatically
    m_arch: model architecture -> {bert, distilbert}
    m_pretrain: model pretraining info -> {mlm}
    m_finetune: model finetuning info -> {none, mnli, distmnli, <other_nli>, dist<other_nli>}
    p_type: type of probe -> {indiv, cond}
    p_task: probe prediction task/label -> {pos, dep}
    '''
    with open('data/probe_outputs.csv', 'a') as csvfile:
        fieldnames = ['timestamp', 'm_arch', 'm_pretrain', 'm_finetune', 'p_type', 'p_task'] \
        + [f'layer{ix}' for ix in [0, 1, 4, 5, 6, 10, 11, 12]]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
        row_data = {'timestamp':timestamp, **model_info, **probe_info, **layer_data}
        writer.writerow(row_data)
    
# MISC
def current_pst_time():
    return pytz.utc.localize(datetime.datetime.utcnow()).astimezone(pytz.timezone('US/Pacific'))

class DotDict:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    def __getitem__(self, k):
        return getattr(self, k)
    def __setitem__(self, k, v):
        setattr(self, k, v)
    def __str__(self):
        return str(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    def _dict(self):
        return self.__dict__
    
def mnli_tok_func(ex, tokenizer):
    return tokenizer(text=ex['premise'], text_pair=ex['hypothesis'], 
                     return_attention_mask=True, return_length=True, return_token_type_ids=True,
                     truncation='longest_first', padding='max_length', max_length=80)