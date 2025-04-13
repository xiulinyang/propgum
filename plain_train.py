import evaluate
import transformers
from transformers import AutoModelForTokenClassification, Trainer, AutoTokenizer, DataCollatorForTokenClassification, \
    Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModel
import torch
import datasets
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from datasets import Features, Sequence, Value, ClassLabel
from transformers import DataCollatorForTokenClassification
from pathlib import Path
import pandas as pd
from datasets import Dataset
from datasets import ClassLabel
from transformers import Trainer, TrainingArguments
import torch
import numpy as np
from datasets import DatasetDict
import torch.nn as nn
from transformers import PretrainedConfig
from transformers import PretrainedConfig
from transformers import DebertaPreTrainedModel, DebertaModel
import torch.nn as nn
from transformers import DebertaConfig
import argparse

MODEL_NAME = 'microsoft/deberta-base'
DATA_PATH = 'tagger_new/{}.new.sample.tab'
FEATURES = ['upos', 'att', 'deprel']
BATCH_SIZE = 8
FEATURES_PATH = 'data/features.pkl'
embedding_dim = 50
label_all_tokens = True
max_len = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model=AutoModelForTokenClassification.from_pretrained(MODEL_NAME)


def read_data(split):
    data = Path(DATA_PATH.format(split)).read_text().strip().split('\n\n')
    raw_data_list = []
    for idx, sent in enumerate(data):
        sent_meta_dict = {}
        sentence = [x.split()[0] for x in sent]
        labels = [x.split()[-1] for x in sent]
        sent_meta_dict['input'] = sentence
        # raw_data_dict[idx]['words'] = sentence
        sent_meta_dict['ner_tags'] = labels
        raw_data_list.append(sent_meta_dict)
    return raw_data_list


train_data_list = read_data('train')
dev_data_list = read_data('dev')
test_data_list = read_data('test')

train_dataset = Dataset.from_dict({k: [d[k] for d in train_data_list] for k in train_data_list[0]})
dev_dataset = Dataset.from_dict({k: [d[k] for d in dev_data_list] for k in dev_data_list[0]})
test_dataset = Dataset.from_dict({k: [d[k] for d in test_data_list] for k in test_data_list[0]})


raw_data = DatasetDict({"train": train_dataset
                        , "dev": dev_dataset,
                        "test": test_dataset})






