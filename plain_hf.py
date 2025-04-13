import evaluate
import transformers
from transformers import AutoModelForTokenClassification, Trainer, AutoTokenizer, DataCollatorForTokenClassification, \
    Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModel

from transformers import pipeline
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
import numpy as np

from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer

import torch
import random
import numpy as np

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MODEL_NAME = 'microsoft/deberta-base'
DATA_PATH = 'tagger_new/{}.new.sample.tab'
FEATURES = ['upos', 'att', 'deprel']
BATCH_SIZE = 8
FEATURES_PATH = 'data/features.pkl'
embedding_dim = 50
MODEL_NAME= "microsoft/deberta-v3-base"
max_len = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)



def read_data(split):
    data = Path(DATA_PATH.format(split)).read_text().strip().split('\n\n')
    raw_data_list = []
    for idx, sent in enumerate(data):
        sent_meta_dict = {}
        sentence = [x.split()[0] for x in sent.split('\n')]
        labels = [x.split()[-1] for x in sent.split('\n')]
        sent_meta_dict['input'] = sentence
        # raw_data_dict[idx]['words'] = sentence
        sent_meta_dict['ner_tags'] = labels
        raw_data_list.append(sent_meta_dict)
    return raw_data_list


train_data_list = read_data('dev')
dev_data_list = read_data('dev')
test_data_list = read_data('eval')

train_dataset = Dataset.from_dict({k: [d[k] for d in train_data_list] for k in train_data_list[0]})
dev_dataset = Dataset.from_dict({k: [d[k] for d in dev_data_list] for k in dev_data_list[0]})
test_dataset = Dataset.from_dict({k: [d[k] for d in test_data_list] for k in test_data_list[0]})


raw_data = DatasetDict({"train": train_dataset
                        , "dev": dev_dataset,
                        "test": test_dataset})

label_list = [y for x in raw_data['train']['ner_tags']+raw_data['dev']['ner_tags']+raw_data['test']['ner_tags'] for y in x]

label_list= sorted(list(set(label_list)))
id2label = {id:label for id, label in enumerate(label_list)}
label2id = {label:id for id, label in enumerate(label_list)}

import pickle

#with open('label_mappings.pkl', 'wb') as f:
#    pickle.dump({
#        'id2label': id2label,
#        'label2id': label2id,
#        'label_list': label_list
#    }, f)


#with open('label_mappings.pkl', 'rb') as f:  # 注意这里是'rb'模式
#    label_data = pickle.load(f)

# 解包数据
#id2label = label_data['id2label']
#label2id = label_data['label2id']
#label_list = label_data['label_list']


def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(examples['input'], truncation=True, is_split_into_words=True)
  labels = []
  for i, label in enumerate(examples['ner_tags']):
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    previous_word_idx =None
    label_ids = []
    for word_idx in word_ids:
      if word_idx is None:
        label_ids.append(-100)
      elif word_idx!= previous_word_idx:
        label_ids.append(label2id[label[word_idx]])
      else:
        label_ids.append(-100)

      previous_word_idx=word_idx
    labels.append(label_ids)
  tokenized_inputs['labels'] = labels
  return tokenized_inputs

tokenized_data = raw_data.map(
    tokenize_and_align_labels,
    batched=True,
    
)


model = AutoModelForTokenClassification.from_pretrained(
   MODEL_NAME,
    num_labels=len(id2label),
    id2label=id2label,
    label2id = label2id
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]


    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load('seqeval')

id2label= {i:label for i, label in enumerate(label_list)}
label2id = {label:i for i, label in enumerate(label_list)}


print(id2label)
print(label2id)

def tag_sentence(text):
  inputs = tokenizer(text, truncation=True, is_split_into_words=True,return_tensors='pt').to('cuda')
  with torch.no_grad():
        outputs = model(**inputs)
  predictions = np.argmax(outputs.logits, axis=2)
  return predictions


def predict_write(model,tokenized_data):
    data = tokenized_data['test']
    labels = data['ner_tags']
    input_sent = data['input']
    with open('pred.tsv', 'w') as p:
        for i,sent in enumerate(input_sent):
            pred = tag_sentence(sent)[0]
            pred =[id2label[p] for p in pred]
            gold = [x for x in labels[i] if x!=-100]
            print(sent, gold)
            for j in range(len(sent)):
                print(gold[j])
                print(sent[j])
                p.write(f'{sent[j]}\t{gold[j]}\t{pred[j]}\n')

        p.write('\n')

def tag_sentence(text):
    model.eval()
    model.to('cuda')

    inputs = tokenizer(
        text,
        truncation=True,
        is_split_into_words=True,
        return_tensors='pt',
        padding='max_length',  # 固定长度对齐
        max_length=max_len
    ).to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)
    predictions = np.argmax(outputs.logits.cpu().numpy(), axis=2)

    # 关键改进：子词对齐处理
    word_ids = inputs.word_ids(batch_index=0)
    print(word_ids)
    previous_word_idx = None
    word_predictions = []

    for idx, word_idx in enumerate(word_ids):
        # 过滤特殊Token和子词
        if word_idx is None:
            continue
        if word_idx == previous_word_idx:
            continue

        # 索引安全检查
        if idx >= len(predictions[0]):
            break

        word_predictions.append(id2label[predictions[0][idx]])
        previous_word_idx = word_idx

    return word_predictions  # 现在长度等于原始句子长度

def predict_write(model, tokenized_data):
    data = tokenized_data['test']
    with open('pred.tsv', 'w') as p:
        for i, sent in enumerate(data['input']):
            # 获取对齐后的预测
            pred = tag_sentence(sent)

            # 获取原始标签（已过滤-100）
            gold = [l for l in data['ner_tags'][i] if l != -100]
            print(gold, pred)
            # 长度验证
            assert len(pred) == len(sent), f"预测长度{len(pred)}≠句子长度{len(sent)}"
            assert len(gold) == len(sent), f"标签长度{len(gold)}≠句子长度{len(sent)}"

            # 写入文件
            for word, gl, pr in zip(sent, gold, pred):
                p.write(f'{word}\t{gl}\t{pr}\n')
            p.write('\n')

training_args = TrainingArguments(
    output_dir='my_ner_model',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='no',
    #load_best_model_at_end=True,
    report_to=['none']
)


trainer=Trainer(
    model=model,
    train_dataset = tokenized_data['test'],
    eval_dataset= tokenized_data['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics = compute_metrics,
    args=training_args
)


trainer.train()
trainer.save_model(f'ner_new/')
predict_write(model, tokenized_data)
