from transformers import EarlyStoppingCallback, IntervalStrategy
import evaluate
import transformers
import os
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
import math
import pandas as pd
from datasets import Dataset
from datasets import ClassLabel
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
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
BATCH_SIZE = 16
FEATURES_PATH = 'data/features.pkl'
embedding_dim = 32
max_len = 256
EPOCH=100
metric = evaluate.load("seqeval")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

os.environ["WANDB_PROJECT"] = "propgum"

def get_label_maps(splits, FEATURES):
    datasets = [x for split in splits for x in Path(DATA_PATH.format(split)).read_text().strip().split('\n\n')]
    label_lists = {f: [] for f in FEATURES}
    label_map = {}
    for dt in datasets:
        text = [x.split()[0] for x in dt.split('\n')]
        upos = [x.split()[1] for x in dt.split('\n')]
        att = [x.split()[3] for x in dt.split('\n')]
        deprel = [x.split()[2] for x in dt.split('\n')]
        arg1 = [x.split()[4] for x in dt.split('\n')]
        arg2 = [x.split()[5] for x in dt.split('\n')]
        arg3 = [x.split()[6] for x in dt.split('\n')]
        ner = [x.split()[7] for x in dt.split('\n')]
        converted = {'tokens': text, 'upos': upos, 'att': att, 'deprel': deprel, 'arg1': arg1, 'arg2': arg2,
                     'arg3': arg3,
                     'ner_tags': ner}
        for feature in FEATURES:
            label_lists[feature].extend(converted[feature])
            label_lists[feature].append('UNK')
    for f in FEATURES:
        label_map[f] = {lab: i for i, lab in enumerate(sorted(list(set(label_lists[f]))))}
    return label_map


def convert_data(data, feature_maps):
    converted_data = []
    for i, dt in tqdm(enumerate(data)):
        lines = dt.split('\n')
        text = [x.split()[0] for x in lines]
        upos = [feature_maps['upos'].get(x.split()[1], feature_maps['upos']['UNK']) for x in lines]
        att = [feature_maps['att'].get(x.split()[3], feature_maps['att']['UNK']) for x in lines]
        deprel = [feature_maps['deprel'].get(x.split()[2], feature_maps['deprel']['UNK']) for x in lines]
        #arg1= [feature_maps['arg1'].get(x.split()[4], feature_maps['arg1']['UNK']) for x in lines]
        #arg2 = [feature_maps['arg2'].get(x.split()[5], feature_maps['arg2']['UNK']) for x in lines]
        #arg3 = [feature_maps['arg3'].get(x.split()[6], feature_maps['arg3']['UNK']) for x in lines]

        ner = [x.split()[7] for x in lines]
        converted = {
            'tokens': text,
            'upos_ids': upos,
            'att_ids': att,
            'deprel_ids': deprel,
            'ner_tags': ner,
         #   'arg1': arg1,
         #   'arg2': arg2,
         #   'arg3': arg3,
        }
        converted_data.append(converted)
    return converted_data

def get_data_and_feature(split, feature_map):
    data_split = Path(DATA_PATH.format(split)).read_text().strip().split('\n\n')
    converted_data = convert_data(data_split, feature_map)
    return Dataset.from_list(converted_data)


def preprocess_function(examples):

    tokenized_inputs = tokenizer(examples['tokens'], truncation=True,padding='max_length', max_length=max_len,  is_split_into_words=True)


    labels = []
    pos_features = []
    att_features = []
    deprel_features = []
    arg1_features = []
    arg2_features = []
    arg3_features = []


    for i, label in enumerate(examples["ner_tags"]):

        label = classmap.str2int(label)


        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None


        label_ids = []
        pos_ids = []
        att_ids = []
        deprel_ids = []
        arg1_ids = []
        arg2_ids = []
        arg3_ids = []

        for word_idx in word_ids:
            if word_idx is None:

                label_ids.append(-100)
                pos_ids.append(-100)
                att_ids.append(-100)
                deprel_ids.append(-100)
                # arg1_ids.append(-100)
                # arg2_ids.append(-100)
                # arg3_ids.append(-100)
            elif word_idx != previous_word_idx:

                label_ids.append(label[word_idx])
                pos_ids.append(examples['upos_ids'][i][word_idx])
                att_ids.append(examples['att_ids'][i][word_idx])
                deprel_ids.append(examples['deprel_ids'][i][word_idx])
               # arg1_ids.append(examples['arg1'][i][word_idx])
               # arg2_ids.append(examples['arg2'][i][word_idx])
               # arg3_ids.append(examples['arg3'][i][word_idx])
            else:

                label_ids.append(-100)
                pos_ids.append(-100)
                att_ids.append(-100)
                deprel_ids.append(-100)
                # arg1_ids.append(-100)
                # arg2_ids.append(-100)
                # arg3_ids.append(-100)
            previous_word_idx = word_idx


        labels.append(label_ids)
        pos_features.append(pos_ids)
        att_features.append(att_ids)
        deprel_features.append(deprel_ids)
        # arg1_features.append(arg1_ids)
        # arg2_features.append(arg2_ids)
        # arg3_features.append(arg3_ids)


    tokenized_inputs["labels"] = labels
    tokenized_inputs["upos_ids"] = pos_features
    tokenized_inputs["att_ids"] = att_features
    tokenized_inputs["deprel_ids"] = deprel_features
    #tokenized_inputs["arg1_ids"] = arg1_features
    #tokenized_inputs["arg2_ids"] = arg2_features
    #tokenized_inputs["arg3_ids"] = arg3_features

    return tokenized_inputs


class CustomDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        batch = super().__call__(features)
        max_length = max(len(f['input_ids']) for f in features)

        def pad_and_stack(field_name, dtype=torch.long):
            padded_tensors = []
            for f in features:
                field_value = f[field_name]
                pad_length = max_length - len(field_value)
                padded_field = torch.tensor(field_value + [-100] * pad_length, dtype=dtype)
                padded_tensors.append(padded_field)
            return torch.stack(padded_tensors)

        batch['upos_ids'] = pad_and_stack('upos_ids')
        batch['att_ids'] = pad_and_stack('att_ids')
        batch['deprel_ids'] = pad_and_stack('deprel_ids')
       # batch['arg1_ids'] = pad_and_stack('arg1_ids')
       # batch['arg2_ids'] = pad_and_stack('arg2_ids')
       # batch['arg3_ids'] = pad_and_stack('arg3_ids')

        return batch



def get_embed_dim(feature_size: int, max_dim: int) -> int:
    return min(max_dim, int(math.sqrt(feature_size)))


class CustomModelConfig(DebertaConfig):
    model_type = "deberta"

    def __init__(self, upos_size=0, att_size=0, deprel_size=0, arg1_size=0, arg2_size=0, arg3_size=0, embedding_dim=embedding_dim,
     model_checkpoint=None, num_labels=0, **kwargs):
        super().__init__(**kwargs)
        self.model_checkpoint = model_checkpoint
        self.num_labels = num_labels
        self.upos_size = upos_size
        self.att_size = att_size
        # self.arg1_size = arg1_size
        # self.arg2_size = arg2_size
        # self.arg3_size = arg3_size
        self.embedding_dim = embedding_dim
        self.deprel_size = deprel_size


    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)

    def to_dict(self):
        output = super().to_dict()
        output.update({
            "model_checkpoint": self.model_checkpoint,
            "num_labels": self.num_labels,
            "upos_size": self.upos_size,
            "att_size": self.att_size,
            "deprel_size": self.deprel_size,
            # "arg1_size": self.arg1_size,
            # "arg2_size": self.arg2_size,
            # "arg3_size": self.arg3_size,
            "embedding_dim": self.embedding_dim
        })
        return output


class CustomModelforClassification(DebertaPreTrainedModel):
    def __init__(self, config: CustomModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.deberta = DebertaModel(config)
        self.deberta_dropout = nn.Dropout(0.3)
        upos_dim = get_embed_dim(config.embedding_dim, config.upos_size)
        deprel_dim = get_embed_dim(config.embedding_dim, config.deprel_size)
        # arg1_dim = get_embed_dim(config.embedding_dim, config.arg1_size)
        # arg2_dim = get_embed_dim(config.embedding_dim, config.arg2_size)
        # arg3_dim = get_embed_dim(config.embedding_dim, config.arg3_size)
        att_dim = get_embed_dim(config.embedding_dim, config.att_size)
        
        
        self.upos_embed = nn.Sequential(
            nn.Embedding(config.upos_size, upos_dim), nn.Dropout(p=0.2))
        self.deprel_embed = nn.Sequential(
            nn.Embedding(config.deprel_size,deprel_dim), nn.Dropout(p=0.2))
        # self.arg1_embed = nn.Sequential(nn.Embedding(config.arg1_size, arg1_dim), nn.Dropout(p=0.2))
        # self.arg2_embed = nn.Sequential(nn.Embedding(config.arg2_size, arg2_dim), nn.Dropout(p=0.2))
        # self.arg3_embed = nn.Sequential(nn.Embedding(config.arg3_size, arg3_dim), nn.Dropout(p=0.2))
        self.att_embed = nn.Sequential(nn.Embedding(config.att_size, att_dim), nn.Dropout(p=0.2))
        self.fusion_dropout = nn.Dropout(0.2)
        lstm_input_dim = config.hidden_size + upos_dim + deprel_dim + att_dim
        # lstm_input_dim = config.hidden_size + upos_dim + deprel_dim + arg1_dim + arg2_dim + arg3_dim + att_dim
        self.bilstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=200,
                num_layers=2,
                dropout=0.1,
                bidirectional=True,
                batch_first=True)
        
        self.classifier = nn.Linear(400, config.num_labels)

    def forward(self, input_ids, attention_mask, upos_ids, att_ids, deprel_ids, arg1_ids, arg2_ids, arg3_ids, labels=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        sequence_output = self.deberta_dropout(sequence_output)
        labels = labels.long()
        upos_ids = torch.where(upos_ids == -100, torch.tensor(0, device=upos_ids.device), upos_ids)
        att_ids = torch.where(att_ids == -100, torch.tensor(0, device=att_ids.device), att_ids)
        # arg1_ids = torch.where(arg1_ids == -100, torch.tensor(0, device=arg1_ids.device), arg1_ids)
        # arg2_ids = torch.where(arg2_ids == -100, torch.tensor(0, device=arg2_ids.device), arg2_ids)
        # arg3_ids = torch.where(arg3_ids == -100, torch.tensor(0, device=arg3_ids.device), arg3_ids)
        deprel_ids = torch.where(deprel_ids == -100, torch.tensor(0, device=deprel_ids.device), deprel_ids)

        upos_emb = self.upos_embed(upos_ids)  # [batch, seq_len, embedding_dim]
        att_emb = self.att_embed(att_ids)
        deprel_emb = self.deprel_embed(deprel_ids)
        # arg1_emb = self.arg1_embed(arg1_ids)
        # arg2_emb = self.arg2_embed(arg2_ids)
        # arg3_emb = self.arg3_embed(arg3_ids)
        combined = torch.cat([sequence_output, upos_emb, att_emb, deprel_emb], dim=-1)
        # combined = torch.cat([sequence_output, upos_emb, att_emb, deprel_emb, arg1_emb, arg2_emb, arg3_emb], dim=-1)
        combined = self.fusion_dropout(combined)
        combined, _ = self.bilstm(combined)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            # labels = labels[labels != -100]

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            logits_flat = logits.view(-1, self.config.num_labels)  # [batch_size * seq_len, num_labels]
            labels_flat = labels.view(-1)  # [batch_size * seq_len]

            active_loss = labels_flat != -100
            active_indices = active_loss.nonzero().squeeze()

            if active_indices.numel() == 0:
                loss = torch.tensor(0.0, device=logits.device)
            else:
                active_logits = logits_flat[active_indices]
                active_labels = labels_flat[active_indices]
                loss = loss_fct(active_logits, active_labels)

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [classmap.int2str(int(p)) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [classmap.int2str(int(l)) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]


    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results['overall_f1'], results['overall_accuracy'])
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def write_pred(split, output_file):
    result = trainer.predict(dataset_dict[split])
    prediction = np.argmax(result.predictions, axis=2)
    label = result.label_ids
    text = [x['tokens'] for x in dataset_dict[split]]
    # labels = [x['ner_tags'] for x in dataset_dict[split]]
    with open(output_file, 'w') as out_f:
        for j, (predictions, labels) in enumerate(zip(prediction, label)):
            true_predictions = [classmap.int2str(int(prediction)) for prediction, label in zip(predictions, labels) if
                                label != -100]
            true_labels = [classmap.int2str(int(label)) for prediction, label in zip(predictions, labels) if
                           label != -100]

            for i, token in enumerate(text[j]):
                out_f.write(f'{token}\t{true_labels[i]}\t{true_predictions[i]}\n')
            out_f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='add training hyperparameters')
    parser.add_argument('-t', '--training_split', type=str, default='train')
    parser.add_argument('-s', '--test_split', type=str, default='test')
    parser.add_argument('-d', '--dev_split', type=str, default='dev')
    parser.add_argument('-c', '--checkpoint', default=None)

    args = parser.parse_args()

    train = args.training_split
    dev = args.dev_split
    test = args.test_split
    checkpoint = args.checkpoint

    data_collator = CustomDataCollator(tokenizer=tokenizer)

    ###################################################################################
    ################# get features ###################################################
    feature_maps = get_label_maps([train, dev, test], FEATURES)

   # upos_map = feature_maps['upos']
    #att_map = feature_maps['att']
    #deprel_map = feature_maps['deprel']
    #feature_map_map = {'upos': upos_map, 'att': att_map, 'deprel': deprel_map}
    #####################################################################################

    train_dataset = get_data_and_feature(train, feature_maps)
    dev_dataset = get_data_and_feature(dev, feature_maps)
    test_dataset = get_data_and_feature(test, feature_maps)

    ner_labels = sorted(list(set(
        [x for y in train_dataset['ner_tags'] + dev_dataset['ner_tags'] + test_dataset['ner_tags'] for x in y])))
    classmap = ClassLabel(num_classes=len(ner_labels), names=list(ner_labels))
    print(ner_labels)
    print(classmap)
    print(feature_maps)
    print('===========================')

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    dev_dataset = dev_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)


    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': dev_dataset,
        'test': test_dataset
    })
    training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=0.0001,
    eval_steps=len(dataset_dict['train']['labels'])//BATCH_SIZE,
    save_steps=len(dataset_dict['train']['labels'])//BATCH_SIZE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCH,
    save_total_limit=5,
    weight_decay=0.01,
    load_best_model_at_end = True,
    metric_for_best_model='f1',
    greater_is_better=True,
    report_to="wandb"
)

    if checkpoint:
        model_config = CustomModelConfig(model_checkpoint=checkpoint, num_labels=len(classmap.names),
                                         upos_size=len(feature_maps['upos']), att_size=len(feature_maps['att']),
                                         deprel_size=len(feature_maps['deprel']), arg1_size=len(feature_maps['arg1']),
                                         arg2_size=len(feature_maps['arg2']), arg3_size=len(feature_maps['arg3']),)
        model = CustomModelforClassification.from_pretrained(checkpoint, config=model_config)
        model.eval()
    else:
        model_config = CustomModelConfig(model_checkpoint=MODEL_NAME, num_labels=len(classmap.names),
                                         upos_size=len(feature_maps['upos']), att_size=len(feature_maps['att']),
                                         deprel_size=len(feature_maps['deprel']), arg1_size=len(feature_maps['arg1']),
                                         arg2_size=len(feature_maps['arg2']), arg3_size=len(feature_maps['arg3']),)
        model = CustomModelforClassification(model_config)

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=6,  
            early_stopping_threshold=0.001,
        )
    ],
            compute_metrics=compute_metrics
        )


    trainer.train()
    trainer.save_model(f'ner_new/')
    trainer.evaluate()
    write_pred('validation', 'pred-dev.tsv')
    write_pred('test', 'pred-test2.tsv')
    

