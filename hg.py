
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


MODEL_NAME = 'microsoft/deberta-base'
DATA_PATH = 'tagger_new_convert/{}.new.sample.tab'
FEATURES = ['upos', 'att', 'deprel']
BATCH_SIZE = 8
FEATURES_PATH = 'data/features.pkl'
embedding_dim = 50
label_all_tokens = True
max_len = 256

metric = evaluate.load("seqeval")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


def get_label_maps(splits, FEATURES):
    datasets = [x for split in splits for x in Path(DATA_PATH.format(split)).read_text().strip().split('\n\n')]
    label_lists = {f: [] for f in FEATURES}
    label_map = {}
    for dt in datasets:
        text = [x.split('\t')[0] for x in dt.split('\n')]
        upos = [x.split('\t')[2] for x in dt.split('\n')]
        att = [x.split('\t')[4] for x in dt.split('\n')]
        deprel = [x.split('\t')[3] for x in dt.split('\n')]
        arg1 = [x.split('\t')[5] for x in dt.split('\n')]
        arg2 = [x.split('\t')[6] for x in dt.split('\n')]
        arg3 = [x.split('\t')[7] for x in dt.split('\n')]
        ner = [x.split('\t')[8] for x in dt.split('\n')]
        converted = {'tokens': text, 'upos': upos, 'att': att, 'deprel': deprel, 'arg1': arg1, 'arg2': arg2,
                     'arg3': arg3,
                     'ner_tags': ner}
        for feature in FEATURES:
            label_lists[feature].extend(converted[feature])
            label_lists[feature].append('UNK')
    for f in FEATURES:
        label_map[f] = {lab: i for i, lab in enumerate(list(set(label_lists[f])))}
    return label_map


def convert_data(data, feature_maps):
    converted_data = []
    for i, dt in tqdm(enumerate(data)):
        lines = dt.split('\n')
        text = [x.split('\t')[0] for x in lines]
        upos = [feature_maps['upos'].get(x.split('\t')[2], feature_maps['upos']['UNK']) for x in lines]
        att = [feature_maps['att'].get(x.split('\t')[4], feature_maps['att']['UNK']) for x in lines]
        deprel = [feature_maps['deprel'].get(x.split('\t')[3], feature_maps['deprel']['UNK']) for x in lines]
        ner = [x.split('\t')[8] for x in lines]
        converted = {
            'tokens': text,
            'upos_ids': upos,
            'att_ids': att,
            'deprel_ids': deprel,
            'ner_tags': ner
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


    for i, label in enumerate(examples["ner_tags"]):

        label = classmap.str2int(label)


        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None


        label_ids = []
        pos_ids = []
        att_ids = []
        deprel_ids = []

        for word_idx in word_ids:
            if word_idx is None:

                label_ids.append(-100)
                pos_ids.append(-100)
                att_ids.append(-100)
                deprel_ids.append(-100)
            elif word_idx != previous_word_idx:

                label_ids.append(label[word_idx])
                pos_ids.append(examples['upos_ids'][i][word_idx])
                att_ids.append(examples['att_ids'][i][word_idx])
                deprel_ids.append(examples['deprel_ids'][i][word_idx])
            else:

                label_ids.append(label[word_idx] if label_all_tokens else -100)
                pos_ids.append(examples['upos_ids'][i][word_idx])
                att_ids.append(examples['att_ids'][i][word_idx])
                deprel_ids.append(examples['deprel_ids'][i][word_idx])
            previous_word_idx = word_idx


        labels.append(label_ids)
        pos_features.append(pos_ids)
        att_features.append(att_ids)
        deprel_features.append(deprel_ids)


    tokenized_inputs["labels"] = labels
    tokenized_inputs["upos_ids"] = pos_features
    tokenized_inputs["att_ids"] = att_features
    tokenized_inputs["deprel_ids"] = deprel_features

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

        return batch





class CustomModelConfig(DebertaConfig):
    model_type = "deberta"

    def __init__(self, upos_size=0, att_size=0, deprel_size=0, embedding_dim=embedding_dim,hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, model_checkpoint=None, num_labels=None, **kwargs):
        super().__init__(**kwargs)
        self.model_checkpoint = model_checkpoint
        self.num_labels = num_labels
        self.upos_size = upos_size
        self.att_size = att_size
        self.deprel_size = deprel_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

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
            "embedding_dim": self.embedding_dim
        })
        return output


class CustomModelforClassification(DebertaPreTrainedModel):
    def __init__(self, config: CustomModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.deberta = DebertaModel(config)
        self.upos_embed = nn.Embedding(config.upos_size, config.embedding_dim)
        self.att_embed = nn.Embedding(config.att_size, config.embedding_dim)
        self.deprel_embed = nn.Embedding(config.deprel_size, config.embedding_dim)
        self.classifier = nn.Linear(config.hidden_size + 3 * config.embedding_dim, config.num_labels)

    def forward(self, input_ids, attention_mask, upos_ids, att_ids, deprel_ids, labels=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        labels = labels.long()
        print(labels)
        valid_labels = labels[labels!=-100]
        assert (valid_labels.min() >= 0).all() and (valid_labels.max() < 375).all(), "标签值越界！"
        upos_emb = self.upos_embed(upos_ids)  # [batch, seq_len, embedding_dim]
        att_emb = self.att_embed(att_ids)
        deprel_emb = self.deprel_embed(deprel_ids)

        combined = torch.cat([sequence_output, upos_emb, att_emb, deprel_emb], dim=-1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            # 检查标签范围
            valid_labels = labels[labels != -100]
            if valid_labels.numel() > 0:
                assert valid_labels.min() >= 0 and valid_labels.max() < self.config.num_labels, \
                    f"标签值越界: 最小值={valid_labels.min()}, 最大值={valid_labels.max()}, num_labels={self.config.num_labels}"

            # 计算损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            logits_flat = logits.view(-1, self.config.num_labels)  # [batch_size * seq_len, num_labels]
            labels_flat = labels.view(-1)  # [batch_size * seq_len]

            # 过滤无效位置（忽略 -100）
            active_loss = labels_flat != -100
            active_indices = active_loss.nonzero().squeeze()

            # 处理全为 -100 的特殊情况
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
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=0.0001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    metric_for_best_model='accuracy',
)


def write_pred(split, output_file):
    result = trainer.predict(dataset_dict[split])
    prediction = np.argmax(result.predictions, axis=2)
    label = result.label_ids
    text = [x['tokens'] for x in dataset_dict[split]]
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
    data_collator = CustomDataCollator(tokenizer=tokenizer)

    ###################################################################################
    ################# get features ###################################################
    feature_maps = get_label_maps(['dev', 'dev', 'dev'], FEATURES)

    upos_map = feature_maps['upos']
    att_map = feature_maps['att']
    deprel_map = feature_maps['deprel']
    feature_map_map = {'upos': upos_map, 'att': att_map, 'deprel': deprel_map}
    #####################################################################################

    train_dataset = get_data_and_feature('dev', feature_map_map)
    dev_dataset = get_data_and_feature('dev', feature_map_map)
    test_dataset = get_data_and_feature('dev', feature_map_map)

    ner_labels = list(set(
        [x for y in train_dataset['ner_tags'] + dev_dataset['ner_tags'] + test_dataset['ner_tags'] for x in y]))
    classmap = ClassLabel(num_classes=len(ner_labels), names=list(ner_labels))

    print(classmap)
    print('===========================')

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    dev_dataset = dev_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)


    print(train_dataset[0])

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': train_dataset,
        'test': test_dataset
    })

    model_config = CustomModelConfig(model_checkpoint=MODEL_NAME, num_labels=len(classmap.names),
                                     upos_size=len(feature_maps['upos']), att_size=len(feature_maps['att']),
                                     deprel_size=len(feature_maps['deprel']))
    model = CustomModelforClassification(model_config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    write_pred('validation', 'pred-dev.tsv')
    write_pred('test', 'pred-test.tsv')

