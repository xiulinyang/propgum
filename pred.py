import numpy as np

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