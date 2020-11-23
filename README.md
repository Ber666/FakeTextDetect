# Fake Text Detection
2020 Fall, Internet Data Mining course, PKU
haoshibo@pku.edu.cn

## Introduction
A bert-based model to distinguish between human-written text and machine-generated text.
Note: the repo isn't well-organized, e.g. some of the codes might be commented out for efficiency in debug. You may have to select some contents what you really want. 

## Requirements
+ python 3.6
+ pytorch 1.4
+ Transformers 3.5.1

## Preprocess
+ first transform the files to jsonlines, like:
```json
{"text": "......", "label": "neg"}
{"text": "......", "label": "neg"}
...
```
For this course, `preprocess_dm.py` transforms the dataset to such formats.

+ second, split a validation set from training set, and generate encodings and labels with bert tokenizer, the result files is `train_labels.pkl`, `train_encodings.pkl`, and counterparts of val and test set.

+ for a coherence feature mentioned in report, two adjacent sentences are concatenated to be fed into tokenizer.
+ The process would be killed when executing `pickle.dump`, probably because of lack of memory. This is a bug to be fixed.

## Training
+ `bert_main.py` use DistilBertForSequenceClassification models to train a 2-class classification model.
+ best model is saved in `state_distilled_bert_best.pth`

## Test
+ run `bert_test.py`

## Experiment Result
+ approximately 0.80 accuracy can be achieved on test set. Details are in the report.