from pathlib import Path
import jsonlines
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import pickle

def savePickle(obj, path):
    print("saving {}".format(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def readFiles(path):
    texts = []
    labels = []
    with open(path, "r+", encoding="utf8") as f:
        for item in tqdm(jsonlines.Reader(f)):
            texts.append(item['text'])
            labels.append((0 if item['label'] == 'neg' else 1))
            
    print("having read {} instances from {}".format(len(texts), path))
    return texts, labels


"""
train_texts, train_labels = readFiles("train.jsonl")
test_texts, test_labels = readFiles("test.jsonl")"""
test2_texts, test2_labels = readFiles("test2.jsonl")
"""train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)"""
print("finish splitting train & val set.")
print("tokenizing...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
"""
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)"""

test2_encodings = tokenizer(test2_texts, truncation=True, padding=True)

# xxx_encodings is a transformers.tokenization_utils_base.BatchEncoding object
# one item of it is a tokenizers.Encoding
#   ids:[101, ..., ..., 1012, 0, 0, 0, ...]
#   type_ids: [0, 0, 0 ...] probably used to distinguish between the first and second sentences
#   tokens: ['[CLS]', 'china', '...', '.', '[SEP]', '[PAD]', ...]


print("finish tokenizing")
"""
savePickle(train_labels, "train_labels.pkl")
savePickle(test_labels, "test_labels.pkl")
savePickle(val_labels, "val_labels.pkl")
savePickle(train_encodings, "train_encodings.pkl")
savePickle(val_encodings, "val_encodings.pkl")
savePickle(test_encodings, "test_encodings.pkl")
"""

savePickle(test2_labels, "test2_labels.pkl")
savePickle(test2_encodings, "test2_encodings.pkl")