from pathlib import Path
import jsonlines
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, BertTokenizerFast
import pickle

def pairGen(texts):
    first_s, second_s = [], []
    st, ed = [], []
    cur = 0
    for i in texts:
        st.append(cur)
        tmp = i.split('. ')
        if len(tmp) == 1:
            first_s.append(tmp[0])
            second_s.append(tmp[0])
            cur += 1
            ed.append(cur)
        else:
            first_s += tmp[:-1]
            second_s += tmp[1:]   
            cur += len(tmp) - 1
            ed.append(cur)     
    assert(len(first_s) == len(second_s))
    return first_s, second_s, st, ed

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



train_texts, train_labels = readFiles("train.jsonl")
test_texts, test_labels = readFiles("test.jsonl")
test2_texts, test2_labels = readFiles("test2.jsonl")



train_1, train_2, train_s, train_e = pairGen(train_texts)
test_1, test_2, test_s, test_e = pairGen(test_texts)
test2_1, test2_2, test_2_s, test_2_e = pairGen(test2_texts)
"""
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
"""
print("finish splitting train & val set.")
print("tokenizing...")
"""
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
"""
tokenizer_pair = BertTokenizerFast.from_pretrained('bert-base-uncased')
"""
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test2_encodings = tokenizer(test2_texts, truncation=True, padding=True)
"""

print("tok train.")
train_pair_encodings = tokenizer_pair(train_1, train_2, truncation=True, padding=True)
train_pair_encodings_full = {
    "encodings" : train_pair_encodings,
    "s": train_s,
    "e": train_e
}
savePickle(train_pair_encodings_full, "train_pair_encodings_full.pkl")

print("tok test.")
test_pair_encodings = tokenizer_pair(test_1, test_2, truncation=True, padding=True)
test_pair_encodings_full = {
    "encodings" : test_pair_encodings,
    "s": test_s,
    "e": test_e
}

print("tok test2.")
test2_pair_encodings = tokenizer_pair(test2_1, test2_2, truncation=True, padding=True)
test2_pair_encodings_full = {
    "encodings" : test2_pair_encodings,
    "s": test_2_s,
    "e": test_2_e
}
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
savePickle(test2_labels, "test2_labels.pkl")
savePickle(train_encodings, "train_encodings.pkl")
savePickle(val_encodings, "val_encodings.pkl")
savePickle(test_encodings, "test_encodings.pkl")
savePickle(test2_encodings, "test2_encodings.pkl")

"""

savePickle(test_pair_encodings_full, "test_pair_encodings_full.pkl")
savePickle(test2_pair_encodings_full, "test2_pair_encodings_full.pkl")