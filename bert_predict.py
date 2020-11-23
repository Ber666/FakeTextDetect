#https://github.com/huggingface/transformers/issues/4689
import torch
from tqdm import tqdm
from util_bert import loadPickle
from transformers import DistilBertForSequenceClassification
from dataset_bert import DMDataset
from torch.utils.data import DataLoader

def predictDataset(model, dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    predict = []
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # labels = batch['labels'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            #print(input_ids)
            #print(outputs)
            predict += [(0 if i[0] > i[1] else 1) for i in outputs[1]]
            a = 0
    return predict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("loading data...")
test_encodings = loadPickle("test_encodings.pkl")


test_labels = loadPickle("test_labels.pkl")
#test_labels = [0] * len(test_encodings['input_ids'])
print(len(test_labels))
test_dataset = DMDataset(test_encodings, test_labels)
print("loading model...")

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
state = torch.load('state_distilled_bert_base_epoch0.pth')
model.load_state_dict(state['net'])
model.to(device)
model.eval()
print("predicting...")
predictions = predictDataset(model, test_dataset)
# .8178 test set 

# .797 epoch2
# best model 7937

test_labels = loadPickle("test_labels.pkl")
print(len(test_labels))
print(len(predictions))
tmp = 0
for t, p in zip(test_labels, predictions):
    tmp += (1 if t == p else 0)
print(tmp/len(test_labels))

# state_distilled_bert_base_epoch0
# with 0 as gold label -> 0.5018
# with true as gold label -> 0.4984