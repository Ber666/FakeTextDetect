import torch
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import pickle
from tensorboardX import SummaryWriter
from tqdm import tqdm
from util_bert import evaluate, binary_accuracy
import json

class DMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def loadPickle(path):
    with open(path, 'rb') as f:
        x = pickle.load(f)
    return x



print("loading data...")
train_labels = loadPickle("train_labels.pkl")
val_labels = loadPickle("val_labels.pkl")
test_labels = loadPickle("test_labels.pkl")
train_encodings = loadPickle("train_encodings.pkl")
val_encodings = loadPickle("val_encodings.pkl")
test_encodings = loadPickle("test_encodings.pkl")

print("building dataset...")
train_dataset = DMDataset(train_encodings, train_labels)
val_dataset = DMDataset(val_encodings, val_labels)
test_dataset = DMDataset(test_encodings, test_labels)


print("building model...")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = 8
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

print("start training.")

train_log = {'Train/Loss':[], 'Train/Acc':[], 'Val/Loss':[], 'Val/Acc':[], 'Save':[]}
writer = SummaryWriter('log')
best_val_acc = 0
for epoch in range(3):
    loss_accum = 0
    correct_accum = 0
    ins_accum = 0
    for niter, batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # print(input_ids, outputs)
        loss = outputs[0]
        loss.backward()
        optim.step()
        loss_accum += loss.item()
        correct_accum += binary_accuracy(outputs[1], labels)
        ins_accum += BATCH_SIZE


        if niter % 1000 == 0:
            print("experiment on val...")
            model.eval()
            val_loss, val_acc = evaluate(model, val_dataset, val_labels)
            model.train()
            train_log['Val/Loss'].append((epoch, niter, val_loss))
            train_log['Val/Acc'].append((epoch, niter, val_acc))
            print(train_log['Val/Loss'][-1])
            print(train_log['Val/Acc'][-1])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                state = {'net':model.state_dict(), 'optimizer':optim.state_dict(), 'epoch':epoch}
                train_log['Save'].append((epoch, niter, best_val_acc))
                print('saving best model at epoch {} iter {}'.format(epoch, niter))
                torch.save(state, 'state_bert_base_best.pth')

        if niter % 100 == 0:
            # print(niter, input_ids, outputs, labels)
            train_log['Train/Loss'].append((epoch, niter, loss_accum/ins_accum))
            train_log['Train/Acc'].append((epoch, niter, correct_accum/ins_accum))
            print(train_log['Train/Loss'][-1])
            print(train_log['Train/Acc'][-1])
            writer.add_scalar('Train/Loss', loss_accum, niter)
            writer.add_scalar('Train/Acc', correct_accum/ins_accum, niter)
            loss_accum = 0
            correct_accum = 0
            ins_accum = 0
            
            with open("log_bert_epoch_{}.json".format(epoch), 'w', encoding='utf-8') as f:
                json.dump(train_log, f)

            
"""
with open("log.json", 'w', encoding='utf-8') as f:
    json.dump(train_log, f)"""

model.eval()