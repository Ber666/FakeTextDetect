import torch
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

class DMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        # 2 keys: input_ids, attention_masks
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item = {input_ids, attention_masks}
        item['labels'] = torch.tensor(self.labels[idx])
        # 0/1
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
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

print("start training.")
for epoch in range(3):
    print("epoch {}".format(epoch))
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        state = {'net':model.state_dict(), 'optimizer':optim.state_dict(), 'epoch':epoch}

    torch.save(state, 'state_bert_base_epoch{}.pth'.format(epoch))
model.eval()