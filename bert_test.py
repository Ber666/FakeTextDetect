#https://github.com/huggingface/transformers/issues/4689
import torch
from tqdm import tqdm
from util_bert import loadPickle
from transformers import DistilBertForSequenceClassification
from dataset_bert import DMDataset
from torch.utils.data import DataLoader
def binary_accuracy(predictions, labels):
    # predictions are logits [batch_size, 2]
    # labels [batch_size]
    correct = 0
    for i in range(predictions.shape[0]):
        correct += 1 if predictions[i][labels[i]] > predictions[i][1-labels[i]] else 0

    return correct


def evaluate(model, dataset, labels):
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            #print(input_ids)
            #print(outputs)
            loss = outputs[0]
            acc = binary_accuracy(outputs[1], labels)
            epoch_loss += loss
            epoch_acc += acc

    return epoch_loss / len(dataset), epoch_acc / len(dataset)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("loading data...")
test_encodings = loadPickle("test_encodings.pkl")
test_labels = loadPickle("test_labels.pkl")
test_dataset = DMDataset(test_encodings, test_labels)
print("loading model...")

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
state = torch.load('state_distilled_bert_base_epoch0.pth')
model.load_state_dict(state['net'])
model.to(device)
model.eval()
print("evaluating...")
print(evaluate(model, test_dataset, test_labels))
# .8178 test set 

# .797 epoch2
# best model 7937