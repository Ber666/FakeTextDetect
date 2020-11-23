import pickle
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def loadPickle(path):
    with open(path, 'rb') as f:
        x = pickle.load(f)
    return x


def binary_accuracy(predictions, labels):
    # predictions are logits [batch_size, 2]
    # labels [batch_size]
    correct = 0
    for i in range(predictions.shape[0]):
        correct += 1 if predictions[i][labels[i]] > predictions[i][1-labels[i]] else 0

    return correct

def evaluate(model, dataset, labels):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
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
            loss = outputs[0].item()
            acc = binary_accuracy(outputs[1], labels)
            epoch_loss += loss
            epoch_acc += acc

    return epoch_loss / len(dataset), epoch_acc / len(dataset)