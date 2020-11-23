import jsonlines

test = []
train = []
with open("data/test1.txt", 'r', encoding='utf-8') as f:
    test = f.readlines()

with open("data/train.txt", 'r', encoding='utf-8') as f:
    train = f.readlines()

print("finish reading.")
labels = ['neg', 'pos']
testls, trainls = [], []

for ins in test:
    label = eval(ins[-2])
    assert label in [0, 1]
    content = ins[:-3]
    testls.append({'text':content, 'label': labels[label]})

print("finish test set.")

for ins in train:
    label = eval(ins[-2])
    assert label in [0, 1]
    content = ins[:-3]
    trainls.append({'text':content, 'label': labels[label]})

with jsonlines.open('train.jsonl', mode='w') as writer:
    for i in trainls:
        writer.write(i)
with jsonlines.open('test.jsonl', mode='w') as writer:
    for i in testls:
        writer.write(i)


"""
with open('test.jsonl', mode='w', encoding='utf-8') as writer:
    writer.writelines(testls)
with open('train.jsonl', mode='w', encoding='utf-8') as writer:
    writer.writelines(trainls)"""