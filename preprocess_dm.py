import jsonlines

test = []
test2 = []
train = []

print("reading...")
with open("test2_no_lable.txt", 'r', encoding='utf-8') as f:
    test2 = f.readlines()

"""
with open("test1.txt", 'r', encoding='utf-8') as f:
    test = f.readlines()

with open("train.txt", 'r', encoding='utf-8') as f:
    train = f.readlines()
"""
print("processing...")
labels = ['neg', 'pos']
testls, trainls, test2ls = [], [], []
"""
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

print("finish train set.")"""

for ins in test2:
    label = eval(ins[-2])
    assert label in [0, 1]
    content = ins[:-3]
    test2ls.append({'text':content, 'label': labels[label]})

print("finish test2 set.")
print("writing to jsonlines..")
"""
with jsonlines.open('train.jsonl', mode='w') as writer:
    for i in trainls:
        writer.write(i)
with jsonlines.open('test.jsonl', mode='w') as writer:
    for i in testls:
        writer.write(i)"""
with jsonlines.open('test2.jsonl', mode='w') as writer:
    for i in test2ls:
        writer.write(i)
print("finish")

"""
with open('test.jsonl', mode='w', encoding='utf-8') as writer:
    writer.writelines(testls)
with open('train.jsonl', mode='w', encoding='utf-8') as writer:
    writer.writelines(trainls)"""