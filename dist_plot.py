import matplotlib.pyplot as plt
import json
logpath = 'log_distilled_bert_main_epoch_2.json'
with open(logpath, 'r', encoding='utf-8') as f:
    x = json.load(f)
"""

plt.plot([i[0] * 5000 + i[1] for i in x['Train/Loss']], [i[2] for i in x['Train/Loss']], label='Train/Loss')
plt.plot([i[0] * 5000 + i[1] for i in x['Val/Loss']], [i[2] for i in x['Val/Loss']], label='Val/Loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("filename.png")
"""
plt.plot([i[0] * 5000 + i[1] for i in x['Train/Acc']], [i[2] for i in x['Train/Acc']], label='Train/Acc')
plt.plot([i[0] * 5000 + i[1] for i in x['Val/Acc']], [i[2] for i in x['Val/Acc']], label='Val/Acc')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")


'''
    "Train/Loss"
    "Train/Acc"
    "Val/Loss"
    "Val/Acc"
'''