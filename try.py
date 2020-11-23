import pickle

with open("val_encodings.pkl", 'rb') as f:
    x = pickle.load(f)
print(x.input_ids)
print(type(x))
print(type(x[0]))
print(x.keys())
print(x[0])
print(x[0].__dict__)
print(x[0].ids)
print(x[0].type_ids)
print(x[0].tokens)

print(type(x[1]))
print(x[1])
print(x[1].__dict__)
print(len(x))

print(x[:5])