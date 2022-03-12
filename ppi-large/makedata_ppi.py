import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import json

feat = np.load("feats.npy")

class_map = json.load(open("class_map.json"))

# For MultiLabel Problems
labels = np.zeros((len(class_map), 121))

for key in class_map:
    labels[int(key)] = class_map[key]

np.save("labels.npy", labels)

id_map = json.load(open("role.json"))

train_ix = sorted(id_map['tr'])
valid_ix = sorted(id_map['va'])
test_ix = sorted(id_map['te'])

np.save("TrainSortedIndex.npy", train_ix)

np.save("ValidSortedIndex.npy", valid_ix)

np.save("TestSortedIndex.npy", test_ix)

scaler = StandardScaler()

scaler.fit(feat[train_ix])

feat_data = scaler.transform(feat)
np.save("feat_preprocessed.npy", feat_data)

