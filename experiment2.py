from sklearn.datasets import load_iris
from tools import model, methons
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

raw_data = load_iris()
features = raw_data.data
targets = raw_data.target
data_count = targets.size
feature_names = raw_data.feature_names
target_names = raw_data.target_names.tolist()

# base_results = []
# for train_size in tqdm(range(len(target_names), data_count - 1, len(target_names))):
#     sss = StratifiedShuffleSplit(n_splits=100, train_size=train_size, random_state=42)
#     for train_idx, test_idx in sss.split(features, targets):
#         train_features, train_targets = features[train_idx], targets[train_idx]
#         test_features, test_targets = features[test_idx], targets[test_idx]
#         ds = model.DSmodel(train_features, train_targets, feature_names, target_names)
#         pred_targets = []
#         for test_feature in test_features:
#             rst, rsts = ds.predict(test_feature, methon=methons.base_methon)
#             pred_targets.append(rst.argmax())
#         base_results.append(
#             [
#                 len(train_targets),
#                 accuracy_score(pred_targets, test_targets),
#                 f1_score(pred_targets, test_targets, average="macro"),
#             ]
#         )
# base_results = np.asarray(base_results)
# np.save("output/experiment2/base_results", base_results)
# print(base_results)

# methon_qky_results = []
# for train_size in tqdm(range(len(target_names), data_count - 1, len(target_names))):
#     sss = StratifiedShuffleSplit(n_splits=100, train_size=train_size, random_state=42)
#     for train_idx, test_idx in sss.split(features, targets):
#         train_features, train_targets = features[train_idx], targets[train_idx]
#         test_features, test_targets = features[test_idx], targets[test_idx]
#         ds = model.DSmodel(train_features, train_targets, feature_names, target_names)
#         pred_targets = []
#         for test_feature in test_features:
#             rst, rsts = ds.predict(test_feature, methon=methons.methon_qky)
#             pred_targets.append(rst.argmax())
#         methon_qky_results.append(
#             [
#                 len(train_targets),
#                 accuracy_score(pred_targets, test_targets),
#                 f1_score(pred_targets, test_targets, average="macro"),
#             ]
#         )
# methon_qky_results = np.asarray(methon_qky_results)
# np.save("output/experiment2/methon_qky_results", methon_qky_results)
# print(methon_qky_results)

methon_mine_results = []
for train_size in tqdm(range(len(target_names), data_count - 1, len(target_names))):
    sss = StratifiedShuffleSplit(n_splits=100, train_size=train_size, random_state=42)
    for train_idx, test_idx in sss.split(features, targets):
        train_features, train_targets = features[train_idx], targets[train_idx]
        test_features, test_targets = features[test_idx], targets[test_idx]
        ds = model.DSmodel(train_features, train_targets, feature_names, target_names)
        pred_targets = []
        for test_feature in test_features:
            rst, rsts = ds.predict(test_feature, methon=methons.methon_mine)
            pred_targets.append(rst.argmax())
        methon_mine_results.append(
            [
                len(train_targets),
                accuracy_score(pred_targets, test_targets),
                f1_score(pred_targets, test_targets, average="macro"),
            ]
        )
methon_mine_results = np.asarray(methon_mine_results)
np.save("output/experiment2/mine_results_2", methon_mine_results)
print(methon_mine_results)
