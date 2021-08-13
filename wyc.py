# -*- encoding: utf-8 -*-
"""
@File    :   wyc.py
@Time    :   2021/08/11 21:55:50
@Author  :   Huang Mengjie
@Version :   1.0
@Email   :   huangmj6016@foxmail.com
@Desc    :   None
"""
import sys
from typing import Any, Union

from scipy.sparse.extract import find

if sys.version_info >= (3, 8):
    from typing import Literal
from functools import reduce
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import wasserstein_distance
from tqdm import tqdm


class Interval(pd.Interval):
    def __init__(
        self,
        left: object,
        right: object,
        closed: Union[str, Literal["left", "right", "both", "neither"]] = ...,
    ):
        super(Interval, self).__init__(left, right, closed)

    def __mul__(self, y):
        if self.overlaps(y):
            return Interval(
                max(self.left, y.left), min(self.right, y.right), closed="both"
            )
        else:
            return Interval(0, 0, closed="neither")

    def __str__(self) -> str:
        if self.closed_left:
            left = "["
        else:
            left = "("
        if self.closed_right:
            right = "]"
        else:
            right = ")"
        return (
            left
            + "{:.2f}".format(self.left)
            + ", "
            + "{:.2f}".format(self.right)
            + right
        )

    def __repr__(self) -> str:
        if self.closed_left:
            left = "["
        else:
            left = "("
        if self.closed_right:
            right = "]"
        else:
            right = ")"
        return (
            left
            + "{:.2f}".format(self.left)
            + ", "
            + "{:.2f}".format(self.right)
            + right
        )


class DSmodel:
    def __init__(self, feature, target, feature_names=None, target_names=None):
        self.feature = feature
        self.target = target
        self.feature_code = [i for i in range(feature.shape[1])]
        self.target_code = [i for i in range(len(set(target)))]
        self.feature_names = feature_names if feature_names else self.feature_code
        self.target_names = target_names if target_names else self.target_code
        self.interval_table, self.comb_target_set = self.__get_interval_model()
        self.interval_table_array = np.asanyarray(list(self.interval_table.values()))
        self.martrixD = self.__get_matrix_D()

    def get_interval_table(self):
        print("column: " + " ".join(self.feature_names))
        print(
            "row:\n\t"
            + "\n\t".join(
                [
                    ",".join([self.target_names[i] for i in s])
                    for s in self.comb_target_set
                ]
            )
        )

        print(
            "\n".join(
                [
                    " ".join([str(item) for item in row])
                    for row in self.interval_table_array
                ]
            )
        )

    def __get_interval_model(self):
        target_dict = {str(i): [] for i in self.target_code}

        for index, label in enumerate(self.target):
            target_dict[str(label)].append(index)
        for key, idx in target_dict.items():
            target_dict[key] = [
                Interval(self.feature[idx, i].min(), self.feature[idx, i].max(), "both")
                for i in range(len(self.feature_code))
            ]
        target_idx = [{i} for i in self.target_code]
        target_idx_compute = np.asanyarray(target_idx)
        for i in range(2 ** len(self.target_code)):
            bin_str = "{0:b}".format(i).zfill(len(self.target_code))
            choose_row = np.asarray([int(i) for i in bin_str])
            idx = np.argwhere(choose_row == 1)
            idx = idx.squeeze()
            if idx.size < 2:
                continue
            comb_row = reduce(lambda x, y: x.union(y), target_idx_compute[idx])
            target_idx.append(comb_row)
            ditc_keys = [str(i) for i in comb_row]
            target_dict["".join(ditc_keys)] = [
                reduce(lambda x, y: x * y, [target_dict[key][i] for key in ditc_keys])
                for i in range(len(self.feature_code))
            ]
        return target_dict, target_idx

    def __get_matrix_D(self):
        set_length = len(self.comb_target_set)
        D = np.zeros(shape=(set_length, set_length), dtype=float)
        for idx1, set1 in enumerate(self.comb_target_set):
            for idx2, set2 in enumerate(self.comb_target_set):
                D[idx1, idx2] = len(set1.intersection(set2)) / len(set1.union(set2))
        return D

    def base_methon(self, BPA):
        """
        基础方法，直接融合BPA
        """
        RST = BPA[0]
        for row in BPA[1:]:
            RST = self.__Dempster_combin(RST, row)
        return RST

    def methon1(self, BPA):
        """
        使用D矩阵计算相似度与冲突度
        使用Deng熵计算信息熵
        """
        sim_matrix = np.zeros((BPA.shape[0], BPA.shape[0]), float)
        for i in range(BPA.shape[0]):
            for j in range(i, BPA.shape[0]):
                if i == j:
                    continue
                sim_matrix[i, j] = self.__sim_by_martix_D(BPA[i], BPA[j])
                sim_matrix[j, i] = sim_matrix[i, j]
        sups = sim_matrix.sum(1).squeeze()
        Fs = self.__get_sorting_factor(BPA)
        msups = sups * Fs / (sups * Fs).sum()
        # msups = sups
        Ens = np.asarray([self.__Deng_entropy(m) for m in BPA])
        ucs = -msups + msups.mean()
        gsups = msups * ((Ens / Ens.sum()) ** ucs)
        ws = gsups / gsups.sum()
        MAE = (BPA.T * ws).sum(1)
        RST = MAE[:]
        for i in range(BPA.shape[0]):
            RST = self.__Dempster_combin(MAE, RST)
        return RST

    def methon2(self, BPA):
        """
        使用余弦相似度计算相似度与冲突度
        使用Deng熵计算信息熵
        """
        sim_matrix = np.zeros((BPA.shape[0], BPA.shape[0]), float)
        for i in range(BPA.shape[0]):
            for j in range(i, BPA.shape[0]):
                if i == j:
                    continue
                # sim_matrix[i, j] = self.__sim_by_D(BPA[i], BPA[j])
                sim_matrix[i, j] = (
                    BPA[i] @ BPA[j] / (np.linalg.norm(BPA[i]) * np.linalg.norm(BPA[j]))
                )
                sim_matrix[j, i] = sim_matrix[i, j]
        sups = sim_matrix.sum(1).squeeze()
        Fs = self.__get_sorting_factor(BPA)
        msups = sups * Fs / (sups * Fs).sum()
        # msups = sups
        Ens = np.asarray([self.__Deng_entropy(m) for m in BPA])
        ucs = -msups + msups.mean()
        gsups = msups * ((Ens / Ens.sum()) ** ucs)
        ws = gsups / gsups.sum()
        MAE = (BPA.T * ws).sum(1)
        RST = MAE[:]
        for i in range(BPA.shape[0]):
            RST = self.__Dempster_combin(MAE, RST)
        return RST

    def methon3(self, BPA):
        """
        使用W距离计算相似度与冲突度
        使用Deng熵计算信息熵
        """
        max_distance = wasserstein_distance(
            np.zeros(BPA.shape[1]), np.ones(BPA.shape[1])
        )
        sim_matrix = np.zeros((BPA.shape[0], BPA.shape[0]), float)
        for i in range(BPA.shape[0]):
            for j in range(i, BPA.shape[0]):
                if i == j:
                    continue
                sim_matrix[i, j] = max_distance - wasserstein_distance(BPA[i], BPA[j])
                sim_matrix[j, i] = sim_matrix[i, j]
        sups = sim_matrix.sum(1).squeeze()
        Fs = self.__get_sorting_factor(BPA)
        msups = sups * Fs / (sups * Fs).sum()
        # msups = sups
        Ens = np.asarray([self.__Deng_entropy(m) for m in BPA])
        ucs = -msups + msups.mean()
        gsups = msups * ((Ens / Ens.sum()) ** ucs)
        ws = gsups / gsups.sum()
        MAE = (BPA.T * ws).sum(1)
        RST = MAE[:]
        for i in range(BPA.shape[0]):
            RST = self.__Dempster_combin(MAE, RST)
        return RST

    def __Dempster_combin(self, m1, m2):
        new_mass = {str(s): 0 for s in self.comb_target_set}
        K = 0
        for idx1, set1 in enumerate(self.comb_target_set):
            for idx2, set2 in enumerate(self.comb_target_set):
                intersection_set = set1.intersection(set2)
                if len(intersection_set) == 0:
                    K += m1[idx1] * m2[idx2]
                else:
                    new_mass[str(intersection_set)] += m1[idx1] * m2[idx2]
        return np.asarray([i / (1 - K) for i in new_mass.values()])

    def __Deng_entropy(self, m):
        En = [
            mA * np.log2(mA / (2 ** len(setA) - 1))
            for mA, setA in zip(m, self.comb_target_set)
            if mA != 0
        ]
        return sum(En)

    def __sim_by_martix_D(self, m1, m2):
        subm = m1 - m2
        return 1 - np.sqrt((subm @ self.martrixD @ subm) / 2)

    def __get_sorting_factor(self, BPA):
        SMs = np.zeros(shape=(BPA.shape[0], BPA.shape[1], BPA.shape[1]), dtype=float)
        for index, m in enumerate(BPA):
            # m = np.sort(m)[-1:]

            J = np.argsort(m, kind="mergesort")
            zero_end_idx = 0
            for idx, j in enumerate(J):
                if m[j] != 0:
                    zero_end_idx = idx
                    break
            J[:zero_end_idx] = J[:zero_end_idx][::-1]
            J = J[::-1]
            for i, j in enumerate(J):
                SMs[index, i, j] = 1
        DMs = SMs - SMs.sum(axis=0) / SMs.shape[0]
        DMs = np.abs(DMs).sum(1).sum(1)
        Fs = DMs.shape[0] * np.exp(-1 * DMs) / np.exp(-1 * DMs).sum()
        return Fs

    def predict(self, feature, alpha=5, methon=base_methon):
        BPA = self.get_BPA(feature, alpha)
        BPA = BPA.T
        return methon(self, BPA)

    def get_BPA(self, feature, alpha=5) -> np.ndarray:
        BPA = np.zeros_like(self.interval_table_array, dtype=float)
        feature = [Interval(i, i, "both") for i in feature]
        BPA = [
            [
                1 / (1 + alpha * self.__compute_interval_distance(i1, i2))
                for i1, i2 in zip(feature, row)
            ]
            for row in self.interval_table.values()
        ]
        BPA = np.asarray(BPA)
        return BPA / BPA.sum(axis=0)

    def __compute_interval_distance(self, interval1: Interval, interval2: Interval):
        if interval1.is_empty or interval2.is_empty:
            return np.Infinity
        else:
            return np.sqrt(
                (interval1.mid - interval2.mid) ** 2
                + (interval1.length - interval2.length) ** 2 / 12
            )


if __name__ == "__main__":
    # please input you test code
    raw_data = load_iris()
    features = raw_data.data
    targets = raw_data.target
    data_count = targets.size
    feature_names = raw_data.feature_names
    target_names = raw_data.target_names.tolist()
    methons = [DSmodel.base_methon, DSmodel.methon1, DSmodel.methon2, DSmodel.methon3]
    result_df = pd.DataFrame(
        columns=(
            "train ratio",
            "base acc ratio",
            "methon1 acc ratio",
            "methon2 acc ratio",
            "methon3 acc ratio",
        )
    )
    for train_size in tqdm(range(len(target_names), data_count - 1, len(target_names))):
        sss = StratifiedShuffleSplit(
            n_splits=100, train_size=train_size, random_state=42
        )
        for train_idx, test_idx in sss.split(features, targets):
            train_features, train_targets = features[train_idx], targets[train_idx]
            test_features, test_targets = features[test_idx], targets[test_idx]
            ds = DSmodel(train_features, train_targets, feature_names, target_names)
            acc_count, sum_count = np.zeros(len(methons)), 0
            for test_feature, test_target in zip(test_features, test_targets):
                sum_count += 1
                for idx, methon in enumerate(methons):
                    acc_count[idx] += (
                        1
                        if ds.predict(test_feature, methon=methon).argmax()
                        == test_target
                        else 0
                    )
            acc_count = acc_count / sum_count
            result_df = result_df.append(
                {
                    "train ratio": 1 - sum_count / data_count,
                    "base acc ratio": acc_count[0],
                    "methon1 acc ratio": acc_count[1],
                    "methon2 acc ratio": acc_count[2],
                    "methon3 acc ratio": acc_count[3],
                },
                ignore_index=True,
            )
    result_df.to_csv("result2.csv")
