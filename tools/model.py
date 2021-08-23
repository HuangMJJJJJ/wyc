# -*- encoding: utf-8 -*-
"""
@File    :   model.py
@Time    :   2021/08/21 20:11:55
@Author  :   Huang Mengjie 
@Version :   1.0
@Email   :   huangmj6016@foxmail.com
@Desc    :   None
"""

# here put the import lib
import sys
from typing import Any, Union

if sys.version_info >= (3, 8):
    from typing import Literal
from functools import reduce
import numpy as np
import pandas as pd


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

    def __compute_interval_distance(self, interval1: Interval, interval2: Interval):
        if interval1.is_empty or interval2.is_empty:
            return np.Infinity
        else:
            return np.sqrt(
                (interval1.mid - interval2.mid) ** 2
                + (interval1.length - interval2.length) ** 2 / 12
            )

    def get_BPA(self, feature, alpha=5, Lambda=0.00001) -> np.ndarray:
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
        K = BPA.shape[0]
        return (BPA + Lambda) / (BPA.sum(axis=0) + K * Lambda)

    def predict(self, feature, methon, alpha=5):
        BPA = self.get_BPA(feature, alpha)
        BPA = BPA.T
        return methon(BPA, self.comb_target_set)
