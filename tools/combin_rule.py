# -*- encoding: utf-8 -*-
"""
@File    :   ds_combin.py
@Time    :   2021/08/21 20:45:28
@Author  :   Huang Mengjie 
@Version :   1.0
@Email   :   huangmj6016@foxmail.com
@Desc    :   证据融合公式实现
"""
import numpy as np


def dempster_combin(m1, m2, target_set, Lambda=0.00001):
    new_mass = {str(s): 0 for s in target_set}
    K = 0
    for idx1, set1 in enumerate(target_set):
        for idx2, set2 in enumerate(target_set):
            intersection_set = set1.intersection(set2)
            if len(intersection_set) == 0:
                K += m1[idx1] * m2[idx2]
            else:
                new_mass[str(intersection_set)] += m1[idx1] * m2[idx2]
    # return np.asarray([i / (1 - K) for i in new_mass.values()])
    N = len(new_mass)
    return np.asarray([(i + Lambda) / (1 - K + N * Lambda) for i in new_mass.values()])
