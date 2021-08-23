# -*- encoding: utf-8 -*-
"""
@File    :   distance.py
@Time    :   2021/08/21 19:18:25
@Author  :   Huang Mengjie 
@Version :   1.0
@Email   :   huangmj6016@foxmail.com
@Desc    :   实现证据理论中常用的距离公式
"""

import numpy as np
from scipy.stats import wasserstein_distance as w_distance


def jousselme_matrix(target_set):
    """
    获取用于计算Jousselme距离的矩阵
    """
    matrix_D = np.ones(shape=(len(target_set), len(target_set)))
    for i in range(matrix_D.shape[0]):
        for j in range(i + 1, matrix_D.shape[1]):
            matrix_D[i, j] = matrix_D[j, i] = len(
                target_set[i].intersection(target_set[j])
            ) / len(target_set[i].union(target_set[j]))
    return matrix_D


def jousselme_distance(m1, m2, matrix_D):
    """
    Jousselme距离实现
    """
    m1, m2 = np.asarray(m1), np.asarray(m2)
    assert m1.shape == m2.shape, "Arguments' Shape Diffirent"
    m1, m2 = m1.reshape((-1, m1.shape[-1])), m2.reshape((-1, m2.shape[-1]))
    shape = list(m1.shape)
    delta_m = m1 - m2
    result = np.asarray(
        [delta_m[i] @ matrix_D @ delta_m[i].T for i in range(delta_m.shape[0])]
    )
    shape[-1] = 1
    return result.reshape(shape).squeeze()


def wasserstein_distance(m1, m2):
    m1, m2 = np.asarray(m1), np.asarray(m2)
    assert m1.shape == m2.shape, "Arguments' Shape Diffirent"
    m1, m2 = m1.reshape((-1, m1.shape[-1])), m2.reshape((-1, m2.shape[-1]))
    shape = list(m1.shape)
    result = np.asarray([w_distance(m1[i], m2[i]) for i in range(m1.shape[0])])
    shape[-1] = 1
    return result.reshape(shape).squeeze()


if __name__ == "__main__":
    # please input you test code
    print(
        jousselme_distance(
            [1, 2, 3, 4], [2, 3, 4, 5], jousselme_matrix([{1}, {2}, {1, 2}, {2, 3}])
        )
    )
