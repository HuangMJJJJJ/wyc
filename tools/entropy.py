# -*- encoding: utf-8 -*-
"""
@File    :   entropy.py
@Time    :   2021/08/21 18:50:50
@Author  :   Huang Mengjie 
@Version :   1.0
@Email   :   huangmj6016@foxmail.com
@Desc    :   实现常用的熵计算公式
"""
import numpy as np
from collections import Iterable


def base_entorpy(m: np.ndarray) -> np.ndarray:
    """
    基础信息熵实现
    """
    if m.ndim == 1:
        m = np.expand_dims(m, 0)
    shape = list(m.shape)
    m = m.reshape((-1, shape[-1]))
    logp = np.log2(m)
    for i, j in np.argwhere(logp == -np.Inf):
        logp[i, j] = 0
    logp.reshape(shape)
    return -(m * logp).sum(-1).squeeze()


np.ndarray


def den_entropy(m: np.ndarray, target_set: Iterable) -> np.ndarray:
    """
    Deng熵实现
    """
    assert m.shape[-1] == len(target_set), "列表与数据长度不一致"
    target_set = np.asarray([len(s) for s in target_set])
    if m.ndim == 1:
        m = np.expand_dims(m, 0)
    # shape = list(m.shape)
    # m = m.reshape((-1, shape[-1]))
    # logp = np.log2(m)
    # for i, j in np.argwhere(logp == -np.Inf):
    #     logp[i, j] = 0
    # logp.reshape(shape)
    # return -(m * logp / target_set).sum(-1).squeeze()
    return -(m * np.log2(m) / target_set).sum(-1).squeeze()


if __name__ == "__main__":

    # please input you test code
    # a = np.asarray([[0.2, 0.2, 0.3, 0.3], [0.1, 0.2, 0.3, 0.4]])
    a = np.asarray([[[0.2, 0.2, 0.3, 0.3], [0.1, 0.2, 0.3, 0.4]]])
    print(den_entropy(a, [{1}, {2}, {3}, {4, 5}]))
    print(base_entorpy(a))
