# -*- encoding: utf-8 -*-
"""
@File    :   methons.py
@Time    :   2021/08/21 20:49:34
@Author  :   Huang Mengjie 
@Version :   1.0
@Email   :   huangmj6016@foxmail.com
@Desc    :   None
"""

from tools import distance, entropy, combin_rule
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import wasserstein_distance


def base_methon(BPA, target_set):
    """
    基础方法，直接融合BPA
    """
    RSTs = []
    RST = BPA[0]
    RSTs.append(RST)
    for row in BPA[1:]:
        RST = combin_rule.dempster_combin(RST, row, target_set)
        RST = RST / RST.sum()  # 防止多次迭代后的精度误差
        RSTs.append(RST)
    return RST, np.stack(RSTs, 0)


jousselme_matrix_dict = {}


def methon_qky(BPA, target_set):
    """
    使用D矩阵计算相似度与冲突度
    使用Deng熵计算信息熵
    """

    if str(target_set) not in jousselme_matrix_dict:
        jousselme_matrix_dict[str(target_set)] = distance.jousselme_matrix(target_set)
    sim_matrix = np.zeros((BPA.shape[0], BPA.shape[0]), float)
    for i in range(BPA.shape[0]):
        for j in range(i, BPA.shape[0]):
            if i == j:
                continue
            sim_matrix[j, i] = sim_matrix[i, j] = 1 - distance.jousselme_distance(
                BPA[i], BPA[j], jousselme_matrix_dict[str(target_set)]
            )
    sups = sim_matrix.sum(1).squeeze()
    Fs = get_sorting_factor(BPA, target_set)
    msups = sups * Fs / (sups * Fs).sum()
    # msups = sups
    Ens = entropy.den_entropy(BPA, target_set)
    ucs = msups - msups.mean()
    gsups = msups * ((Ens / Ens.sum()) ** (-ucs))
    ws = gsups / gsups.sum()
    MAE = (BPA.T * ws).sum(1)
    RSTs = []
    RST = MAE[:]
    RSTs.append(RST)
    for i in range(BPA.shape[0] - 1):
        RST = combin_rule.dempster_combin(MAE, RST, target_set)
        RST = RST / RST.sum()  # 防止多次迭代后的精度误差
        RSTs.append(RST)
    return RST, np.stack(RSTs, 0)


def methon_mine(BPA, target_set):
    """
    使用D矩阵计算相似度与冲突度
    使用Deng熵计算信息熵
    """

    # def get_sorting_factor():  # [{"A"}, {"B"}, {"C"}, {"A", "B"}]
    #     pass

    if str(target_set) not in jousselme_matrix_dict:
        jousselme_matrix_dict[str(target_set)] = distance.jousselme_matrix(target_set)
    sim_matrix = np.zeros((BPA.shape[0], BPA.shape[0]), float)
    for i in range(BPA.shape[0]):
        for j in range(i, BPA.shape[0]):
            if i == j:
                continue
            sim_matrix[j, i] = sim_matrix[i, j] = 1 - distance.jousselme_distance(
                BPA[i], BPA[j], jousselme_matrix_dict[str(target_set)]
            )
    sups = sim_matrix.sum(1).squeeze()
    ######################################################
    #       排序因子修正基于Jousselme距离的支持度，         #
    #       排序因子直接与证据相乘，再归一化                #
    #       Fs = get_sorting_factor(BPA,target_set)      #
    #       msups = sups * Fs / (sups * Fs).sum()        #
    #       1、改可信度与排序因子组合规则                   #
    #       2、改排序因子生成方法 SM                       #
    #           2.1、证据与平均证据的W距离                 #
    #           2.2、证据与默认排序证据的W距离             #
    #           7/28，6/28，5/28，4/28，3/28，2/28，1/28  #
    #           2.3、与绝对平均征集的W距离                 #
    #       3、改熵和可信度的组合规则                      #
    ######################################################
    # 1、改组合规则
    # Fs = get_mine_factor(BPA, target_set)
    Fs = get_mine_factor(BPA)
    # Fs = Fs / Fs.shape[0]
    # msups = sups * np.exp(Fs)
    msups = sups * Fs / (sups * Fs).sum()
    # 2、改排序因子
    Fs2 = get_sorting_factor(BPA, target_set)
    msups = msups * Fs2 / (msups * Fs2).sum()

    Ens = entropy.den_entropy(BPA, target_set)

    ucs = msups - msups.mean()
    # ucs = msups * msups.shape[0]
    gsups = msups * ((Ens / Ens.sum()) ** (-ucs))
    # gsups = msups * ((Ens.sum() / Ens) * ucs)

    ws = gsups / gsups.sum()
    MAE = (BPA.T * ws).sum(1)
    RSTs = []
    RST = MAE[:]
    RSTs.append(RST)
    for i in range(BPA.shape[0] - 1):
        RST = combin_rule.dempster_combin(MAE, RST, target_set)
        RST = RST / RST.sum()  # 防止多次迭代后的精度误差
        RSTs.append(RST)
    return RST, np.stack(RSTs, 0)


def get_mine_factor(BPA):
    # BPA_mean = BPA.mean(0)
    BPA_mean = np.asarray([1 / BPA.shape[1]] * BPA.shape[1])
    dist_list = []
    for m in BPA:
        dist_list.append(wasserstein_distance(m, BPA_mean))
    dist_list = np.asarray(dist_list)
    dist_list = dist_list.shape[0] * dist_list / dist_list.sum()
    return dist_list
    # cov_mat = np.cov(BPA.T)
    # mvn(BPA_mean, cov_mat)
    # mvn.cdf(BPA[0])


def get_sorting_factor(BPA, target_set):  # [{"A"}, {"B"}, {"C"}, {"A", "B"}]
    """
    by Paper: An improvement for combination rule in evidence theory
    """
    SMs = []
    for _m in BPA:
        m = np.copy(_m)
        SM = np.zeros(shape=(len(m), len(m)))
        F = target_set
        J = []
        for idx, value in enumerate(m):
            m[idx] = value / len(target_set[idx])
        m = m / m.sum()
        while True:
            i = np.argmax(m)
            if m[i] == -np.Inf:
                break
            J.append(target_set[i])
            m[i] = -np.Inf
        for _set in target_set:
            if _set in F and _set in J:
                SM[F.index(_set), J.index(_set)] = 1
        SMs.append(SM)
    SMs = np.stack(SMs, 0)
    DMs = SMs - np.sum(SMs, axis=0) / BPA.shape[0]
    # DM_abs = -np.abs(DMs).sum(axis=1).sum(axis=1)
    DM_abs = -(DMs ** 2).sum(axis=1).sum(axis=1)
    Fs = BPA.shape[0] * np.exp(DM_abs) / np.exp(DM_abs).sum()
    return Fs


def debug():
    # please input you test code
    BPA = np.asarray(
        [
            [0.55, 0.25, 0.12, 0.08],
            [0, 0.5, 0.2, 0.3],
            [0.49, 0.31, 0.1, 0.1],
            [0.7, 0.21, 0.06, 0.03],
            [0.82, 0.09, 0.05, 0.04],
            [0.65, 0.30, 0.03, 0.02],
        ]
    )
    combin_set = [{"A"}, {"B"}, {"C"}, {"A", "B"}]
    # RST, RSTs = base_methon(BPA, combin_set)
    print(BPA)
    print(get_sorting_factor(BPA, combin_set))
    print(get_mine_factor(BPA))
