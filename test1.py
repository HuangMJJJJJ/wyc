import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

sns.color_palette("tab10")


def methon3(BPA, comb_target_set):
    """
    使用W距离计算相似度与冲突度
    使用Deng熵计算信息熵
    """
    max_distance = wasserstein_distance(np.zeros(BPA.shape[1]), np.ones(BPA.shape[1]))
    sim_matrix = np.zeros((BPA.shape[0], BPA.shape[0]), float)
    for i in range(BPA.shape[0]):
        for j in range(i, BPA.shape[0]):
            if i == j:
                continue
            sim_matrix[i, j] = wasserstein_distance(BPA[i], BPA[j])
            sim_matrix[j, i] = sim_matrix[i, j]
    sups = sim_matrix.sum(1).squeeze()
    Fs = get_sorting_factor(BPA, comb_target_set)
    msups = sups * Fs / (sups * Fs).sum()
    Ens = np.asarray([Deng_entropy(m, comb_target_set) for m in BPA])
    ucs = -msups + msups.mean()
    gsups = msups * ((Ens / Ens.sum()) ** ucs)
    ws = gsups / gsups.sum()
    MAE = (BPA.T * ws).sum(1)
    MAE = MAE / MAE.sum()
    RST = MAE[:]
    RSTs = np.zeros((BPA.shape[0] - 1, RST.shape[0]), float)
    for i in range(BPA.shape[0] - 1):
        RSTs[i] = RST
        RST = Dempster_combin(MAE, RST, comb_target_set)
        RST = RST / RST.sum()
    return RST, RSTs


def get_sorting_factor(BPA, combin_target_set):  # [{"A"}, {"B"}, {"C"}, {"A", "B"}]
    """
    by Paper: An improvement for combination rule in evidence theory
    """
    SMs = []
    for _m in BPA:
        m = np.copy(_m)
        SM = np.zeros(shape=(len(m), len(m)))
        F = combin_target_set
        J = []
        for idx, value in enumerate(m):
            m[idx] = value / len(combin_target_set[idx])
        while True:
            i = np.argmax(m)
            if m[i] == -np.Inf:
                break
            J.append(combin_target_set[i])
            m[i] = -np.Inf
        for _set in combin_target_set:
            if _set in F and _set in J:
                SM[F.index(_set), J.index(_set)] = 1
        SMs.append(SM)
    SMs = np.stack(SMs, 0)
    DMs = SMs - np.sum(SMs, axis=0) / BPA.shape[0]
    DM_abs = -np.abs(DMs).sum(axis=1).sum(axis=1)
    Fs = len(combin_target_set) * np.exp(DM_abs) / np.exp(DM_abs).sum()
    return Fs


# def get_sorting_factor(BPA):
#     SMs = np.zeros(shape=(BPA.shape[0], BPA.shape[1], BPA.shape[1]), dtype=float)
#     for index, m in enumerate(BPA):
#         J = np.argsort(m, kind="mergesort")
#         zero_end_idx = 0
#         for idx, j in enumerate(J):
#             if m[j] != 0:
#                 zero_end_idx = idx
#                 break
#         J[:zero_end_idx] = J[:zero_end_idx][::-1]
#         J = J[::-1]
#         for i, j in enumerate(J):
#             SMs[index, i, j] = 1
#     DMs = SMs - SMs.sum(axis=0) / SMs.shape[0]
#     DMs = np.abs(DMs).sum(1).sum(1)
#     Fs = DMs.shape[0] * np.exp(-1 * DMs) / np.exp(-1 * DMs).sum()
#     return Fs


def Deng_entropy(m, comb_target_set):
    En = [
        mA * np.log2(mA / (2 ** len(setA) - 1))
        for mA, setA in zip(m, comb_target_set)
        if mA != 0
    ]
    return sum(En)


def Dempster_combin(m1, m2, comb_target_set):
    new_mass = {str(s): 0 for s in comb_target_set}
    K = 0
    for idx1, set1 in enumerate(comb_target_set):
        for idx2, set2 in enumerate(comb_target_set):
            intersection_set = set1.intersection(set2)
            if len(intersection_set) == 0:
                K += m1[idx1] * m2[idx2]
            else:
                new_mass[str(intersection_set)] += m1[idx1] * m2[idx2]
    return np.asarray([i / (1 - K) for i in new_mass.values()])


def methon1(BPA, comb_target_set):
    """
    使用D矩阵计算相似度与冲突度
    使用Deng熵计算信息熵
    """
    martrixD = get_matrix_D(comb_target_set)
    sim_matrix = np.zeros((BPA.shape[0], BPA.shape[0]), float)
    for i in range(BPA.shape[0]):
        for j in range(i, BPA.shape[0]):
            if i == j:
                continue
            sim_matrix[i, j] = sim_by_martix_D(BPA[i], BPA[j], martrixD)
            sim_matrix[j, i] = sim_matrix[i, j]
    sups = sim_matrix.sum(1).squeeze()
    Fs = get_sorting_factor(BPA, comb_target_set)
    msups = sups * Fs / (sups * Fs).sum()
    # msups = sups
    Ens = np.asarray([Deng_entropy(m, comb_target_set) for m in BPA])
    ucs = -msups + msups.mean()
    gsups = msups * ((Ens / Ens.sum()) ** ucs)
    ws = gsups / gsups.sum()
    MAE = (BPA.T * ws).sum(1)
    RST = MAE[:]
    RSTs = np.zeros((BPA.shape[0] - 1, RST.shape[0]), float)
    for i in range(BPA.shape[0] - 1):
        RSTs[i] = RST
        RST = Dempster_combin(MAE, RST, comb_target_set)
        RST = RST / RST.sum()
    return RST, RSTs


def sim_by_martix_D(m1, m2, martrixD):
    subm = m1 - m2
    return 1 - np.sqrt((subm @ martrixD @ subm) / 2)


def get_matrix_D(comb_target_set):
    set_length = len(comb_target_set)
    D = np.zeros(shape=(set_length, set_length), dtype=float)
    for idx1, set1 in enumerate(comb_target_set):
        for idx2, set2 in enumerate(comb_target_set):
            D[idx1, idx2] = len(set1.intersection(set2)) / len(set1.union(set2))
    return D


def base_methon(BPA, comb_target_set):
    """
    基础方法，直接融合BPA
    """
    RST = BPA[0, :]
    RSTs = np.zeros((BPA.shape[0] - 1, RST.shape[0]), float)
    for i, row in enumerate(BPA[1:]):
        RSTs[i] = RST
        RST = Dempster_combin(RST, row, comb_target_set)
        RST = RST / RST.sum()
    return RST, RSTs


if __name__ == "__main__":
    # please input you test code

    df = pd.read_excel("实验一数据.xlsx")
    BPA = df.iloc[:, 1:].to_numpy()
    Theta = [{1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}]
    result1, results1 = base_methon(BPA, Theta)
    result2, results2 = methon1(BPA, Theta)
    result3, results3 = methon3(BPA, Theta)
    results = pd.DataFrame(
        data=np.vstack([results1, results2, results3]), columns=df.columns[1:]
    )
    labals = ["base methon"] * 39 + ["methon1"] * 39 + ["methon2"] * 39
    results["labels"] = labals
    results["idx"] = [i for i in range(1, 40)] * 3
    # plt.figure(figsize=(19, 10), dpi=100)
    fig, subs = plt.subplots(3, 3,  sharex=True)
    # plt.subplots_adjust(hspace=0)
    subs = subs.flatten()
    sns.lineplot(
        x="idx",
        y="{θ1}",
        data=results,
        hue="labels",
        ax=subs[0],
    ).set_ylim(0, 1)
    sns.lineplot(
        x="idx",
        y="{θ2}",
        data=results,
        hue="labels",
        ax=subs[1],
    ).set_ylim(0, 1)

    sns.lineplot(
        x="idx",
        y="{θ3}",
        data=results,
        hue="labels",
        ax=subs[2],
    ).set_ylim(0, 1)

    sns.lineplot(
        x="idx",
        y="{θ1,θ2}",
        data=results,
        hue="labels",
        ax=subs[3],
    ).set_ylim(0, 1)

    sns.lineplot(
        x="idx",
        y="{θ1,θ3}",
        data=results,
        hue="labels",
        ax=subs[4],
    ).set_ylim(0, 1)

    sns.lineplot(
        x="idx",
        y="{θ2,θ3}",
        data=results,
        hue="labels",
        ax=subs[5],
    ).set_ylim(0, 1)

    sns.lineplot(
        x="idx",
        y="{θ1,θ2,θ3}",
        data=results,
        hue="labels",
        ax=subs[6],
    ).set_ylim(0, 1)

    plt.ylim(0, 1)
    plt.show()
