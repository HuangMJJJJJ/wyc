import numpy as np


def get_sorting_factor(BPA, combin_target_set):  # [{"A"}, {"B"}, {"C"}, {"A", "B"}]
    """
    by Paper: An improvement for combination rule in evidence theory
    """
    SMs = []
    for m in BPA:
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
print(get_sorting_factor(BPA, combin_set))
# print: 0.79922827 0.00385863 0.79922827 0.79922827 0.79922827 0.79922827
# paper result: 1.1288 0.3558 1.1288 1.1288 1.1288 1.1288
# data from paper: 降低相似度碰撞的证据融合方法
