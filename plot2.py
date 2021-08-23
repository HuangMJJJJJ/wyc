import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base = np.load(r"output\experiment2\base_results.npy")
qky = np.load(r"output\experiment2\methon_qky_results.npy")
mine = np.load(r"output\experiment2\mine_results_1.npy")
mine2 = np.load(r"output\experiment2\mine_results_2.npy")
cls = (
    ["base"] * base.shape[0]
    + ["qky"] * qky.shape[0]
    + ["mine"] * mine.shape[0]
    + ["mine2"] * mine2.shape[0]
)
df = pd.DataFrame(
    np.concatenate([base, qky, mine, mine2]),
    columns=["train size", "accuracy score", "f1 score"],
)
df["cls"] = cls
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
axs = axs.flatten()
ax = sns.lineplot(x="train size", y="accuracy score", hue="cls", data=df, ax=axs[0])
ax.set_ylim([0.8, 1.0])
ax = sns.lineplot(x="train size", y="f1 score", hue="cls", data=df, ax=axs[1])
ax.set_ylim([0.8, 1.0])
plt.show()
