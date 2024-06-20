import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning, message=".*use_inf_as_na.*")


def plot_results(df: pd.DataFrame, title: str, path: str = "plots") -> None:
    """Plot the dataframe results using white background.

    Recall and MRR plots will each be saved to png and eps files.

    """
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=True)
    models = list(df.Model.unique())
    past = sns.color_palette()
    palette = {}
    order = ["QA"]
    for i, m in enumerate(models):
        palette[m] = past[i]
        if m != "QA":
            order.append(m)
    palette["QA"] = "#54B6B8"
    # Plot recalls
    sns.lineplot(
        data=df,
        x="Iteration",
        y="True Hits",
        hue="Model",
        errorbar=None,
        palette=palette,
        style="Model",
        hue_order=order,
        style_order=order,
    )
    sns.despine()
    plt.title(title)
    file_title = title.replace(" ", "").replace("|", "-")
    plot_file = f"{path}/recall-{file_title}"
    plt.savefig(f"{plot_file}.png", dpi=600)
    plt.savefig(f"{plot_file}.pdf", dpi=1200)
    plt.close()
    print(f"Recall plots saved to {plot_file}.png/pdf.")

    # Plot mean reciprocal ranks, but first average the recall curves for each disease
    tdf = df.groupby(["Model", "Disease", "Iteration"]).mean().reset_index()
    tdf = tdf.groupby(["Model", "Disease", "Iteration"]).mean().reset_index()
    tdf["Rank"] = tdf.groupby(["Disease", "Iteration"]).rank(
        ascending=False,
        method="min",
    )["True Hits"]
    tdf["Mean Reciprocal Rank (avg)"] = 1 / tdf["Rank"]
    ax = sns.lineplot(
        data=tdf,
        x="Iteration",
        y="Mean Reciprocal Rank (avg)",
        hue="Model",
        errorbar=None,
        palette=palette,
        style="Model",
        hue_order=order,
        style_order=order,
    )
    sns.despine()
    sns.move_legend(ax, "upper right", title=None)
    plt.title(title)
    file_title = title.replace(" ", "").replace("|", "-")
    plot_file = f"{path}/MRR-{file_title}"
    plt.savefig(f"{plot_file}.png", dpi=600)
    plt.savefig(f"{plot_file}.pdf", dpi=1200)
    plt.close()
    print(f"MRR plots saved to {plot_file}.png/pdf.")
