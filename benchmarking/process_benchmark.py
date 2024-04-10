import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    res_df = pd.read_csv("benchmark.csv")
    res_df = res_df[res_df.Network == "wl"]
    print(res_df.columns)
    mean_df = res_df.groupby(["Method", "Num_seeds", "Network"]).agg(
        ({"Time (s)": ["mean", "std"]})
    )
    print(mean_df)
    # sns.lineplot(res_df, y="Time (s)", x="Num_seeds", hue="Method", row="Network")
    # grid = sns.FacetGrid(res_df, col="Network", col_wrap=2)
    # grid.map(sns.lineplot, kwargs={"x": "Num_seeds", "y": "Time (s)", "hue": "Method"})
    sns.relplot(
        res_df, y="Time (s)", x="Num_seeds", hue="Method", col="Network", kind="line"
    )
    # plt.show()
    plt.tight_layout()
    # plt.show()
    plt.savefig("benchmarking/benchamark.png")


if __name__ == "__main__":
    main()
