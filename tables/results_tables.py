"""Output latex tables describing Recalls and MRRs."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def process_df_ap(all_dfs: pd.DataFrame, iteration: int = 1) -> pd.DataFrame:
    all_dfs = all_dfs[all_dfs.Iteration == iteration]
    all_dfs = all_dfs[["Model", "Ap"]]
    tdf = all_dfs.groupby(["Model"]).agg({"Ap": ["mean", "std"]}).reset_index()
    tdf.columns = [" ".join(col).strip() for col in tdf.columns.to_numpy()]
    return tdf[["Model", "Ap mean", "Ap std"]]


def process_df_recall(all_dfs: pd.DataFrame, iteration: int = 1) -> pd.DataFrame:
    all_dfs = all_dfs[all_dfs.Iteration == iteration]
    all_dfs = all_dfs[["Model", "Recall"]]
    tdf = all_dfs.groupby("Model").agg({"Recall": ["mean", "std"]}).reset_index()
    tdf.columns = [" ".join(col).strip() for col in tdf.columns.to_numpy()]
    return tdf[["Model", "Recall mean", "Recall std"]]


def process_df_mrr(all_dfs: pd.DataFrame, iteration: int = 1) -> pd.DataFrame:
    all_dfs = all_dfs[all_dfs.Iteration == iteration]

    tdf = all_dfs
    tdf = tdf.groupby(["Model", "Disease", "Iteration"]).mean().reset_index()
    tdf["Rank"] = tdf.groupby(["Disease", "Iteration"]).rank(
        ascending=False,
        method="min",
    )["True Hits"]
    tdf["MRR"] = 1 / tdf["Rank"]
    tdf = tdf.groupby("Model").agg({"MRR": ["mean", "std"]}).reset_index()
    tdf.columns = [" ".join(col).strip() for col in tdf.columns.to_numpy()]
    return tdf[["Model", "MRR mean", "MRR std"]]


def bold_extreme_values(data, data_max=-1) -> str:
    if data == data_max:
        return "\\textbf{%.3f}" % data
    return "%.3f" % data


def highlight(data, data_largest) -> str:
    if data == data_largest[0]:
        return "\\textbf{%.3f}" % data
    return "%.3f" % data


def to_table(all_dfs: pd.DataFrame, kind: str) -> pd.DataFrame:
    means_df = all_dfs[["Model", "Disease Set", "Network", f"{kind} mean"]]
    means_df.columns = ["Model", "Disease Set", "Network", f"{kind}"]
    means_table = means_df.pivot_table(
        index="Model",
        columns=["Disease Set", "Network"],
        values=[kind],
    )
    stds_df = all_dfs[["Model", "Disease Set", "Network", f"{kind} std"]]
    stds_df.columns = ["Model", "Disease Set", "Network", kind]
    stds_table = stds_df.pivot_table(
        index="Model",
        columns=["Disease Set", "Network"],
        values=[kind],
    )

    for k in means_table.columns[0:]:
        means_table[k] = means_table[k].apply(
            lambda data: highlight(data, means_table[k].nlargest(2)),
        )

    for k in stds_table.columns[0:]:
        stds_table[k] = stds_table[k].apply(
            lambda data: f"({data:.2f})",
        )

    d_m = means_table.to_dict()
    d_s = stds_table.to_dict()
    for k in d_m:
        for kk in d_m[k]:
            d_m[k][kk] += f" \\scriptsize{{{d_s[k][kk]}}}"
    return pd.DataFrame(d_m)


def rename_items(all_dfs: pd.DataFrame) -> pd.DataFrame:
    d1 = {
        "wl-ppi": "WL",
        "menche": "GMB",
        "loami-apid": "APID",
        "loami-iid": "IID",
        "loami-biogrid": "BioGRID",
        "loami-hprd": "HPRD",
        "loami-string": "STRING",
    }
    d2 = {"menche": "GMB"}
    d3 = {"Dia": "DIA"}
    return all_dfs.replace({"Network": d1, "Disease Set": d2, "Model": d3})


def build_tables(kind: str = "Recall") -> None:
    for i in [25, 300]:
        all_dfs = []
        for ds in ["DGN", "menche", "OT"]:
            # for ds in ["OT"]:
            for network in [
                "loami-apid",
                "loami-biogrid",
                "loami-hprd",
                "menche",
                "loami-iid",
                "loami-string",
                "wl-ppi",
            ]:
                try:
                    # f = f"out/{ds}/{network}/0.500/ALLdf-{network}-{ds}-0.500.csv"
                    f = f"out-az/{ds}/{network}/0.5/ALLdf-{network}-{ds}-0.5.csv"
                    all_df = pd.read_csv(f)
                    all_df = all_df[all_df.Num_seeds >= 15]
                    if kind == "MRR":
                        res = process_df_mrr(all_df, iteration=i)
                    elif kind == "Ap":
                        res = process_df_ap(all_df, iteration=i)
                    else:
                        res = process_df_recall(all_df, iteration=i)
                    res["Disease Set"] = ds
                    res["Network"] = network
                    all_dfs.append(res)
                except:
                    pass

        all_dfs = pd.concat(all_dfs)

        all_dfs = rename_items(all_dfs)

        avgs = all_dfs.groupby(["Model"]).agg({f"{kind} mean": ["mean"]}).to_dict()
        avgs = avgs[(f"{kind} mean", "mean")]

        max_avg = max([v for k, v in avgs.items()])
        for k in avgs:
            if avgs[k] == max_avg:
                avgs[k] = f"\\textbf{{{avgs[k]:.3f}}}"
            else:
                avgs[k] = f"{avgs[k]:.3f}"

        all_dfs[kind] = all_dfs.apply(
            lambda row: f"{round(row[f'{kind} mean'], 3):.3f} ({round(row[f'{kind} std'], 2):.2f})",
            axis=1,
        )

        models = ["QA", "DIA", "DK", "NBR", "RWR"]
        if kind == "Ap":
            models = ["QA", "DK", "NBR", "RWR"]

        all_dfs = all_dfs[all_dfs.Model.isin(models)]  # keep only relevant models

        table = to_table(all_dfs, kind)

        table = table.droplevel(level=0, axis=1)

        table["Average"] = ""
        for key, value in avgs.items():
            table.at[key, "Average"] = value  # f"{round(value, 3):.3f}"

        s = table.transpose()[models].to_latex(
            escape=False,
            multirow=False,
        )

        path = Path(f"tables/ResTable-{kind}{i}-withstd---.txt")
        with path.open("w") as text_file:
            text_file.write(s)


if __name__ == "__main__":
    build_tables(kind="Recall")
    build_tables(kind="MRR")
    build_tables(kind="Ap")
