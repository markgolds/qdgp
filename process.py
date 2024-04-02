"""Combine data files generated from parallel runs and plot results."""

import argparse
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

import qdgp.plot as pl

logger = logging.getLogger(__name__)


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(description="Plot results from parallel runs.")
    parser.add_argument("-n", "--network", type=str, default="menche")
    parser.add_argument("-d", "--disease_set", type=str, default="menche")
    parser.add_argument("-t", "--title", type=str, default="")
    parser.add_argument("-s", "--split_ratio", type=str, default="0.500")
    args = parser.parse_args()
    return {
        "network": args.network,
        "disease_set": args.disease_set,
        "title": args.title,
        "split_ratio": args.split_ratio,
    }


def main() -> None:
    params = parse_args()
    network = params["network"].lower()
    plot_title = params["title"]
    disease_set = params["disease_set"].lower()
    split_ratio = params["split_ratio"]
    path = Path(f"out/{disease_set}/{network}/{split_ratio}")
    print(path)
    files = "*.csv"
    dfs = []
    for fname in path.glob(files):
        logger.info(fname)
        runs_df = pd.read_csv(fname)
        runs_df["Network"] = network
        dfs.append(runs_df)
    all_results_df = pd.concat(dfs)
    all_results_df["Model"] = all_results_df["Model"].astype(str)
    all_results_df = all_results_df.drop("Network", axis=1)
    pl.plot_white(all_results_df, title=plot_title)


if __name__ == "__main__":
    main()
