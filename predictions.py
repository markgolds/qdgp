import argparse
import logging
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd

import qdgp.data as dt
import qdgp.models as md
import qdgp.utils as ut

logger = logging.getLogger(__name__)


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(description="Score genes for a disease.")
    parser.add_argument("-n", "--network", type=str, default="gmb")
    parser.add_argument("-D", "--disease_set", type=str, default="gmb")
    parser.add_argument("-d", "--disease", type=str, default="vasculitis")
    parser.add_argument("-t", "--topn", type=int, default=200)

    args = parser.parse_args()
    return {
        "network": args.network,
        "disease_set": args.disease_set,
        "disease": args.disease,
        "topn": args.topn,
    }


def main() -> None:
    params = parse_args()
    network = params["network"]
    disease_set = params["disease_set"]
    disease = params["disease"]
    topn = params["topn"]
    logs_path = "logs"
    if not Path(logs_path).exists():
        Path(logs_path).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=f"logs/predictions-{disease_set}_{network}.log",
        filemode="w",
        level=logging.INFO,
    )
    logging.info(params)

    all_dfs = []
    G, code_dict, seeds_by_disease = dt.load_dataset(
        disease_set,
        network,
        dt.FilterGCC.TRUE,
    )
    if disease not in seeds_by_disease:
        e = f"{disease} is not found in {disease_set} disease set."
        raise ValueError(e)
    inv_code_dict = ut.inv_code_dict(code_dict)
    n = G.number_of_nodes()
    nl = range(n)
    A = nx.adjacency_matrix(G, nodelist=nl)
    rows = []
    seeds = seeds_by_disease[disease]
    scores = md.qrw_score(G, seeds, H=A, t=0.45, diag=5)
    train_seed_mask = ut.seed_list_to_mask(seeds, G.number_of_nodes())
    test_mask = (1 - train_seed_mask).astype(bool)
    scores_test = scores[test_mask]
    ind = np.argpartition(scores_test, -topn)[-topn:]
    top_N = ind[np.argsort(scores_test[ind])]
    for i in range(topn):
        gene = inv_code_dict[top_N[i]]
        rank = topn - i
        deg = G.degree(top_N[i])
        rows.append([gene, network, disease, rank, deg])

    results_df = pd.DataFrame(
        rows, columns=["Gene", "Network", "Disease", "Rank", "Degree"]
    )
    all_dfs.append(results_df)
    results_df = pd.concat(all_dfs)
    results_df.to_csv(f"Predictions-{disease}.csv")


if __name__ == "__main__":
    main()
