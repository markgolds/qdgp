import argparse
import logging
from pathlib import Path
from typing import Dict

import networkx as nx
import pandas as pd
from scipy.sparse.linalg import expm

import qdgp.data as dt
import qdgp.evaluate as ev
import qdgp.models as md
import qdgp.plot as pl
import qdgp.utils as ut

logger = logging.getLogger(__name__)


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(description="Score genes for a disease.")
    parser.add_argument("-r", "--runs", type=int, default=1)
    parser.add_argument(
        "-n",
        "--network",
        type=str,
        default="hprd",
        choices=dt.valid_networks,
    )
    parser.add_argument(
        "-d",
        "--disease_set",
        type=str,
        default="ot",
        choices=dt.valid_datasets,
    )
    parser.add_argument("-s", "--split_ratio", type=float, default=0.5)
    args = parser.parse_args()
    return {
        "runs": args.runs,
        "network": args.network,
        "disease_set": args.disease_set,
        "split_ratio": args.split_ratio,
    }


def cross_validate() -> None:
    params = parse_args()
    runs = params["runs"]
    network = params["network"]
    disease_set = params["disease_set"]
    split_ratio = params["split_ratio"]
    if not Path("logs").exists():
        Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=f"logs/{disease_set}_{network}_output.log",
        filemode="w",
        level=logging.INFO,
    )
    logging.info(params)
    G, code_dict, seeds_by_disease = dt.load_dataset(
        disease_set,
        network,
        filter_method=dt.FilterGCC.TRUE,
    )
    logger.info("(%d, %d) (nodes, edges).", G.number_of_nodes(), G.number_of_edges())
    logger.info(list(seeds_by_disease.keys()))
    diseases = list(seeds_by_disease.keys())

    # Pre-compute basic matrices:
    n = G.number_of_nodes()
    nl = range(n)
    L = nx.laplacian_matrix(G, nodelist=nl)
    A = nx.adjacency_matrix(G, nodelist=nl)
    A_d = A.toarray()
    CP = expm(-0.3 * L.toarray())  # for the classical random walk
    R = md.normalize_adjacency(G, A_d)  # for random walk with restart

    top_n = 300  # for diamond

    # Set up the models
    m_qa = md.Model(md.qrw_score, "QA", {"t": 0.45, "H": A, "diag": 5})
    m_dk = md.Model(md.crw_score, "DK", {"P": CP})
    m_dia = md.Model(
        md.diamond_score, "DIA", {"alpha": 9, "number_to_rank": top_n, "A": A_d}
    )
    m_rwr = md.Model(
        md.rwr_score, "RWR", {"return_prob": 0.4, "normalized_adjacency": R}
    )
    m_nei = md.Model(md.neighbourhood_score, "NEI", {"A": A_d})

    models = [m_qa, m_dk, m_dia, m_rwr, m_nei]

    rows = ev.run_models(
        G,
        models,
        num_runs=runs,
        top_n=top_n,
        diseases=diseases,
        n_by_d=seeds_by_disease,
        split_ratio=split_ratio,
    )
    results_df = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Disease",
            "Run",
            "Iteration",
            "True Hits",
            "Recall",
            "Num_seeds",
            "Num_train_seeds",
            "Auroc",
            "Ap",
        ],
    )
    results_df["Network"] = network
    path = "out"
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(f"{path}/{disease_set}-{network}-{split_ratio:.3f}.csv")
    results_df = results_df.drop("Network", axis=1)
    pl.plot_results(results_df, title=f"{disease_set.upper()} | {network.upper()}")

    tables = ut.summarize_results(results_df)
    for r in tables:
        print(r.name)
        logger.info(r.name)
        print(r)
        logger.info("\n%s", r)


if __name__ == "__main__":
    cross_validate()
