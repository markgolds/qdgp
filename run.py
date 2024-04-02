import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict

import networkx as nx
import pandas as pd
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import expm

import qdgp.data as dt
import qdgp.evaluate as ev
import qdgp.models as md

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(description="Score genes for a disease.")
    parser.add_argument("-i", "--index", type=int, default=0)
    parser.add_argument("-r", "--runs", type=int, default=1)
    parser.add_argument("-n", "--network", type=str, default="hprd")
    parser.add_argument("-d", "--disease_set", type=str, default="ot")
    parser.add_argument("-s", "--split_ratio", type=float, default=0.5)
    args = parser.parse_args()
    return {
        "index": args.index,
        "runs": args.runs,
        "network": args.network,
        "disease_set": args.disease_set,
        "split_ratio": args.split_ratio,
    }


def main() -> None:
    params = parse_args()
    idx = params["index"]
    runs = params["runs"]
    network = params["network"]
    disease_set = params["disease_set"]
    split_ratio = params["split_ratio"]

    if not Path("logs").exists():
        Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=f"logs/{disease_set}_{network}_output{idx}.log",
        filemode="w",
        level=logging.INFO,
    )
    logging.info(params)
    gcc_filt = dt.FilterGCC.TRUE
    G, code_dict, seeds_by_disease = dt.load_dataset(disease_set, network, gcc_filt)

    logger.info("(%d, %d) (nodes, edges).", G.number_of_nodes(), G.number_of_edges())
    logger.info(list(seeds_by_disease.keys()))
    diseases = list(seeds_by_disease.keys())

    # ------------------------------
    # diseases = [diseases[0]]
    # ------------------------------

    # Pre-compute basic matrices:
    n = G.number_of_nodes()
    nl = range(n)
    L = nx.laplacian_matrix(G, nodelist=nl)
    A = nx.adjacency_matrix(G, nodelist=nl)
    A_d = A.toarray()
    # QP = md.qrw(H=A_d, t=0.45)
    # CP = md.crw(L=L.toarray(), t=0.3)
    CP = expm(-0.3 * L.toarray())

    R = md.normalize_adjacency(G, A_d)

    TOP_N = 300

    models = [
        md.qrw_score,
        md.crw_score,
        md.diamond_score,
        md.rwr_score,
        md.neighbourhood_score,
    ]

    m_names = ["QA", "DK", "Dia", "RWR", "NBR"]
    kws = [
        {"t": 0.45, "H": A, "diag": 5},
        {"t": 0.3, "L": L, "P": CP},
        {"alpha": 9, "number_to_rank": TOP_N, "A": A_d},
        {"return_prob": 0.4, "normalized_adjacency": R},
        {"A": A_d},
    ]

    if len(models) != len(m_names):
        e = "len(models) != len(m_names)"
        raise ValueError(e)
    if len(models) != len(kws):
        e = "len(models) == len(kws)"
        raise ValueError(e)

    rows = ev.run_models(
        G,
        models,
        m_names,
        kws,
        num_runs=runs,
        top_n=TOP_N,
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
            "Train_gcc_size",
            "Avg_deg_train_seeds",
            "Seed_density",
            "Seed_shortest_paths",
            "Conductance",
            "Auroc",
            "Ap",
        ],
    )
    results_df["Network"] = network
    path = f"out/{disease_set}/{network}/{split_ratio:.3f}/"
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(f"{path}/df{idx}.csv")


if __name__ == "__main__":
    main()
