import logging
from functools import wraps
from pathlib import Path
from time import time
from typing import Callable, Dict

import networkx as nx
import pandas as pd

import qdgp.data as dt
import qdgp.models as md

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(message)s")


def timing(f: Callable) -> Callable:
    @wraps(f)
    def wrap(*args: Dict, **kw: Dict) -> float:
        ts = time()
        _ = f(*args, **kw)
        te = time()
        f_name = f.__name__
        diff = te - ts
        return diff

    return wrap


@timing
def benchmark_qa(G, nl, diseases, seeds_by_disease) -> None:
    A = nx.adjacency_matrix(G, nodelist=nl)
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.qrw_score(G, seeds, t=0.45, H=A, diag=None)


@timing
def benchmark_crw(G, nl, diseases, seeds_by_disease) -> None:
    L = nx.laplacian_matrix(G, nodelist=nl)
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.crw_score(G, seeds, L=L, t=0.3)


@timing
def benchmark_rwr(G, nl, diseases, seeds_by_disease) -> None:
    A = nx.adjacency_matrix(G, nodelist=nl)
    A_d = A.toarray()
    R = md.normalize_adjacency(G, A_d)
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.rwr_score(G, seeds, normalized_adjacency=R, return_prob=0.4)


@timing
def benchmark_dia(G, nl, diseases, seeds_by_disease) -> None:
    A = nx.adjacency_matrix(G, nodelist=nl)
    A_d = A.toarray()
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.diamond_score(G, seeds, A=A_d, alpha=9, number_to_rank=100)


@timing
def benchmark_nei(G, nl, diseases, seeds_by_disease) -> None:
    A = nx.adjacency_matrix(G, nodelist=nl)
    A_d = A.toarray()
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.neighbourhood_score(G, seeds, A=A_d)


def main(network: str) -> pd.DataFrame:
    G, code_dict, seeds_by_disease = dt.load_dataset("gmb", network, dt.FilterGCC.TRUE)
    diseases = list(seeds_by_disease.keys())

    n = G.number_of_nodes()
    nl = range(n)
    n_runs = 5
    rows = []
    funcs = [benchmark_nei, benchmark_dia, benchmark_crw, benchmark_rwr, benchmark_qa]
    for run in range(n_runs):
        for i, dis in enumerate(diseases[:]):
            for f in funcs:
                res = f(G, nl, [dis], seeds_by_disease)
                rows.append(
                    [f.__name__, dis, len(seeds_by_disease[dis]), res, run, network]
                )
                logger.info(
                    "%s - %s - %d - %d %f %d %s",
                    f.__name__,
                    dis[:4],
                    i,
                    len(seeds_by_disease[dis]),
                    res,
                    run,
                    network,
                )

    return pd.DataFrame(
        rows, columns=["Method", "Disease", "Num_seeds", "Time (s)", "Run", "Network"]
    )


if __name__ == "__main__":
    if not Path("logs").exists():
        Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/benchmark.log",
        filemode="w",
        level=logging.INFO,
    )

    # Create a handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
    logger.info("Starting benchmark.")
    all_dfs = []
    for network in ["wl"]:
        net_df = main(network)
        all_dfs.append(net_df)
    DF = pd.concat(all_dfs)
    DF.to_csv("benchmark.csv")
