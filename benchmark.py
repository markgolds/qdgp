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


def timing(f: Callable) -> Callable:
    @wraps(f)
    def wrap(*args: Dict, **kw: Dict) -> float:
        ts = time()
        _ = f(*args, **kw)
        te = time()
        f_name = f.__name__
        diff = te - ts
        logger.info("Function %s took: %2.4f seconds.", f_name, diff)
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


def main() -> None:
    G, code_dict, seeds_by_disease = dt.load_dataset("gmb", "gmb", dt.FilterGCC.TRUE)
    diseases = list(seeds_by_disease.keys())

    n = G.number_of_nodes()
    nl = range(n)
    n_runs = 2
    rows = []
    funcs = [benchmark_nei, benchmark_dia, benchmark_crw, benchmark_rwr, benchmark_qa]
    for run in range(n_runs):
        for dis in diseases:
            for f in funcs:
                res = f(G, nl, [dis], seeds_by_disease)
                rows.append([f.__name__, dis, len(seeds_by_disease[dis]), res, run])
                print([f.__name__, dis[:4], len(seeds_by_disease[dis]), res, run])

    res_df = pd.DataFrame(
        rows, columns=["Method", "Disease", "Num_seeds", "Time", "Run"]
    )
    print(res_df)
    res_df.to_csv("benchmark.csv")


if __name__ == "__main__":
    if not Path("logs").exists():
        Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/benchmark.log",
        filemode="w",
        level=logging.INFO,
    )
    print("Starting benchmark.")
    main()
