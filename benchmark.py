import logging
from functools import wraps
from pathlib import Path
from time import time
from typing import Callable, Dict

import networkx as nx

import qdgp.data as dt
import qdgp.models as md

logger = logging.getLogger(__name__)


def timing(f: Callable) -> Callable:
    @wraps(f)
    def wrap(*args: Dict, **kw: Dict) -> None:
        ts = time()
        _ = f(*args, **kw)
        te = time()
        # print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        f_name = f.__name__
        diff = te - ts
        print(f"Function:{f_name} took: {diff:2.4} seconds.")
        logger.info("Function %s took: %2.4f seconds.", f_name, diff)

    return wrap


@timing
def benchmark_qa(G, nl, diseases, seeds_by_disease) -> None:
    A = nx.adjacency_matrix(G, nodelist=nl)
    A_d = A.toarray()
    QP = md.qrw(H=A_d, t=0.45)
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.qrw_score(G, seeds, H=A_d, t=0.45, diag=5, P=QP)


@timing
def benchmark_qa_sparse(G, nl, diseases, seeds_by_disease) -> None:
    A = nx.adjacency_matrix(G, nodelist=nl)
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.qrw_score(G, seeds, H=A, t=0.45, diag=5, P=None)


@timing
def benchmark_crw_sparse(G, nl, diseases, seeds_by_disease) -> None:
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
        md.diamond_score(G, seeds, A=A_d, alpha=9, number_to_rank=500)


@timing
def benchmark_nei(G, nl, diseases, seeds_by_disease) -> None:
    A = nx.adjacency_matrix(G, nodelist=nl)
    A_d = A.toarray()
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.neighbourhood_score(G, seeds, A=A_d)


def main() -> None:
    G, code_dict, seeds_by_disease = dt.load_dataset("gmb", "string", dt.FilterGCC.TRUE)
    # G, code_dict, seeds_by_disease = dt.load_dataset("dgn", "hprd", dt.FilterGCC.TRUE)
    diseases = list(seeds_by_disease.keys())

    diseases = ["aneurysm"]  # small
    diseases = ["multiple sclerosis"]  # big

    n = G.number_of_nodes()
    nl = range(n)

    benchmark_nei(G, nl, diseases, seeds_by_disease)
    benchmark_dia(G, nl, diseases, seeds_by_disease)
    benchmark_rwr(G, nl, diseases, seeds_by_disease)
    benchmark_crw_sparse(G, nl, diseases, seeds_by_disease)
    benchmark_qa_sparse(G, nl, diseases, seeds_by_disease)
    # benchmark_qa(G, nl, diseases, seeds_by_disease)


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
