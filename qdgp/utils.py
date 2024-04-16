"""Helper functions."""

import itertools
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def scorify(S: np.ndarray, seed_list: List[int]) -> np.ndarray:
    """Sum edge-scores over the seed nodes.

    Args:
    ----
    S: nxn score matrix.
    seed_list: List of nodes that are seeds.

    Returns:
    -------
    Score for each node in the graph, which is the sum of scores to each seed node.

    """
    return S[:, seed_list].sum(axis=1)


def seed_list_to_mask(seed_list: List[int], n: int) -> np.ndarray:
    """Convert list of seed nodes to a mask vector.

    Args:
    ----
    seed_list: List of nodes that are seeds.
    n: Number of entries in the returned array.

    Returns:
    -------
    Indicator array with 1s at seed indices.

    """
    train_seed_mask = np.zeros(n, dtype=int)
    train_seed_mask[seed_list] = 1
    return train_seed_mask


def seed_mask_to_list(seed_mask: np.ndarray) -> np.ndarray:
    """Retreive indices where seeds are located."""
    return np.where(seed_mask > 0)[0]


def avg_degree(G: "nx.Graph") -> float:
    """Average degree of graph of G."""
    s = 0
    for node in G.nodes:
        s += G.degree(node)
    return s / G.number_of_nodes()


def avg_seed_degree(G: "nx.Graph", seeds: List) -> float:
    """Average degree of graph of G."""
    s = 0
    for node in seeds:
        s += G.degree(node)
    return s / len(seeds)


def inv_code_dict(code_dict: Dict[int, int]) -> Dict[int, int]:
    """Invert the mapping defined in code_dict.

    Args:
    ----
    code_dict: A dictionary mapping node indices to gene ids.

    Returns:
    -------
    Dictionary mapping gene ides to node indices.

    """
    return {v: k for k, v in code_dict.items()}


def sub_density(G: "nx.Graph", seed_list: List) -> float:
    """Calculate the density of the subgraph induced by seed_list nodes."""
    return nx.density(G.subgraph(seed_list))


def seed_avg_shortest_path(G: "nx.Graph", seed_list: List) -> float:
    """Compute the average shortest path, averaged over ever pair of seeds."""
    c = 0
    d = 0
    for u, v in itertools.combinations(seed_list, 2):
        c += 1
        d += nx.shortest_path_length(G, source=u, target=v)
    return d / c


def sub_gcc(G: "nx.Graph", seed_list: List) -> int:
    """Return size of GCC in the subgraph of G given by seed_list."""
    G0 = G.subgraph(seed_list)
    Gcc = sorted(nx.connected_components(G0), key=len, reverse=True)
    return len(Gcc[0])


def const_seed_diagonals(G: "nx.Graph", seeds: np.ndarray, val: float) -> csr_matrix:
    """Set a constant value on the diagonal indices corresponding to seeds.

    Args:
    ----
    G: The ppi network.
    seeds: Nodes corresponding to seed genes.
    val: The constant value to set for the diagonals.

    Returns:
    -------
    Sparse array with `val` at indices given by `seeds`.

    """
    diag = csr_matrix((1, G.number_of_nodes()), dtype=np.float64)
    for s in seeds:
        diag[0, s] = val
    return diag


def process_df_ap(results_df: pd.DataFrame, iteration: int = 300) -> pd.DataFrame:
    """Aggregate simulation results to produce average precision results."""
    results_df = results_df[results_df.Iteration == iteration]
    results_df = results_df[["Model", "Ap"]]
    tdf = results_df.groupby(["Model"]).agg({"Ap": ["mean", "std"]}).reset_index()
    tdf.columns = [" ".join(col).strip() for col in tdf.columns.to_numpy()]
    return tdf[["Model", "Ap mean", "Ap std"]]


def process_df_recall(results_df: pd.DataFrame, iteration: int = 300) -> pd.DataFrame:
    """Aggregate simulation results to produce mean recall results."""
    results_df = results_df[results_df.Iteration == iteration]
    results_df = results_df[["Model", "Recall"]]
    tdf = results_df.groupby("Model").agg({"Recall": ["mean", "std"]}).reset_index()
    tdf.columns = [" ".join(col).strip() for col in tdf.columns.to_numpy()]
    return tdf[["Model", "Recall mean", "Recall std"]]


def process_df_mrr(results_df: pd.DataFrame, iteration: int = 300) -> pd.DataFrame:
    """Aggregate simulation results to produce mean reciprocal rank results."""
    results_df = results_df[results_df.Iteration == iteration]
    tdf = results_df
    tdf = tdf.groupby(["Model", "Disease", "Iteration"]).mean().reset_index()
    tdf["Rank"] = tdf.groupby(["Disease", "Iteration"]).rank(
        ascending=False,
        method="min",
    )["True Hits"]
    tdf["MRR"] = 1 / tdf["Rank"]
    tdf = tdf.groupby("Model").agg({"MRR": ["mean", "std"]}).reset_index()
    tdf.columns = [" ".join(col).strip() for col in tdf.columns.to_numpy()]
    return tdf[["Model", "MRR mean", "MRR std"]]


def summarize_results(results_df: pd.DataFrame) -> List[pd.DataFrame]:
    """Build recall, mean reciprocal rank, and average precision tables."""
    recall25 = process_df_recall(results_df, iteration=25)
    recall25.name = "Recall@25"
    recall300 = process_df_recall(results_df, iteration=300)
    recall300.name = "Recall@300"
    mrr25 = process_df_mrr(results_df, iteration=25)
    mrr25.name = "MRR@25"
    mrr300 = process_df_mrr(results_df, iteration=300)
    mrr300.name = "MRR@300"
    ap = process_df_ap(results_df, iteration=1)
    ap.name = "Avg. Precision"
    return [recall25, recall300, mrr25, mrr300, ap]
