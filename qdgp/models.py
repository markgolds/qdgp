import logging
from typing import List, Optional, Union

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.stats import hypergeom

import qdgp.utils as ut

logger = logging.getLogger(__name__)


def qrw_score(
    G: nx.Graph,
    seed_list: List,
    t: float,
    H: csr_matrix,
    diag: Union[float, None] = None,
) -> np.ndarray:
    """Calculate quantum walk scores if transition matrix is not pre-computed.

    This method will be faster than qrw_score if only computing scores for
    a few diseases.

    Args:
    ----
        G: Graph that the walk occures on.
        seed_list: List of seed nodes.
        t: Time for which the walk lasts.
        H: Matrix to use as Hamiltonian.
        diag: How to set diagoanls of the Hamiltonian.

    Returns:
    -------
        Score for each node in G.

    """
    n = G.number_of_nodes()
    n_seeds = len(seed_list)
    if isinstance(diag, (float, int)):
        # Construct sparse matrix representation of nxn matrix, with values `diag`
        # at entry (seed, seed), for each seed in seed_list:
        D = csr_matrix(([diag] * n_seeds, (seed_list, seed_list)), shape=(n, n))
        H += D
    # Trick for doing "quantum expm_multiply":
    Z = np.zeros((n, n_seeds), dtype=int)
    Z[seed_list, np.arange(n_seeds)] = 1  # each column corresponds to a seed
    res = expm_multiply(-1j * t * H, Z)
    res = np.abs(res) ** 2
    return res.sum(axis=1)


def crw_score(
    G: nx.Graph,
    seed_list: List,
    t: Optional[float] = None,
    L: Optional[csr_matrix] = None,
    P: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score nodes based on classical walk.

    If the probability transition matrix P is pre-computed, use it.
    Otherwise, the Laplacian L will be used to perform the exponential matrix action
    on the seeds.

    Args:
    ----
        G: Graph upon which to walk.
        seed_list: List of seed nodes.
        L: Laplacian of G.
        t: Walk for time t.
        P: Probability transition matrix, if pre-computed.

    Returns:
    -------
        Scores for each node in G.

    """
    if P is None:
        if L is None or t is None:
            e = "One of L and t  must not be provided if P is None."
            raise ValueError(e)
        train_seed_mask = ut.seed_list_to_mask(seed_list, G.number_of_nodes())
        return expm_multiply(-t * L, train_seed_mask)
    return ut.scorify(P, seed_list)


def neighbourhood_score(G: nx.Graph, seed_list: List, A: np.ndarray) -> np.ndarray:
    """Calculate node scores using weighted neighbours.

    Args:
    ----
        G: Graph to use.
        seed_list: List of seed nodes.
        A: Dense adjacency of G.

    Returns:
    -------
        Scores for each node in G.

    """
    n = G.number_of_nodes()
    train_seed_mask = ut.seed_list_to_mask(seed_list, n)
    num_seed_neighbours = np.dot(A, train_seed_mask)
    degrees = np.sum(A, axis=1)
    scores = num_seed_neighbours / (degrees + 1e-50)
    return scores * (1 - train_seed_mask)


def normalize_adjacency(G: nx.Graph, A: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute normalized adjacnecy matrix for RWR.

    Args:
    ----
        G: Graph to use.
        A: Dense adjacency of G.

    Returns:
    -------
        The normalized adjacency matrix, defined by D^(-1/2)AD^(-1/2).

    """
    # TODO: will break if there are isolated nodes?:
    if A is None:
        A = nx.adjacency_matrix(G).toarray()
    D = [1 / np.sqrt(d) for _, d in G.degree(range(G.number_of_nodes()))]
    D = np.diag(D)
    return D @ A @ D


def rwr_score(
    G: nx.Graph,
    seed_list: List,
    return_prob: float = 0.75,
    normalized_adjacency: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score nodes based on random walk with restart.

    Args:
    ----
        G: Graph to use.
        seed_list: List of nodes that are seeds.
        return_prob: Probability of return to seed nodes.
        normalized_adjacency: D^{-1/2} A D^{-1/2}.

    Returns:
    -------
        Scores for each node in G.

    """
    # Modified from https://github.com/mims-harvard/pathways/blob/master/prediction/randomWalk.py
    if normalized_adjacency is None:
        normalized_adjacency = normalize_adjacency(G)
    train_seed_mask = ut.seed_list_to_mask(seed_list, G.number_of_nodes())
    assoc_gene_vector = train_seed_mask
    ratio = return_prob
    convergence_metric = 1
    p0 = assoc_gene_vector / np.sum(assoc_gene_vector)
    old_vector = p0
    while convergence_metric > 1e-6:
        new_vector = (1 - ratio) * np.dot(normalized_adjacency, old_vector) + ratio * p0
        convergence_metric = np.linalg.norm(new_vector - old_vector)
        old_vector = np.copy(new_vector)
    return old_vector * (1 - assoc_gene_vector)


def _compare_to_existing(
    processed_list: List,
    seed_conns: np.ndarray,
    total_conns: np.ndarray,
) -> bool:
    less_likely = [
        (a, b) for (a, b) in processed_list if a >= seed_conns and b <= total_conns
    ]
    if len(less_likely) == 0:
        return False
    return True


def diamond_score(
    G: nx.Graph,
    seed_list: List,
    A: np.ndarray,
    alpha: float = 5,
    number_to_rank: int = 100,
) -> np.ndarray:
    """Score nodes based on the diamond algorithm.

    Args:
    ----
        G: Graph to use.
        seed_list: List of nodes that are seeds.
        A: Dense adjacency of G.
        alpha: diamond parameter.
        number_to_rank: Score only this many nodes.

    Returns:
    -------
        Scores for the top `number_to_rank` nodes, according to the diamond algorithm.

    """
    train_seed_mask = ut.seed_list_to_mask(seed_list, G.number_of_nodes())
    assoc_gene_vector = train_seed_mask
    num_genes = assoc_gene_vector.shape[0]
    edges_per_gene = np.sum(A, axis=0)
    scores = np.zeros(assoc_gene_vector.shape)
    seeds = np.copy(assoc_gene_vector)
    connections_to_seeds = np.sum(A[:, np.nonzero(assoc_gene_vector)[0]], axis=1)
    num_gene_edges = edges_per_gene + (alpha - 1) * connections_to_seeds
    N = num_genes + np.sum(assoc_gene_vector) * (alpha - 1)
    connections_to_seeds = connections_to_seeds * (alpha)
    num_seeds = alpha * np.sum(assoc_gene_vector)
    for index in range(1, number_to_rank + 1):
        potential_cand = np.nonzero(connections_to_seeds * (1 - seeds) >= 1)[0]
        num_candidates = potential_cand.shape[0]
        if num_candidates == 0:
            break
        best_cand = -1
        best_conn = 1
        existing_calculations = []
        for i in range(num_candidates):
            cand_index = potential_cand[i]
            if _compare_to_existing(
                existing_calculations,
                connections_to_seeds[cand_index],
                num_gene_edges[cand_index],
            ):
                continue
            conn = hypergeom.sf(
                connections_to_seeds[cand_index] - 1,
                N,
                num_seeds,
                num_gene_edges[cand_index],
            )
            existing_calculations.append(
                (connections_to_seeds[cand_index], num_gene_edges[cand_index]),
            )
            if conn < best_conn:
                best_conn = conn
                best_cand = cand_index
        connections_to_seeds += A[:, best_cand]
        seeds[best_cand] = 1
        scores[best_cand] = 1.0 / index
        num_seeds += 1
    return scores
