import logging
from typing import List, Optional, Union

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.stats import hypergeom

import qdgp.utils as ut

# from qdgp.utils import scorify


logger = logging.getLogger(__name__)


def qrw(H: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Get the probability transition matrix for a quantum walk on G.

    Args:
    ----
        H: Matrix to use as Hamiltonian
        t: Walk for time t

    """
    H_ = torch.from_numpy(H)
    P = torch.matrix_exp(-1j * t * H_)
    P = torch.abs(P) ** 2
    return P.numpy()


def qrw_score(
    G: nx.Graph,
    seed_list: List,
    H: np.ndarray,
    t: float = 1.0,
    P: Optional[np.ndarray] = None,
    diag: Union[float, None] = None,
) -> np.ndarray:
    """Score nodes based on quantum walk.

    Args:
    ----
    G: Graph upon which to walk
    seed_list: List of nodes that are seeds
    H: Matrix to use as Hamiltonian
    t: Walk for time t
    P: Probability transition matrix, if pre-computed
    diag: How to set diagoanls of the Hamiltonian

    """
    if P is None:
        return _sparse_qrw_score(G, t, np.array(seed_list), H, diag)
    S = P
    return ut.scorify(S, seed_list)


def _sparse_qrw_score(
    G: nx.Graph,
    t: float,
    seeds: np.ndarray,
    H: np.ndarray,
    diag: Union[float, None] = None,
) -> np.ndarray:
    """Calculate quantum walk scores if transition matrix is not pre-computed.

    Args:
    ----
    G: Graph that the walk occures on
    t: Time for which the walk lasts
    seeds: Seed nodes
    train_seed_mask: 0/1 array of false/true test nodes
    H: Matrix to use as Hamiltonian
    diag: How to set diagoanls of the Hamiltonian

    """
    n = G.number_of_nodes()
    d = np.zeros((n, n))
    if isinstance(diag, (float, int)):
        d = ut.const_seed_diagonals(G, seeds, diag)
    H_sparse = csr_matrix(H + d)
    z = np.zeros(n, dtype=int)
    scores = np.zeros(n)
    # Trick for doing "quatum expm_multiply":
    for idx in seeds:
        z[idx] = 1
        res = expm_multiply(-1j * t * H_sparse, z)
        res = np.abs(res) ** 2
        scores += res
        z[idx] = 0
    return scores


def crw(L: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Get the probability transition matrix for a classical walk on G.

    Args:
    ----
        L: Laplacian of G
        t: Walk for time t

    """
    L_ = torch.from_numpy(L)
    P = torch.matrix_exp(-1 * t * L_)
    return P.numpy()


def crw_score(
    G: nx.Graph,
    seed_list: List,
    L: Optional[csr_matrix] = None,
    t: float = 1.0,
    P: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score nodes based on classical walk.

    Args:
    ----
        G: Graph upon which to walk
        seed_list: List of seed nodes
        L: Laplacian of G
        t: Walk for time t
        P: Probability transition matrix, if pre-computed

    """
    if P is None:
        if L is None:
            e = "One of L or P must not be None"
            raise ValueError(e)
        train_seed_mask = ut.seed_list_to_mask(seed_list, G.number_of_nodes())
        return expm_multiply(-t * L, train_seed_mask)
    S = P
    return ut.scorify(S, seed_list)


def neighborhood_score(G: nx.Graph, seed_list: List, A: np.ndarray) -> np.ndarray:
    """Calculate node scores using weighted neighbors.

    Args:
    ----
        G: Graph to use
        seed_list: List of nodes that are seeds
        A: Dense adjacency of G

    """
    n = G.number_of_nodes()
    train_seed_mask = ut.seed_list_to_mask(seed_list, n)
    num_seed_neighbors = np.dot(A, train_seed_mask)
    degrees = np.sum(A, axis=1)
    scores = num_seed_neighbors / (degrees + 1e-50)
    return scores * (1 - train_seed_mask)


def normalize_adjacency(G: nx.Graph, A: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute normalized adjacnecy matrix for RWR.

    Args:
    ----
        G: Graph to use
        A: Dense adjacency of G

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
        G: Graph to use
        seed_list: List of nodes that are seeds
        return_prob: Probability of return to seed nodes
        normalized_adjacency: D^{-1/2} A D^{-1/2}

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
    """Score nodes based on diamond algorithm.

    Args:
    ----
    G: Graph to use
    seed_list: List of nodes that are seeds
    A: Dense adjacency of G
    alpha: diamond parameter
    number_to_rank: Score only this many nodes

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
