import networkx as nx
import numpy as np
import pytest
from scipy.sparse.linalg import expm

import qdgp.models as md
import qdgp.utils as ut


@pytest.mark.parametrize(("n_nodes", "diag"), [(1000, None), (1000, 5), (1000, 5.5)])
def test_qa_score(n_nodes, diag):
    n_seeds = 30
    t = 1.0
    diag = 5
    G = nx.erdos_renyi_graph(n=n_nodes, p=0.1)
    H = nx.adjacency_matrix(G)
    rng = np.random.default_rng(1337)
    seeds = rng.choice(np.arange(n_nodes), n_seeds, False)

    r1 = md.qa_score(G, list(seeds), t=t, H=H, diag=diag)

    H_ = H.toarray()
    for s in seeds:
        H_[s, s] = diag
    P = expm(-1j * t * H_)
    P = np.abs(P) ** 2
    r2 = ut.scorify(P, list(seeds))

    assert np.allclose(r1, r2)


@pytest.mark.parametrize(("n_nodes"), [100, 1000])
def test_dk_score(n_nodes):
    n_seeds = 30
    t = 1.0
    G = nx.erdos_renyi_graph(n=n_nodes, p=0.1)
    L = nx.laplacian_matrix(G)
    rng = np.random.default_rng(1337)
    seeds = rng.choice(np.arange(n_nodes), n_seeds, False)

    r1 = md.dk_score(G, list(seeds), t=t, L=L, P=None)

    P = expm(-t * L.toarray())
    r2 = ut.scorify(P, list(seeds))

    assert np.allclose(r1, r2, atol=1e-5)
