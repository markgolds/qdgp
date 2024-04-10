import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .utils import (
    avg_seed_degree,
    seed_list_to_mask,
    seed_shortest_paths,
    sub_density,
    sub_gcc,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def hit_results(
    method: Callable,
    G: "nx.Graph",
    train_seeds: List[int],
    test_seeds: List[int],
    shuffled_nodes: np.ndarray,
    **kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Run method on datasets and return number of hits by iteration.

    Args:
    ----
    method: Node scoring algorithm to be used
    G: The underlying graph
    train_seeds: List of seed nodes that are known to the algorithms
    test_seeds: List of seed nodes reserved for testing
    shuffled_nodes: List of nodes in random order
    kwargs: Arguments for model

    """
    n = G.number_of_nodes()
    train_seed_mask = seed_list_to_mask(train_seeds, n)
    test_mask = (1 - train_seed_mask).astype(bool)
    logger.info("Training...")
    scores = method(G, train_seeds, **kwargs)
    logger.info("Done.")
    scores = scores[test_mask]  # discard scores of the train nodes
    y_true = seed_list_to_mask(
        test_seeds,
        n,
    )  # vector of non-seeds/seeds in test set
    y_true = y_true[test_mask]  # discard labels of the train nodes
    ordered_labels = [
        y
        for _, _, y in sorted(
            zip(scores, shuffled_nodes[test_mask], y_true),
            reverse=True,
        )
    ]  # order labels by scores, breaking ties according to order given in shuffle
    cumulative_hits = np.cumsum(ordered_labels)  # number of true hits by iteration
    num_test_seeds = len(test_seeds)
    # We only care about the top 5000 scores or so
    ordered_recalls = cumulative_hits[:5000] / num_test_seeds

    auroc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)

    return (
        cumulative_hits,
        ordered_recalls,
        float(auroc),
        float(ap),
    )


def run_models(
    G: "nx.Graph",
    models: List[Callable],
    m_names: List[str],
    kws: List[Dict],
    runs: int,
    top_n: int,
    diseases: List[str],
    n_by_d: Dict[str, List[int]],
    split_ratio: float = 0.5,
    train_test_seed: Optional[int] = None,
) -> List[List]:
    """Evaluate models for diseases on G.

    Args:
    ----
        G: Graph upon which to walk
        models: Node scoring algorithms to be used
        m_names: Strings for the model names
        kws: Arguments for the model methods
        runs: How many runs per diseases, for averages
        top_n: Keep this many of the top scores
        diseases: Which diseases to evaluate
        n_by_d: Mapping of disease name to seed nodes
        split_ratio: Fraction of seeds to use for training
        train_test_seed: Random seed for consistent train/test splits

    """
    # shuffle node list to break ties among scores
    rng = np.random.default_rng(0)
    shuffled_nodes = rng.permutation(list(G.nodes()))

    rows = []
    for run in range(runs):
        for c, disease in enumerate(diseases):
            genes = n_by_d[disease]
            logger.info(
                "Run: %d/%d - Disease: %d/%d: %s, %d seeds",
                run + 1,
                runs,
                c + 1,
                len(diseases),
                disease,
                len(genes),
            )
            if train_test_seed:
                train_seeds, test_seeds = train_test_split(
                    genes,
                    train_size=split_ratio,
                    random_state=train_test_seed + run,
                )
            else:
                train_seeds, test_seeds = train_test_split(
                    genes,
                    train_size=split_ratio,
                )

            avg_train_seed_deg = avg_seed_degree(G, train_seeds)
            train_gcc_size = sub_gcc(G, train_seeds)
            train_density = sub_density(G, train_seeds)
            seed_sp = seed_shortest_paths(G, train_seeds)
            conductance = nx.conductance(G, train_seeds)

            for m, mn, kw in zip(models, m_names, kws):
                logger.info("model: %s", mn)
                hits, recalls, auroc, ap = hit_results(
                    m,
                    G,
                    train_seeds,
                    test_seeds,
                    shuffled_nodes,
                    **kw,
                )
                logger.info("model: %s complete.", mn)
                for i in range(top_n):
                    rows.append(
                        [
                            mn,
                            disease,
                            run + 1,
                            i + 1,
                            hits[i],
                            recalls[i],
                            len(genes),
                            len(train_seeds),
                            train_gcc_size,
                            avg_train_seed_deg,
                            train_density,
                            seed_sp,
                            conductance,
                            auroc,
                            ap,
                        ],
                    )
    return rows
