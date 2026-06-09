"""Network analysis — product-reviewer graph + unsupervised suspicious clusters.

We build a bipartite reviewer↔product graph, project it onto products (two
products are linked when they share reviewers), and find densely connected
product communities — an *unsupervised* signal for coordinated activity that
replicates the network finding in the fake-review literature. We then combine
behavioral signals into a per-product suspicion score.

Label note: communities and scores here are UNSUPERVISED candidate signals,
validated later against the weak proxy and (Step 5) the real label — never
assumed to be fake.
"""

from __future__ import annotations

from typing import Mapping

import networkx as nx
import numpy as np
import pandas as pd

from . import utils

LOGGER = utils.get_logger("review_deception.network")


def build_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    """Reviewer↔product bipartite graph; edge weight = #reviews by that reviewer."""
    g = nx.Graph()
    for (reviewer, product), grp in df.groupby(["reviewer_id", "product_id"]):
        r, p = f"r:{reviewer}", f"p:{product}"
        g.add_node(r, kind="reviewer")
        g.add_node(p, kind="product")
        g.add_edge(r, p, weight=len(grp))
    return g


def project_products(df: pd.DataFrame, min_shared: int = 1) -> nx.Graph:
    """Product-product graph; edge weight = #reviewers shared by the two products."""
    reviewers_by_product: dict[str, set[str]] = {}
    for product, grp in df.groupby("product_id"):
        reviewers_by_product[str(product)] = set(grp["reviewer_id"])
    g = nx.Graph()
    g.add_nodes_from(reviewers_by_product)
    products = list(reviewers_by_product)
    for i, a in enumerate(products):
        for b in products[i + 1:]:
            shared = len(reviewers_by_product[a] & reviewers_by_product[b])
            if shared >= min_shared:
                g.add_edge(a, b, weight=shared)
    return g


def detect_communities(product_graph: nx.Graph) -> dict[str, int]:
    """Greedy-modularity communities → ``{product_id: community_id}`` (singletons = -1)."""
    if product_graph.number_of_edges() == 0:
        return {n: -1 for n in product_graph.nodes}
    communities = nx.community.greedy_modularity_communities(product_graph, weight="weight")
    mapping: dict[str, int] = {}
    for cid, members in enumerate(communities):
        for node in members:
            mapping[node] = cid
    for node in product_graph.nodes:  # isolated products
        mapping.setdefault(node, -1)
    return mapping


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    return (s - s.mean()) / std if std and not np.isnan(std) else pd.Series(0.0, index=s.index)


def suspicious_product_table(df: pd.DataFrame, config: Mapping | None = None,
                             min_shared: int = 2) -> pd.DataFrame:
    """Per-product table of behavioral signals + an unsupervised suspicion score.

    Score combines (z-scored): low verified-purchase rate, high review burst, high
    mean near-duplicate similarity, and high shared-reviewer degree. Higher =
    more suspicious. This is a heuristic ranking, not a verdict.
    """
    cfg = config or utils.load_config()
    work = df.copy()

    # Behavioral inputs (compute on the fly if a feature table wasn't supplied).
    if "product_burst_max" not in work or "max_intra_product_similarity" not in work:
        from . import features
        work = features.add_burst_features(work, config=cfg)
        work = features.add_near_duplicate_features(work, config=cfg)

    verified_num = work["verified_purchase"].map({True: 1.0, False: 0.0})
    grp = work.groupby("product_id")
    table = pd.DataFrame({
        "review_count": grp.size(),
        "verified_ratio": verified_num.groupby(work["product_id"]).mean(),
        "mean_rating": grp["rating"].mean(),
        "burst_max": grp["product_burst_max"].max() if "product_burst_max" in work else grp.size(),
        "mean_near_dup": grp["max_intra_product_similarity"].mean(),
        "n_reviewers": grp["reviewer_id"].nunique(),
    })

    product_graph = project_products(work, min_shared=min_shared)
    degree = dict(product_graph.degree())
    communities = detect_communities(product_graph)
    table["shared_reviewer_degree"] = table.index.map(lambda p: degree.get(str(p), 0)).astype("int64")
    table["community"] = table.index.map(lambda p: communities.get(str(p), -1)).astype("int64")

    # Unsupervised suspicion score (verified_ratio inverted because LOW is suspect).
    table["suspicion_score"] = (
        _zscore(1.0 - table["verified_ratio"].fillna(table["verified_ratio"].mean()))
        + _zscore(table["burst_max"].astype("float64"))
        + _zscore(table["mean_near_dup"].fillna(0.0))
        + _zscore(table["shared_reviewer_degree"].astype("float64"))
    )
    return table.sort_values("suspicion_score", ascending=False)
