"""Smoke + correctness tests for src.models."""

from __future__ import annotations

import pandas as pd

from src import features, models, utils


def _fast_cfg() -> dict:
    return utils.deep_merge(utils.load_config(), {"modeling": {
        "cv_folds": 3,
        "knn": {"n_neighbors_grid": [3, 5]},
        "random_forest": {"n_estimators_grid": [50], "max_depth_grid": [None, 5]},
    }})


def test_make_label_excludes_na():
    df = pd.DataFrame({"verified_purchase": pd.array([True, False, None], dtype="boolean")})
    y, mask = models.make_label(df, positive="unverified")
    assert mask.tolist() == [True, True, False]   # the <NA> row is excluded
    assert y.tolist() == [0, 1]                    # verified->0, unverified(positive)->1


def test_run_classification_smoke(medium_reviews_df):
    out = models.run_classification(medium_reviews_df, _fast_cfg())
    assert set(out["results"]) == {"knn", "random_forest"}
    for ev in out["results"].values():
        assert 0.0 <= ev["roc_auc"] <= 1.0
        assert ev["roc"][0].shape == ev["roc"][1].shape
        assert ev["pr"][0].shape == ev["pr"][1].shape
    # leakage guard: the label-derived feature is excluded from the design matrix
    assert "product_verified_ratio" not in out["feature_names"]
    imp = out["results"]["random_forest"]["importances"]
    assert len(imp) == len(out["feature_names"])


def test_pca_and_kmeans(medium_reviews_df):
    feat = features.build_feature_table(medium_reviews_df)
    X = models.design_matrix(feat)
    _, coords = models.fit_pca(X, n_components=2)
    assert coords.shape == (len(X), 2)
    _, labels = models.fit_kmeans(X, k=3)
    assert len(labels) == len(X)
    assert len(set(labels)) <= 3


def test_cluster_review_styles(medium_reviews_df):
    labels, top_terms, vec = models.cluster_review_styles(medium_reviews_df, n_clusters=3)
    assert len(labels) == len(medium_reviews_df)
    assert set(top_terms) == set(int(c) for c in labels)
    assert all(len(terms) > 0 for terms in top_terms.values())
