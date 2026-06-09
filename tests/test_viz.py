"""Smoke tests for src.viz — every helper returns a labeled Axes and saves."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

from src import features, models, utils, viz


def test_eda_plots_return_labeled_axes(clean_reviews_df):
    feat = features.build_feature_table(clean_reviews_df)
    for ax in (viz.plot_class_balance(clean_reviews_df),
               viz.plot_rating_distribution(clean_reviews_df),
               viz.plot_sentiment_by_group(feat)):
        assert ax is not None
        assert ax.get_title() and ax.get_xlabel() and ax.get_ylabel()


def test_eval_plots_and_save(medium_reviews_df, tmp_path):
    cfg = utils.deep_merge(utils.load_config(), {"modeling": {
        "cv_folds": 3, "knn": {"n_neighbors_grid": [5]},
        "random_forest": {"n_estimators_grid": [50], "max_depth_grid": [None]},
    }})
    out = models.run_classification(medium_reviews_df, cfg)
    roc_ax = viz.plot_roc_curves(out["results"])
    viz.plot_pr_curves(out["results"])
    viz.plot_feature_importances(out["results"]["random_forest"]["importances"])
    path = viz.savefig(roc_ax, str(tmp_path / "roc.png"))
    assert os.path.exists(path)


def test_pca_scatter(medium_reviews_df):
    feat = features.build_feature_table(medium_reviews_df)
    X = models.design_matrix(feat)
    _, coords = models.fit_pca(X, n_components=2)
    ax = viz.plot_pca_scatter(coords, labels=None)
    assert ax.get_xlabel() == "PC 1"
