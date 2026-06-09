"""Modeling — supervised + unsupervised, with honest evaluation (in-scope only).

Supervised "trust classifier" (target = weak ``verified_purchase`` proxy):
    KNN baseline + random forest, selected via cross-validation + grid search,
    evaluated with ROC and precision-recall curves.

Unsupervised structure:
    K-means + hierarchical clustering of review *styles*; PCA for 2-D views.

Reproducibility: every estimator/split is seeded from ``project.random_seed``.

Label note
----------
The classification target is a WEAK PROXY (purchase verification, not deception);
reported performance is interpreted accordingly, and Step 5 re-runs this pipeline
against a real fake-review label. We also DROP features that are deterministic
functions of the label (see :data:`LEAKY_FOR_PROXY`) so the classifier can't
cheat by reading the answer back off the product.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features, utils

LOGGER = utils.get_logger("review_deception.models")

# Features that are deterministic functions of the proxy label -> would leak it.
LEAKY_FOR_PROXY: frozenset[str] = frozenset({"product_verified_ratio"})


# ---------------------------------------------------------------------------
# Labels + matrix
# ---------------------------------------------------------------------------
def make_label(df: pd.DataFrame, proxy_col: str = "verified_purchase", positive: str = "unverified"):
    """Return ``(y, mask)`` where the positive class is the suspicious (unverified) one.

    Rows with a missing proxy (``<NA>``) are excluded via ``mask`` — we never
    invent a label. ``positive='unverified'`` makes 1 = the class we want to flag.
    """
    series = df[proxy_col]
    mask = series.notna()
    verified = series[mask].astype("boolean").astype("int64")
    y = (1 - verified) if positive == "unverified" else verified
    return y.astype("int64"), mask


def design_matrix(df: pd.DataFrame, drop_leaky: bool = True) -> pd.DataFrame:
    """Numeric design matrix from a feature table, optionally dropping leaky columns."""
    X = features.assemble_behavioral_features(df)
    if drop_leaky:
        X = X.drop(columns=[c for c in LEAKY_FOR_PROXY if c in X.columns])
    return X


# ---------------------------------------------------------------------------
# Supervised
# ---------------------------------------------------------------------------
def build_estimators(config: Mapping) -> dict[str, tuple[object, dict]]:
    """Return ``{name: (estimator, param_grid)}`` for KNN (scaled) and random forest."""
    seed = config["project"]["random_seed"]
    m = config["modeling"]
    knn = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())])
    knn_grid = {"knn__n_neighbors": list(m["knn"]["n_neighbors_grid"])}
    # n_jobs=1: GridSearchCV parallelizes across folds/params, so keeping the RF
    # single-threaded avoids nested-parallelism oversubscription (and its warning).
    rf = RandomForestClassifier(random_state=seed, n_jobs=1, class_weight="balanced")
    rf_grid = {
        "n_estimators": list(m["random_forest"]["n_estimators_grid"]),
        "max_depth": list(m["random_forest"]["max_depth_grid"]),
    }
    return {"knn": (knn, knn_grid), "random_forest": (rf, rf_grid)}


def grid_search(estimator, param_grid: dict, X, y, config: Mapping) -> GridSearchCV:
    """Stratified-CV grid search using the configured scoring metric."""
    seed = config["project"]["random_seed"]
    folds = min(config["modeling"]["cv_folds"], int(pd.Series(y).value_counts().min()))
    folds = max(folds, 2)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    gs = GridSearchCV(estimator, param_grid, scoring=config["modeling"]["scoring"],
                      cv=cv, n_jobs=-1, refit=True)
    gs.fit(X, y)
    return gs


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Compute ROC/PR curves + scalar metrics + confusion matrix on a held-out set."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, proba)
    precision, recall, _ = precision_recall_curve(y_test, proba)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "roc": (fpr, tpr),
        "pr": (precision, recall),
        "confusion_matrix": confusion_matrix(y_test, pred),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "y_score": proba,
    }


def feature_importances(model, feature_names: Iterable[str]) -> pd.Series:
    """Random-forest importances as a named, sorted Series (empty if unavailable)."""
    est = model.named_steps["rf"] if isinstance(model, Pipeline) and "rf" in model.named_steps else model
    if not hasattr(est, "feature_importances_"):
        return pd.Series(dtype="float64")
    return pd.Series(est.feature_importances_, index=list(feature_names)).sort_values(ascending=False)


def univariate_feature_auc(X: pd.DataFrame, y) -> pd.Series:
    """Direction-agnostic standalone ROC-AUC of each feature against ``y``.

    Used in Step 5 to compare which signals separate the REAL fake-review label
    vs. only the weak proxy: a feature with high AUC under the proxy but ~0.5
    under the real label is a proxy artifact, not a deception signal.
    """
    y = pd.Series(y).reset_index(drop=True)
    out: dict[str, float] = {}
    for col in X.columns:
        values = pd.to_numeric(X[col], errors="coerce").reset_index(drop=True)
        if values.nunique() < 2 or y.nunique() < 2:
            out[col] = 0.5
            continue
        try:
            auc = roc_auc_score(y, values)
        except ValueError:
            auc = 0.5
        out[col] = max(auc, 1.0 - auc)
    return pd.Series(out).sort_values(ascending=False)


def run_classification(df: pd.DataFrame, config: Mapping | None = None,
                       positive: str = "unverified", label_col: str | None = None) -> dict:
    """End-to-end: features → label → split → grid-search KNN & RF → evaluate.

    ``label_col=None`` uses the weak ``verified_purchase`` proxy (via
    :func:`make_label`). Pass ``label_col='is_fake'`` to train on the real
    ground-truth label (Step 5) through the identical pipeline. Returns a dict
    with per-model evaluations, fitted searches, the held-out set, and feature
    names. Consumed by ``notebooks/04_modeling`` and ``notebooks/07``.
    """
    cfg = config or utils.load_config()
    utils.set_seeds(cfg["project"]["random_seed"])
    feat = features.build_feature_table(df, cfg)
    if label_col is None:
        y, mask = make_label(feat, positive=positive)
    else:
        series = feat[label_col]
        mask = series.notna()
        y = series[mask].astype("int64")
    X = design_matrix(feat).loc[mask].reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["modeling"]["test_size"],
        random_state=cfg["project"]["random_seed"], stratify=y)

    results, searches = {}, {}
    for name, (estimator, grid) in build_estimators(cfg).items():
        gs = grid_search(estimator, grid, X_train, y_train, cfg)
        ev = evaluate_classifier(gs.best_estimator_, X_test, y_test)
        ev["best_params"] = gs.best_params_
        ev["cv_best_score"] = float(gs.best_score_)
        if name == "random_forest":
            ev["importances"] = feature_importances(gs.best_estimator_, X.columns)
        results[name] = ev
        searches[name] = gs
        LOGGER.info("%s: CV %s=%.3f, test ROC-AUC=%.3f", name,
                    cfg["modeling"]["scoring"], ev["cv_best_score"], ev["roc_auc"])

    return {"results": results, "searches": searches, "feature_names": list(X.columns),
            "X_test": X_test, "y_test": y_test,
            "label": label_col or f"verified_purchase[positive={positive}]"}


# ---------------------------------------------------------------------------
# Unsupervised: PCA + clustering
# ---------------------------------------------------------------------------
def fit_pca(X, n_components: int = 2, config: Mapping | None = None, scale: bool = True):
    """Standardize (optional) then fit PCA; returns ``(pipeline, coords)``."""
    seed = (config or utils.load_config())["project"]["random_seed"]
    steps = ([("scale", StandardScaler())] if scale else []) + [("pca", PCA(n_components=n_components, random_state=seed))]
    pipe = Pipeline(steps)
    return pipe, pipe.fit_transform(X)


def fit_kmeans(X, k: int, config: Mapping | None = None, scale: bool = True):
    """Standardize (optional) then fit K-means; returns ``(pipeline, labels)``."""
    seed = (config or utils.load_config())["project"]["random_seed"]
    steps = ([("scale", StandardScaler())] if scale else []) + [("km", KMeans(n_clusters=k, random_state=seed, n_init=10))]
    pipe = Pipeline(steps)
    labels = pipe.fit_predict(X)
    return pipe, labels


def select_kmeans_k(X, k_grid: Iterable[int] | None = None, config: Mapping | None = None):
    """Silhouette score across ``k_grid``; returns ``(scores_by_k, best_k)``."""
    cfg = config or utils.load_config()
    grid = list(k_grid or cfg["modeling"]["clustering"]["kmeans_k_grid"])
    Xs = StandardScaler().fit_transform(X)
    scores = {}
    for k in grid:
        if k < 2 or k >= len(Xs):
            continue
        labels = KMeans(n_clusters=k, random_state=cfg["project"]["random_seed"], n_init=10).fit_predict(Xs)
        scores[k] = float(silhouette_score(Xs, labels))
    best_k = max(scores, key=scores.get) if scores else None
    return scores, best_k


def fit_hierarchical(X, n_clusters: int, linkage: str = "ward", scale: bool = True):
    """Agglomerative clustering labels (ward linkage by default)."""
    Xs = StandardScaler().fit_transform(X) if scale else np.asarray(X)
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(Xs)


def linkage_matrix(X, method: str = "ward", scale: bool = True) -> np.ndarray:
    """SciPy linkage matrix for a dendrogram (``viz.plot_dendrogram``)."""
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    Xs = StandardScaler().fit_transform(X) if scale else np.asarray(X)
    return scipy_linkage(Xs, method=method)


def cluster_review_styles(df: pd.DataFrame, config: Mapping | None = None,
                          n_clusters: int | None = None, text_col: str = "review_text"):
    """Cluster reviews by *style* on TF-IDF→SVD space; returns ``(labels, top_terms, vectorizer)``.

    Useful for the secondary question: do unverified reviews split into distinct
    styles (generic praise vs. detailed)?
    """
    cfg = config or utils.load_config()
    seed = cfg["project"]["random_seed"]
    matrix, vec = features.build_tfidf(df[text_col], cfg, min_df=1, max_df=1.0)
    n_comp = max(2, min(50, matrix.shape[1] - 1, matrix.shape[0] - 1))
    reduced = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(matrix)
    k = n_clusters or 3
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(reduced)
    return labels, _top_terms_per_cluster(matrix, vec, labels), vec


def _top_terms_per_cluster(matrix, vectorizer, labels, n_terms: int = 8) -> dict[int, list[str]]:
    terms = np.asarray(vectorizer.get_feature_names_out())
    dense = matrix.toarray()
    out: dict[int, list[str]] = {}
    for cluster in sorted(set(int(c) for c in labels)):
        rows = dense[np.asarray(labels) == cluster]
        mean = rows.mean(axis=0) if len(rows) else np.zeros(len(terms))
        out[cluster] = list(terms[mean.argsort()[::-1][:n_terms]])
    return out
