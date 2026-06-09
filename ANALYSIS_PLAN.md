# ANALYSIS_PLAN.md

A living map from **grading-rubric criteria** → **where each is addressed** in
this repo, plus the **stepwise build protocol** and **in-scope technique
ledger**. Keep this in sync as notebooks/modules land.

---

## 1. Rubric criteria → where addressed

| Rubric criterion | Where it's addressed | Status |
|---|---|---|
| **Publication-worthy research question** | `README.md` (primary + validation-layer questions); narrative carried in `notebooks/08_story` | scaffolded |
| **Extraordinarily complex data collection** (scraped / textual / multi-source / joined) | `src/scrape.py` (robust polite scraper: caching, checkpointing, backoff, UA rotation) + `src/fallback_loader.py` (research-dataset path) + **product-metadata join** (2nd source) → `notebooks/01_collection` | scaffolded |
| **Insightful, labeled visualizations** | `src/viz.py` (consistent labeled house style) used across `notebooks/02`–`08`; geo in `src/geo.py` | scaffolded |
| **Broad range of in-scope techniques, applied correctly** | see the technique ledger (§3) — spread across `features.py`, `models.py`, `network.py`, `geo.py` | scaffolded |
| **Compelling narrative thread** | notebooks are numbered *chapters*; `08_story` assembles the arc | scaffolded |
| **Real-world impact** | trust-classifier that flags reviews the platform's own flag misses; Step-5 reality check on the proxy → `notebooks/04`, `07` | scaffolded |
| **Honest treatment of weak proxy** | proxy caveat restated in `README.md`, every label-touching docstring, and `notebooks/07_groundtruth_validation` | scaffolded |
| **Poster + 3-min talk** | distilled from `notebooks/08_story` (figures exported via `src/viz.py`) | pending |

---

## 2. Stepwise build protocol (checkpoints)

| Step | Deliverable | Status |
|---|---|---|
| **1** | Repo scaffold + `config.yaml` + README skeleton + `requirements.txt` + this plan | ✅ done |
| **2** | `scrape.py` + `fallback_loader.py` + caching/checkpointing + unified schema; ≤5-product dry run | ✅ done |
| **3** | `clean.py` + `features.py` + `tests/` | ✅ done |
| **4** | `models.py` + `viz.py` + `network.py` + notebook chapter stubs (01–08) | ✅ done |
| **5** | `groundtruth_loader.py` + `notebooks/07` proxy-vs-real-label validation (real data downloaded, MIT) | ✅ done |

> **Env:** Python 3.14 venv; pins bumped to the 3.14 wheel line (`requirements.lock` is the exact set). Test suite: 34 passing.

> **Guardrails:** stop and ask before any real scraping beyond a ≤5-product
> dry run, before downloading external datasets (state filename + source + size
> first), before installing heavy/unusual deps, before deleting files, and
> before committing.

---

## 3. In-scope technique ledger

Only techniques the course covered are used. Each maps to a planned home.

| Technique (in-scope) | Planned home | Narrative role |
|---|---|---|
| pandas / numpy wrangling | `clean.py`, all notebooks | foundation |
| matplotlib / seaborn / plotly viz | `viz.py` | every chapter |
| Bag-of-words | `features.py` | baseline text rep |
| TF-IDF / vector-space model | `features.py` | primary text rep |
| n-grams | `features.py` | phrase-level signal |
| Sentiment (VADER) | `features.py` | tone / over-positivity signal |
| KNN (classification) | `models.py` | baseline classifier |
| K-means clustering | `models.py` | review-*style* clusters |
| Hierarchical clustering | `models.py` | nested style structure |
| PCA | `models.py` | dimensionality reduction / viz |
| Classification eval: ROC, precision-recall | `models.py` | proxy-vs-real comparison |
| Cross-validation | `models.py` | honest performance |
| Grid search | `models.py` | hyperparameter tuning |
| Ensemble (random forest) | `models.py` | main classifier + importances |
| Combining data / joins | `fallback_loader.py`, `clean.py` | multi-source enrichment |
| Geospatial (choropleth / heatmap) | `geo.py` | *optional*, if location obtainable |
| Network / graph (unsupervised) | `network.py` | suspicious-product clusters |

**Explicitly OUT of scope (not covered → not used):** transformers / BERT;
formal hypothesis-testing frameworks.

---

## 4. Behavioral signals that strengthen the weak proxy

The verified-purchase proxy is reinforced (not replaced) by behavioral
features engineered in `features.py`:

- reviewer **review-burst timing** (many reviews in a short window),
- per-product **rating-distribution skew**,
- **duplicate / near-duplicate** review text,
- **helpful-vote ratios**,
- **reviewer history breadth** (how many distinct products/categories).

Step 5 reports which of these survive against the real fake-review label and
which only correlated with purchase-verification.
