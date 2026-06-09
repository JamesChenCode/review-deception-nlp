# ANALYSIS_PLAN.md

A living map from **grading-rubric criteria** → **where each is addressed** in
this repo, plus the **stepwise build protocol** and **in-scope technique
ledger**. Keep this in sync as notebooks/modules land.

---

## 1. Rubric criteria → where addressed

| Rubric criterion | Where it's addressed | Status |
|---|---|---|
| **Publication-worthy research question** | `README.md` (primary + validation + cross-platform questions); narrative in `notebooks/08_story` | ✅ done |
| **Extraordinarily complex data collection** (scraped / textual / multi-source / joined) | **Self-collected Steam reviews** via public API (`src/steam_collector.py`: cursor pagination, retry/backoff, gzip cache, checkpointing + `appdetails` metadata join) → `notebooks/09`; Amazon Reviews 2023 loader + product-metadata join (`src/fallback_loader.py`) → `notebooks/01`; robust Amazon scraper built but blocked by Amazon's bot wall (`src/scrape.py`). **Three real sources, two platforms.** | ✅ done |
| **Insightful, labeled visualizations** | `src/viz.py` (consistent labeled house style) across `notebooks/02`–`09`; figures exported to `reports/figures/` | ✅ done |
| **Broad range of in-scope techniques, applied correctly** | see the technique ledger (§3) — `features.py`, `models.py`, `network.py` | ✅ done |
| **Compelling narrative thread** | numbered chapters; `08_story` assembles the arc; `09` adds the cross-platform twist | ✅ done |
| **Real-world impact** | trust classifier flags reviews the platform's flag misses; reality check on the proxy → `notebooks/04`, `07`, `09` | ✅ done |
| **Honest treatment of weak proxy** | caveat restated in `README.md`, every label-touching docstring, `notebooks/07` + `09` (free-game confound, label circularity) | ✅ done |
| **Poster + 3-min talk** | distilled from `notebooks/08_story` + `reports/figures/` | pending (human) |

---

## 2. Stepwise build protocol (checkpoints)

| Step | Deliverable | Status |
|---|---|---|
| **1** | Repo scaffold + `config.yaml` + README skeleton + `requirements.txt` + this plan | ✅ done |
| **2** | `scrape.py` + `fallback_loader.py` + caching/checkpointing + unified schema; ≤5-product dry run | ✅ done |
| **3** | `clean.py` + `features.py` + `tests/` | ✅ done |
| **4** | `models.py` + `viz.py` + `network.py` + notebook chapter stubs (01–08) | ✅ done |
| **5** | `groundtruth_loader.py` + `notebooks/07` proxy-vs-real-label validation (real data downloaded, MIT) | ✅ done |
| **+** | Notebooks 01–09 filled & **executed** with real results; `<br>`-tag cleaning fix; **self-collected Steam** source (`src/steam_collector.py`) + `notebooks/09` cross-platform; `reports/` figures + rendered HTML | ✅ done |

> **Env:** Python 3.14 venv; pins bumped to the 3.14 wheel line (`requirements.lock` is the exact set). Test suite: **41 passing**.

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
| Combining data / joins | `fallback_loader.py`, `steam_collector.py`, `clean.py` | multi-source enrichment (review + metadata) |
| API data collection (self-collected) | `steam_collector.py` | cross-platform Steam dataset (`notebooks/09`) |
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

On **Steam** the self-collected API adds even richer per-author behavior —
**playtime**, global **review count**, **games owned**, `weighted_vote_score`
(wired into the feature matrix when present). `notebooks/09` shows
`author_playtime_forever` is the top predictor of an unverified Steam review.

`notebooks/07` (ground truth) and `notebooks/09` (cross-platform) report which
of these signals survive the real fake-review label / a second platform, and
which only correlate with purchase-verification.
