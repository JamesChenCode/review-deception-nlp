# review-deception-nlp

**Analyzing deceptive / low-trust reviews on Amazon using NLP.**

A reproducible data-science project that mines the linguistic and behavioral
signatures of untrustworthy Amazon reviews, builds a "trust classifier," and
then **stress-tests that classifier against a real labeled fake-review
dataset** to see which signals actually hold up.

> ⚠️ **Label honesty (read this first).** The primary label in this project is
> Amazon's **"Verified Purchase"** flag, used as a **weak proxy** for review
> trustworthiness. It measures *purchase verification* — **not** ground-truth
> deception. A verified review can still be biased or incentivized; an
> unverified review can be perfectly honest. We strengthen the proxy with
> behavioral signals and, in **Step 5**, validate it against a real labeled
> fake-review dataset. This caveat is repeated in code, docstrings, and the
> analysis narrative wherever the label is used.

---

## Research question

**Primary.** What linguistic and behavioral signals distinguish
verified-purchase reviews from unverified ones on Amazon, and can these
signals build a trust classifier that flags suspicious reviews a platform's
verification flag misses?

**Validation layer (Step 5).** When we hold our weak-proxy findings up against
a real ground-truth fake-review dataset, which of our signals actually survive
— and where does the verified-purchase proxy mislead us?

**Secondary threads.**
- Do unverified reviews cluster into distinguishable *styles* (generic praise
  vs. detailed, experiential prose)?
- Are suspicious-review bursts concentrated on specific products or time
  windows?
- Do suspicious products cluster together in a product–reviewer network,
  replicating the unsupervised finding from the fake-review literature?

---

## Repository layout

```
review-deception-nlp/
├── README.md                  # this file
├── ANALYSIS_PLAN.md           # maps each grading-rubric criterion -> where it's addressed
├── config.yaml                # source switch, rate limits, paths, target categories
├── requirements.txt           # pinned dependencies
├── data/                      # ALL gitignored (only .gitkeep is tracked)
│   ├── raw/                   # raw scraped HTML/JSON + cache/ + checkpoints/
│   ├── interim/               # parsed-but-not-cleaned intermediate tables
│   ├── processed/             # analysis-ready feature tables
│   └── groundtruth/           # Hollenbeck et al. labeled fake-review data (Step 5)
├── src/
│   ├── scrape.py              # polite, robust Amazon scraper (PRIMARY path)
│   ├── fallback_loader.py     # McAuley/UCSD + Hou et al. 2024 loader -> unified schema
│   ├── groundtruth_loader.py  # labeled fake-review loader (Step 5)
│   ├── clean.py               # pure cleaning/normalization functions
│   ├── features.py            # TF-IDF, n-grams, sentiment, behavioral signals
│   ├── models.py              # KNN, random forest, clustering, PCA + CV/grid/ROC/PR
│   ├── network.py             # product-reviewer graph + suspicious-cluster detection
│   ├── geo.py                 # optional choropleth / heatmap helpers
│   └── viz.py                 # reusable labeled-plot helpers (consistent house style)
├── notebooks/                 # narrative chapters 01..08 (see ANALYSIS_PLAN.md)
└── tests/                     # pytest fixtures for cleaning + feature functions
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (Python 3.11 or 3.12)
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Download NLTK data used for sentiment + tokenization
python -m nltk.downloader vader_lexicon punkt punkt_tab stopwords
```

> The optional **geospatial** chapter can use Plotly choropleths (no extra
> install) or a heavier `geopandas` backend — see the commented block at the
> bottom of `requirements.txt`.

**Reproducible pins.** `requirements.txt` carries lower-bound constraints that
resolve to current Python-3.14 wheels; **`requirements.lock`** is the exact
frozen set — use `pip install -r requirements.lock` to reproduce a known-good
environment.

**Troubleshooting — NLTK `CERTIFICATE_VERIFY_FAILED`.** Some interpreters (e.g.
the python.org 3.14 framework) ship without a CA bundle, so the NLTK download
fails TLS verification. Either run the interpreter's *Install Certificates*
step, or fetch the public corpora with an unverified context **for local setup
only** (never in `src/`):

```python
import ssl, nltk
ssl._create_default_https_context = ssl._create_unverified_context
for pkg in ["vader_lexicon", "punkt", "punkt_tab", "stopwords"]:
    nltk.download(pkg)
```

---

## Data sources & the source switch

This project is **source-agnostic**: every loader emits the same *unified
review schema*, so cleaning, features, and modeling are byte-for-byte
identical no matter where the data came from. Flip one key in `config.yaml`:

```yaml
data_source:
  source: fallback     # "scrape" (live) | "fallback" (research dataset)
  auto_fallback: true  # scraper degrades to the fallback dataset on repeated failure
```

| Source     | Module                | When to use                                              |
|------------|-----------------------|---------------------------------------------------------|
| `scrape`   | `src/scrape.py`       | Live Amazon scrape (rate-limited; **blocked by a bot wall** in practice) |
| `fallback` | `src/fallback_loader.py` | Documented, ToS-safe Amazon Reviews 2023 research data |
| `steam`    | `src/steam_collector.py` | **Self-collected** cross-platform data via Steam's public API |

> **Self-collected data (note).** Amazon serves polite scrapers a JavaScript
> anti-bot interstitial (`/_sec/verify`), so real Amazon reviews aren't
> retrievable without circumventing it (out of scope). The genuinely
> *self-collected* dataset therefore comes from the **Steam reviews public API**
> (`src/steam_collector.py`): `steam_purchase` is a direct analog of Amazon's
> "Verified Purchase" proxy, `received_for_free` flags incentivized reviews, and
> the API exposes rich behavioral signals (author review count, games owned,
> playtime). Collect with `python -m src.steam_collector --collect`; analysis in
> `notebooks/09_steam_crossplatform.ipynb`. This makes the project genuinely
> cross-platform (Amazon research data + self-collected Steam).

A second source — **product metadata** (category, price, brand) — is joined
onto reviews so the dataset is genuinely multi-source. _(Schema details land
in Step 2.)_

### Downloading the fallback dataset
Files are **not** committed. Download instructions, expected filenames, and
sizes will be documented here in **Step 2** (the loader only *reads* from
`data/raw/`; it never downloads anything without an explicit, confirmed
action).

---

## Reproducibility

- **One seed to rule them all:** `project.random_seed` in `config.yaml` is
  threaded through numpy, scikit-learn, and scraper jitter.
- **Config-driven:** no magic numbers in notebooks — paths, rate limits,
  grids, and thresholds all live in `config.yaml`.
- **Deterministic pipeline order** (each notebook is a narrative chapter):

  | # | Notebook                     | Produces                                  |
  |---|------------------------------|-------------------------------------------|
  | 01 | `01_collection`             | raw → unified review table                |
  | 02 | `02_cleaning_eda`           | cleaned table + EDA                        |
  | 03 | `03_features`               | TF-IDF / n-gram / sentiment / behavioral   |
  | 04 | `04_modeling`               | trust classifier + ROC/PR evaluation       |
  | 05 | `05_clustering_pca`         | review-style clusters + PCA views          |
  | 06 | `06_geo`                    | optional choropleth / heatmap              |
  | 07 | `07_groundtruth_validation` | proxy-vs-real-label comparison             |
  | 08 | `08_story`                  | the assembled narrative for poster/talk    |

  _(Notebooks are stubbed in Step 4.)_

---

## Ethics & legality

- **Amazon's Terms of Service restrict automated scraping**, and Amazon
  actively blocks scrapers. Treat `src/scrape.py` as an *educational,
  polite-by-default* tool, not a bulk-extraction engine.
- **Rate-limiting etiquette is built in:** randomized delays, exponential
  backoff, user-agent rotation, cache-first requests (never re-fetch what we
  already have), and `robots.txt` awareness. Defaults are conservative.
- **Personal data:** reviewer identifiers are pseudonymous; we never attempt
  to deanonymize reviewers, and no scraped personal data is committed.
- **The recommended path for real analysis is the fallback research dataset**,
  which is published for research use and avoids ToS friction entirely. The
  scraper exists to demonstrate robust, ethical collection engineering; the
  `fallback` switch makes the analysis fully reproducible without it.
- **Weak-proxy honesty:** see the label caveat at the top of this README.

---

## Data-source citations

- **McAuley/UCSD Amazon Reviews dataset** — J. McAuley et al.
  <https://jmcauley.ucsd.edu/data/amazon/>
- **Amazon Reviews 2023 (newer edition)** — Hou, Y., Li, J., He, Z., Yan, A.,
  Chen, X., & McAuley, J. (2024). *Bridging Language and Items for Retrieval
  and Recommendation.* <https://amazon-reviews-2023.github.io/>
- **Labeled fake-review data (Step 5)** — He, S., Hollenbeck, B., Overgoor, G.,
  Proserpio, D., & Tosyali, A. *Detecting fake-review buyers using network
  structure* (PNAS). Data repo:
  <https://github.com/bretthollenbeck/fake-reviews-data>
  _(File format & license confirmed before the loader is written — see Step 5.)_

---

## Status

Scaffolding in progress; built incrementally against a stepwise execution
protocol. See `ANALYSIS_PLAN.md` for the rubric map and current step.

## License

See [LICENSE](LICENSE).
