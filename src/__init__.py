"""review-deception-nlp — source package.

Every loader in this package emits a single **unified review schema** so that
all downstream analysis (cleaning, features, modeling) is identical regardless
of whether the data came from the live scraper or the McAuley/UCSD fallback
dataset. The canonical schema is defined and documented in
:mod:`src.fallback_loader`.

LABEL NOTE (read me)
--------------------
The PRIMARY label used throughout this project is Amazon's "Verified Purchase"
flag, used as a **weak proxy** for review trustworthiness. It measures
*purchase verification*, NOT ground-truth deception. Every module that touches
the label restates this caveat. Step 5 (:mod:`src.groundtruth_loader` +
``notebooks/07``) validates the proxy against a real labeled fake-review
dataset.
"""

__version__ = "0.1.0"
