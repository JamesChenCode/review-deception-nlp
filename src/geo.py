"""Geospatial views — OPTIONAL choropleths / heatmaps.

Only used if reviewer/product location data turns out to be obtainable
(``geo.enabled`` in ``config.yaml``). Kept optional so the core pipeline never
depends on heavy GDAL/GEOS native libraries.

Planned responsibilities (implemented in **Step 4**, optional)
--------------------------------------------------------------
* Aggregate suspicious-review rates by region (state/country).
* Render choropleths / heatmaps. Default backend is Plotly (no GDAL); a
  ``geopandas`` backend is available behind the optional extra.

NOTE: scaffold stub — implemented in Step 4 (optional). Backend selected by
``geo.backend``.
"""
