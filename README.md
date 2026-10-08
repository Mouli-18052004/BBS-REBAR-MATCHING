# BBS Rebar Matching

An image-based shape matching system that compares hand-drawn sketches and screenshots with a CUBE shape catalog. It extracts foregrounds and skeletons, analyses bends, arcs, crossings and segment geometry, and ranks candidate matches. The Streamlit interface shows the top five results, confidence, extracted traces and downloadable logs for review.

## Public demo data

This repository includes only a tiny, generated demo catalog with three synthetic shapes. It contains no company catalog database, catalog cards, real catalog geometry, source PDF, customer images, model weights or derived company caches. The demo is for exploring the code and UI; it is not representative of real catalog matching performance.

The application was designed to work with an authorized catalog distribution. Do not add company catalog files to this public repository. The `.gitignore` excludes common local catalog assets and generated files.

## Run the demo

Requires Python 3.13 and PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools\create_demo_catalog.py
& .\Start-App.ps1
```

The launcher selects `demo_catalog.db`. The matcher builds its small feature cache on the first run. To run against another catalog, set `BBS_CATALOG_DB` to an authorized local database path before starting the app.

Upload one PNG or JPEG containing a single shape, then inspect the ranked results, trace preview and downloadable JSON log. Confidence and fit scores are heuristic, not calibrated accuracy probabilities or engineering approval.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Project files

- `app.py` — Streamlit interface, uploads, result views and downloadable logs.
- `matcher.py` — image processing, skeleton tracing, feature extraction and ranking.
- `shape_features.py` — geometry features, confidence and result decisions.
- `demo_catalog.db` and `demo_data/geometry/` — generated synthetic demo fixtures only.
- `tools/create_demo_catalog.py` — reproducibly creates the synthetic fixtures.
- `tests/` — focused unit and regression tests.

The MIT license applies to this project's code. The synthetic demo fixtures are generated for this repository; no company catalog data is licensed or included.
