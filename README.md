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

## Project workflow and implementation

The app is a deterministic computer-vision pipeline built with Python, Streamlit, OpenCV, scikit-image, NetworkX, NumPy and Pillow. It uses neither an external AI service nor a trained classification model or OCR.

1. **Receive an image.** The app accepts PNG or JPEG images, normally one shape per image. It corrects image orientation and handles transparency and grayscale inputs.
2. **Find the foreground.** The matcher evaluates multiple threshold and contrast hypotheses, cleans obvious borders and noise, and groups stroke fragments that appear to belong together.
3. **Extract structure.** It skeletonizes the selected foreground, creates a pixel graph, and traces an ordered path where supported. Coverage information helps identify traces that may be incomplete.
4. **Describe geometry.** The path is resampled and described using endpoints, junctions, loops, bends, headings, trajectory, crossings, straight segments and circular-arc features.
5. **Retrieve and rerank.** Stage 1 retrieves a shortlist from the local catalog. Stage 2 compares shortlisted candidates in more detail; the app then calculates fit, ambiguity and a candidate/review/no-match decision.
6. **Show evidence.** The interface displays ranked results, extracted masks and skeletons, confidence and fit indicators, and a downloadable JSON processing log.

Rotation, uniform scale and traversal reversal are handled. Reflection, non-uniform stretch, arbitrary graph-edit matching and multiple shapes in one image are not supported.

### Development approach

Work on the matcher has concentrated on trace quality, geometry evidence and diagnosability:

- Image decoding was hardened for orientation, transparency, grayscale, high-bit-depth images and Unicode paths.
- Foreground extraction was extended with alternative thresholding and contrast hypotheses, border rejection and component grouping.
- Skeleton cleanup and path extraction gained topology handling, coverage checks and deterministic fallbacks.
- Geometry comparisons were refined to count turns consistently, preserve short terminal details, distinguish finite line segments from collinear returns, and compare arc presence and sweep.
- Ranking was adjusted to consider ordered bends, trajectory and local line/arc evidence rather than relying on one global silhouette score.
- Fit, candidate ambiguity and match status are reported separately; confidence is a heuristic evidence score, not an empirically calibrated probability.
- Structured processing events and per-image downloads make intermediate decisions easier to inspect without putting image pixels in ordinary logs.
- Focused regression tests cover image decoding, geometry invariances, skeleton tracing, ranking, decision boundaries and UI states.

The synthetic demo provides a reproducible way to start the UI and exercise a few simple cases. It does not provide a representative evaluation set. Improvements in tests or synthetic matching must not be presented as measured improvements in real-world recognition.

## Result interpretation and limitations

| Status | Meaning |
| --- | --- |
| Candidate found | The leading candidate passed current fit and separation rules. Verify its identity and dimensions. |
| Review required | The trace may be incomplete, fit may be weak, alternatives may be close, or geometry may be ambiguous. |
| No suitable match | No candidate met the current distance rule. This does not prove that the shape is absent from a catalog. |

Lower structural distance ranks first. Confidence, geometric fit and status are different signals; none is a measured accuracy percentage. Results support review and do not verify engineering dimensions or replace engineering approval.

Known limitations include:

- Touching labels, shadows, glare, blur and perspective can distort the extracted foreground.
- Short hooks and terminals may be lost or misinterpreted at low resolution.
- Arbitrary crossings, gaps, and over/under relationships are not fully resolved.
- Shallow arcs may resemble straight or angular paths; rasterization may split or merge line/arc primitives.
- Different catalog identities with identical geometry cannot be separated from shape pixels alone.
- An incomplete trace or a correct candidate omitted from the Stage-1 shortlist can produce a wrong result.
- Batches are processed sequentially, and difficult images can take time.
- Representative hand-drawn Top-1/Top-5 accuracy and calibrated confidence have not been established.

## Verification scope

The focused test suite contains 69 tests for functional behavior and regressions. The Streamlit UI has also been checked for basic rendering, previews and result states. The three generated synthetic examples have been smoke-tested against their expected demo entries.

These checks validate code paths and the small synthetic demo only. They are not a benchmark of a real company catalog, real photographs, or hand-drawn recognition accuracy. A defensible accuracy report requires engineer-verified labels, representative and held-out inputs, explicit treatment of ambiguous IDs and out-of-catalog shapes, and reported sample counts and failure categories.

## Privacy and data handling

Only generated synthetic shapes are included in this public repository. Do not commit a company catalog database, catalog cards or geometry, source PDFs, customer photographs, private evaluation inputs, model weights or caches derived from private catalog data without explicit redistribution approval. Ignore rules do not remove files that are already tracked, so inspect `git status` and the commit contents before publishing.

Uploads are handled with temporary files; previews, results and downloadable event details are held in Streamlit session memory. The ordinary matching flow does not permanently archive uploaded image pixels. Terminal and downloadable logs can include filenames and structural information, so review them before sharing.
