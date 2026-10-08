"""
CUBE Visual Shape Matcher - Testing & Evaluation Interface
==========================================================
Streamlit visual testing dashboard for reinforcing bar shape matching.
Calls the match_shape() pipeline from matcher.py.
"""

import os
import json
import tempfile
import sqlite3
import pandas as pd
import streamlit as st
from PIL import Image

# Deterministic matching engine.
import matcher as vm
from shape_features import MIN_GEOMETRIC_FIT

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="CUBE Visual Shape Matcher",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean engineering dashboard (light/dark mode compatible)
st.markdown(
    """
    <style>
    /* Metric & Card styles */
    .metric-card {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .result-container {
        background-color: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 24px;
    }
    .badge-confirmed {
        display: inline-block;
        background-color: #2e7d32;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-review {
        display: inline-block;
        background-color: #ed6c02;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-error {
        display: inline-block;
        background-color: #d32f2f;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .unavailable-box {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 220px;
        background-color: rgba(128, 128, 128, 0.1);
        border: 1px dashed rgba(128, 128, 128, 0.4);
        border-radius: 8px;
        color: #888;
        font-style: italic;
        text-align: center;
        padding: 20px;
    }
    .candidate-card {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 6px;
        padding: 8px;
        text-align: center;
        background-color: rgba(128, 128, 128, 0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CATALOG DATABASE LOOKUP
# ============================================================
@st.cache_data(show_spinner=False)
def get_catalog_image_lookup():
    """
    Loads catalog shape image paths and metadata from catalog.db once.
    Prefers clean geometry image (data/catalog_geometry/) and falls back to
    card image (data/catalog_cards/).
    """
    lookup = {}
    db_path = vm.DB_PATH if hasattr(vm, "DB_PATH") else "catalog.db"

    if not os.path.exists(db_path):
        return lookup

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Inspect columns
        cursor.execute("PRAGMA table_info('shapes')")
        cols = [r[1] for r in cursor.fetchall()]

        card_col = "card_path" if "card_path" in cols else "NULL"
        geom_col = "geometry_path" if "geometry_path" in cols else "NULL"
        cat_col = "category" if "category" in cols else "NULL"
        thr_col = "threads" if "threads" in cols else "0"

        query = f"SELECT shape_id, {geom_col}, {card_col}, {cat_col}, {thr_col} FROM shapes"
        cursor.execute(query)
        for sid, geom_p, card_p, cat, thr in cursor.fetchall():
            sid_str = str(sid).strip()
            chosen_img = None
            if geom_p and os.path.exists(geom_p):
                chosen_img = geom_p
            elif card_p and os.path.exists(card_p):
                chosen_img = card_p

            lookup[sid_str] = {
                "image_path": chosen_img,
                "geometry_path": geom_p if (geom_p and os.path.exists(geom_p)) else None,
                "card_path": card_p if (card_p and os.path.exists(card_p)) else None,
                "category": cat or "N/A",
                "threads": thr if thr is not None else 0,
            }
        conn.close()
    except Exception as err:
        st.warning(f"Could not load catalog.db metadata: {err}")

    return lookup


def resolve_catalog_image(shape_id, catalog_lookup):
    """
    Resolves the actual catalog image for a given shape_id.
    First checks catalog_lookup, then checks disk directly if needed.
    Returns path or None.
    """
    sid_clean = str(shape_id).strip()
    if sid_clean in catalog_lookup and catalog_lookup[sid_clean].get("image_path"):
        return catalog_lookup[sid_clean]["image_path"]

    # Direct filesystem check if DB entry was missing or relative path issue
    for base_dir in ["data/catalog_geometry", "data/catalog_cards"]:
        candidate_p = os.path.join(base_dir, f"shape_{sid_clean}.png")
        if os.path.exists(candidate_p):
            return candidate_p

    return None


# ============================================================
# HELPER FORMATTING
# ============================================================
def format_val(val, fmt="{:.2f}", default="N/A"):
    if val is None:
        return default
    try:
        return fmt.format(val)
    except (ValueError, TypeError):
        return str(val) if str(val) else default


# ============================================================
# UI HEADER
# ============================================================
st.title("CUBE Visual Shape Matcher")
st.caption("Visual Testing & Evaluation Interface")

# Initialize session state for results and batch tracking
if "matching_results" not in st.session_state:
    st.session_state["matching_results"] = None
if "processed_count" not in st.session_state:
    st.session_state["processed_count"] = 0

catalog_lookup = get_catalog_image_lookup()

# ============================================================
# SIDEBAR / CONTROLS
# ============================================================
with st.sidebar:
    st.subheader("⚙️ Matcher Settings")
    st.write(f"**Catalog DB:** `{vm.DB_PATH}`")
    st.write(f"**Catalog Shapes:** `{len(catalog_lookup)} loaded`")
    st.write(f"**Confidence Threshold:** `{vm.CONFIDENCE_THRESHOLD * 100:.0f}%`")
    st.write(f"**Stage 1 Shortlist:** `{vm.STAGE1_SHORTLIST}`")
    st.write(f"**Top K Display:** `{vm.TOP_K}`")
    st.divider()
    st.markdown(
        """
        **Pipeline Information:**
        - **Stage 1:** Topological graph filtering (Endpoints, Junctions, Loops, Bends).
        - **Stage 2:** Structural DTW Discriminator with Net Rotation & Curvature Verification.
        """
    )


# ============================================================
# FILE UPLOAD SECTION
# ============================================================
st.subheader("Upload Reinforcement / Bar Shape Images")

uploaded_files = st.file_uploader(
    label="Upload one or multiple shape images (PNG, JPG, JPEG):",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="You can upload 1, 4, 10, 20 or more images simultaneously.",
)

col_info, col_btn = st.columns([3, 2])
with col_info:
    num_uploaded = len(uploaded_files) if uploaded_files else 0
    st.write(f"**Uploaded Images:** {num_uploaded}")

with col_btn:
    run_button = st.button(
        "🚀 RUN SHAPE MATCHING",
        type="primary",
        disabled=(num_uploaded == 0),
        width='stretch',
    )


# ============================================================
# EXECUTION (Only run when button is pressed)
# ============================================================
if run_button and uploaded_files:
    results = []
    total = len(uploaded_files)

    progress_text = st.empty()
    progress_bar = st.progress(0)

    # Use a temporary directory for safe local filepath passing to match_shape()
    with tempfile.TemporaryDirectory(prefix="bbs_test_") as temp_dir:
        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            progress_text.text(f"Processing {idx} of {total}: {uploaded_file.name}")
            progress_bar.progress(idx / total)

            # Save uploaded bytes temporarily to pass path to match_shape()
            file_extension = os.path.splitext(uploaded_file.name)[1] or ".png"
            temp_path = os.path.join(temp_dir, f"query_{idx}{file_extension}")
            file_bytes = uploaded_file.getvalue()

            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            # Call existing matcher inside isolated try-except
            item_result = {
                "filename": uploaded_file.name,
                "file_bytes": file_bytes,
                "success": False,
                "error": None,
                "stage1": None,
                "final": None,
                "best": None,
                "second_best": None,
                "structural_margin": None,
                "confidence_margin": None,
                "log_events": [],
            }

            try:
                # Run the deterministic matching pipeline.
                preview = {}
                stage1_results, final_results = vm.match_shape(temp_path,
                    preview=preview, source_name=uploaded_file.name,
                    log_events=item_result["log_events"])
                item_result["preview"] = preview

                if not final_results:
                    raise RuntimeError("Matcher returned empty final results.")

                item_result["success"] = True
                item_result["stage1"] = stage1_results
                item_result["final"] = final_results

                best = final_results[0]
                second_best = final_results[1] if len(final_results) > 1 else None

                item_result["best"] = best
                item_result["second_best"] = second_best

                if second_best is not None:
                    struct_margin = second_best.get("structural_dist", 0.0) - best.get("structural_dist", 0.0)
                    conf_margin = best.get("confidence", 0.0) - second_best.get("confidence", 0.0)
                    item_result["structural_margin"] = struct_margin
                    item_result["confidence_margin"] = conf_margin

            except Exception as ex:
                item_result["success"] = False
                item_result["error"] = str(ex)

            results.append(item_result)

    progress_text.empty()
    progress_bar.empty()

    # Store in session state so interaction with expanders doesn't re-run matching
    st.session_state["matching_results"] = results
    st.session_state["processed_count"] = total
    st.success(f"Successfully processed {total} image(s)!")


# ============================================================
# RESULTS DISPLAY
# ============================================================
results = st.session_state.get("matching_results")

if results:
    st.divider()

    total_images = len(results)
    successful_count = sum(1 for r in results if r["success"])
    failed_count = total_images - successful_count

    # --------------------------------------------------------
    # BATCH SUMMARY
    # --------------------------------------------------------
    st.subheader("📊 Batch Summary")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    sum_col1.metric("Total Images", total_images)
    sum_col2.metric("Successfully Processed", successful_count)
    sum_col3.metric("Failed", failed_count)

    # Summary Table
    summary_rows = []
    for r in results:
        if r["success"]:
            best = r["best"]
            sec = r["second_best"]
            conf = best.get("confidence", 0.0)
            status_text = {"no_match":"No suitable match", "review":"Review", "matched":"Candidate found"}.get(
                best.get("match_status"), "Review")
            summary_rows.append({
                "Input File": r["filename"],
                "Best Shape ID": "—" if best.get("match_status")=="no_match" else str(best.get("shape_id", "N/A")),
                "Confidence": f"{conf * 100:.1f}%",
                "Structural Distance": f"{best.get('structural_dist', 0.0):.2f}",
                "Second Best Shape ID": str(sec.get("shape_id", "N/A")) if sec else "N/A",
                "Structural Margin": f"{r['structural_margin']:.2f}" if r["structural_margin"] is not None else "N/A",
                "Status": status_text,
            })
        else:
            summary_rows.append({
                "Input File": r["filename"],
                "Best Shape ID": "—",
                "Confidence": "—",
                "Structural Distance": "—",
                "Second Best Shape ID": "—",
                "Structural Margin": "—",
                "Status": "Failed",
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, width='stretch', hide_index=True)

    st.divider()
    st.subheader("🔍 Detailed Matching Results")

    # --------------------------------------------------------
    # DETAILED RESULT CARDS PER IMAGE
    # --------------------------------------------------------
    for i, res in enumerate(results, start=1):
        filename = res["filename"]

        with st.container():
            st.markdown(f"#### #{i} &nbsp; `{filename}`")

            if res.get("log_events"):
                st.download_button(
                    "Download matching log", type="tertiary",
                    data=json.dumps({"format_version": 1, "filename": filename,
                                     "processing_success": res["success"],
                                     "error": res.get("error"),
                                     "events": res["log_events"]}, indent=2, ensure_ascii=False),
                    file_name=f"{os.path.splitext(os.path.basename(filename))[0]}_matching_log.json",
                    mime="application/json", key=f"matching_log_{i}",
                    on_click="ignore",
                    help="Download this image's processing steps, comparisons and decision. No image pixels are included.",
                )

            # Handle failed image
            if not res["success"]:
                st.markdown(
                    f"""
                    <div style="padding: 16px; border: 1px solid #d32f2f; border-radius: 8px; background-color: rgba(211,47,47,0.05); margin-bottom: 20px;">
                        <span class="badge-error">Processing Failed</span>
                        <p style="margin-top: 8px; margin-bottom: 4px; font-weight: 500;"><strong>File:</strong> {filename}</p>
                        <p style="color: #d32f2f; margin: 0;"><strong>Error:</strong> {res['error']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                continue

            best = res["best"]
            second_best = res["second_best"]
            stage1 = res["stage1"]
            final = res["final"]

            if res.get("preview"):
                with st.expander("See extracted shape and skeleton"):
                    preview = res["preview"]
                    columns = st.columns(3)
                    for column, key, title in zip(columns,
                            ("foreground", "skeleton", "ordered_path"),
                            ("Extracted foreground", "Skeleton", "Ordered centerline")):
                        column.image(preview[key], caption=title, width="stretch")
                    st.caption("Red shows the traced centerline; green and blue mark its ends. "
                               f"Trace covers {preview['path_coverage']:.0%} of the cleaned skeleton edges. "
                               f"Largest connected stroke contains {preview.get('foreground_coverage', 1.):.0%} "
                               "of the significant skeleton components before cleanup. "
                               "Preview stays in this session and is not saved to disk.")

            best_shape_id = str(best.get("shape_id", "N/A"))
            confidence = best.get("confidence", 0.0)
            is_confirmed = best.get("match_status")=="matched"
            no_match = best.get("match_status")=="no_match"
            if no_match:
                st.warning(best.get("match_message", "No suitable catalog match found. This may be a new shape."))
            elif best.get("match_message"):
                st.info(best["match_message"])

            # Status pill
            status_badge = (
                '<span class="badge-confirmed">CANDIDATE FOUND</span>'
                if is_confirmed
                else ('<span class="badge-review">NO SUITABLE MATCH</span>' if no_match else '<span class="badge-review">REVIEW REQUIRED</span>')
            )
            st.markdown(status_badge, unsafe_allow_html=True)
            st.write("")

            # ------------------------------------------------
            # THREE MAIN COLUMNS
            # ------------------------------------------------
            col1, col2, col3 = st.columns([1, 1, 1.2])

            # COLUMN 1: INPUT SHAPE
            with col1:
                st.markdown("**INPUT SHAPE**")
                st.image(
                    res["file_bytes"],
                    caption=f"Uploaded: {filename}",
                    width='stretch',
                )

            # COLUMN 2: BEST CUBE MATCH
            with col2:
                candidate_title = ('NEAREST CANDIDATE (NOT ACCEPTED)' if no_match else
                                   'MATCHED CANDIDATE' if is_confirmed else 'LEADING CANDIDATE (REVIEW REQUIRED)')
                st.markdown(f"**{candidate_title}: `{best_shape_id}`**")
                catalog_img_path = resolve_catalog_image(best_shape_id, catalog_lookup)

                if catalog_img_path and os.path.exists(catalog_img_path):
                    st.image(
                        catalog_img_path,
                        caption=f"Catalog Shape ID: {best_shape_id}",
                        width='stretch',
                    )
                else:
                    st.markdown(
                        '<div class="unavailable-box">Catalog image unavailable</div>',
                        unsafe_allow_html=True,
                    )

            # COLUMN 3: MATCH DETAILS
            with col3:
                st.markdown("**MATCH DETAILS**")
                with st.container():
                    c_conf1, c_conf2 = st.columns(2)
                    c_conf1.metric("Nearest ID (not accepted)" if no_match else "Candidate ID", best_shape_id)
                    c_conf2.metric("Confidence", f"{confidence * 100:.1f}%")

                    m1, m2 = st.columns(2)
                    m1.metric("Structural Distance", format_val(best.get("structural_dist"), "{:.2f}"))
                    m2.metric("DTW Score", format_val(best.get("dtw_score"), "{:.2f}"))

                    m3, m4 = st.columns(2)
                    m3.metric("Stage 1 Rank", f"#{best.get('stage1_rank', 'N/A')}")
                    m4.metric("Threads", str(best.get("threads", "N/A")))

                    st.markdown(f"**Category:** {best.get('category', 'N/A')}")
                    structure = best.get('query_structure',{})
                    st.caption(f"Estimated input structure: {structure.get('line_count','?')} straight segments, "
                        f"{structure.get('arc_count','?')} arcs, {structure.get('bend_count','?')} sharp bends, "
                        f"{best.get('query_crossings',0)} crossing points. Counts depend on drawing resolution.")
                    st.metric("Geometric fit", f"{best.get('fit_score', confidence) * 100:.1f}%")
                    st.caption(f"With a complete trace, geometric fit at or below {MIN_GEOMETRIC_FIT:.1%} "
                               "means no suitable catalog match. Low ID confidence alone means review.")
                    st.caption("Lower structural distance determines ranking. Geometric fit measures similarity; "
                               "ID confidence also accounts for competing candidates. Neither is a measured accuracy percentage.")

                    st.divider()

                    # Second Best Details
                    st.markdown("**Second Best Candidate:**")
                    if second_best is not None:
                        sec_shape_id = str(second_best.get("shape_id", "N/A"))
                        sec_conf = second_best.get("confidence", 0.0)
                        sec_dist = second_best.get("structural_dist", None)

                        s_col1, s_col2, s_col3 = st.columns(3)
                        s_col1.metric("Shape ID", sec_shape_id)
                        s_col2.metric("Confidence", f"{sec_conf * 100:.1f}%")
                        s_col3.metric("Distance", format_val(sec_dist, "{:.2f}"))

                        # Margins
                        struct_margin = res.get("structural_margin")
                        conf_margin = res.get("confidence_margin")

                        mg_col1, mg_col2 = st.columns(2)
                        mg_col1.metric("Structural Margin", format_val(struct_margin, "{:.2f}"))
                        mg_col2.metric("Confidence Margin", f"{conf_margin * 100:.1f}%" if conf_margin is not None else "N/A")
                    else:
                        st.write("No second candidate available.")

            # ------------------------------------------------
            # EXPANDER 1: TOP 5 CANDIDATES
            # ------------------------------------------------
            with st.expander("🔎 View Top 5 Candidates", expanded=False):
                top5 = final[:5]

                # Top 5 Table
                top5_table_data = []
                for item in top5:
                    top5_table_data.append({
                        "Rank": item.get("final_rank", "N/A"),
                        "Shape ID": str(item.get("shape_id", "N/A")),
                        "Confidence": f"{item.get('confidence', 0.0) * 100:.1f}%",
                        "Geometric fit": f"{item.get('fit_score', item.get('confidence', 0.0)) * 100:.1f}%",
                        "Structural Distance": f"{item.get('structural_dist', 0.0):.2f}",
                        "DTW Score": f"{item.get('dtw_score', 0.0):.2f}",
                        "Stage 1 Rank": f"#{item.get('stage1_rank', 'N/A')}",
                        "Category": item.get("category", "N/A"),
                        "Threads": item.get("threads", "N/A"),
                    })
                st.dataframe(pd.DataFrame(top5_table_data), width='stretch', hide_index=True)

                # Top 5 Visual Grid
                st.markdown("**Top 5 Catalog Shape Visual Comparison:**")
                grid_cols = st.columns(min(5, len(top5)))
                for col_idx, candidate in enumerate(top5):
                    cid = str(candidate.get("shape_id", "N/A"))
                    c_img_p = resolve_catalog_image(cid, catalog_lookup)
                    with grid_cols[col_idx]:
                        st.markdown(f"**Rank #{candidate.get('final_rank', col_idx + 1)}: `{cid}`**")
                        if c_img_p and os.path.exists(c_img_p):
                            st.image(c_img_p, width='stretch')
                        else:
                            st.markdown(
                                '<div class="unavailable-box" style="height: 120px; font-size: 0.8rem;">Catalog image unavailable</div>',
                                unsafe_allow_html=True,
                            )
                        st.caption(
                            f"Conf: {candidate.get('confidence', 0.0) * 100:.1f}% | Dist: {candidate.get('structural_dist', 0.0):.2f}"
                        )

            # ------------------------------------------------
            # EXPANDER 2: STAGE 1 DEBUG VIEW
            # ------------------------------------------------
            with st.expander("🛠️ View Stage 1 Candidates (Debug Shortlist)", expanded=False):
                st.caption(f"Showing all {len(stage1)} Stage 1 shortlisted candidates retrieved by topological filtering:")
                s1_table_data = []
                for item in stage1[:vm.STAGE1_SHORTLIST]:
                    s1_table_data.append({
                        "Stage 1 Rank": item.get("stage1_rank", "N/A"),
                        "Shape ID": str(item.get("shape_id", "N/A")),
                        "Stage 1 Score": f"{item.get('stage1_score', 0.0):.2f}",
                        "Category": item.get("category", "N/A"),
                        "Threads": item.get("threads", "N/A"),
                        "Parameters": str(item.get("parameters", "")),
                    })
                st.dataframe(pd.DataFrame(s1_table_data), width='stretch', hide_index=True)

            st.divider()
