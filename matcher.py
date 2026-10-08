import os
import sys
import argparse
import json
import hashlib
import cv2
import math
import pickle
import sqlite3
import tempfile
import warnings
import logging
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from importlib.metadata import version as package_version
import numpy as np
import networkx as nx
from PIL import Image, ImageOps, UnidentifiedImageError

from skimage.morphology import skeletonize
from shape_features import primitives, primitive_distance, primitive_distance_components, match_decision, apply_confidence, MIN_GEOMETRIC_FIT


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = os.environ.get("BBS_CATALOG_DB", str(PROJECT_ROOT / "catalog.db"))

# Derived signatures only; preserve the original cache and catalog assets.
CACHE_PATH = str(PROJECT_ROOT / "data/catalog_structure_v6.pkl")
CACHE_VERSION = "bbs_v11_finite_line_segments"

STAGE1_SHORTLIST = 35
TOP_K = 10

# Confidence threshold for automatic match confirmation vs review
CONFIDENCE_THRESHOLD = 0.65

CANONICAL_CANVAS_SIZE = 320
CANONICAL_MAX_DIMENSION = 300
MAX_GRAPH_PIXELS = 120000
MAX_GRAPH_FOREGROUND_FRACTION = 0.55

# Console only: no FileHandler, pixel arrays, or persistent query archive.
MATCH_LOG = logging.getLogger("bbs.matching")
if not MATCH_LOG.handlers:
    _console = logging.StreamHandler()
    _console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    MATCH_LOG.addHandler(_console)
MATCH_LOG.propagate = False
MATCH_LOG.setLevel(getattr(logging, os.getenv("BBS_LOG_LEVEL", "INFO").upper(), logging.INFO))
_RUN = ContextVar("bbs_matching_run", default=None)
_CANDIDATE = ContextVar("bbs_matching_candidate", default=None)
_EVENTS = ContextVar("bbs_matching_events", default=None)


def _log_step(step, *, level=logging.INFO, **details):
    run = _RUN.get()
    events = _EVENTS.get()
    capture = events is not None and level >= logging.INFO
    if run is None or (not capture and not MATCH_LOG.isEnabledFor(level)):
        return
    payload = dict(run=run[0], elapsed_s=round(time.perf_counter()-run[1], 4),
                   step=step, **details)
    encoded = json.dumps(payload, ensure_ascii=True,
        default=lambda value: value.tolist() if isinstance(value, np.ndarray)
        else value.item() if isinstance(value, np.generic) else str(value))
    if capture:
        events.append(json.loads(encoded))  # detached, JSON-safe snapshot
    MATCH_LOG.log(level, encoded)


def _log_structure(signature):
    return {key: signature.get(key) for key in (
        "endpoint_count", "junction_count", "loop_count", "crossing_count", "closed",
        "detected_threads", "raw_topology", "path_coverage", "foreground_coverage",
        "net_rotation", "total_curvature", "bends", "bend_positions", "bend_values",
        "segment_ratios", "primitives")}


# ============================================================
# 1. IMAGE -> BINARY & COLOR ANNOTATION FILTERING
# ============================================================

def to_binary(image, customer=False):
    """
    Converts image to binary foreground (white pixels on black background).
    Filters out colored text annotations (blue/red dimension labels, formulas).
    """
    if image is None:
        raise ValueError("Empty image")

    if len(image.shape) == 3:
        b, g, r = cv2.split(image)
        # Filter blue annotations (e.g. blue letter 'A' or 'R' or blue formula text)
        is_blue = (b.astype(np.int32) - np.maximum(r, g).astype(np.int32) > 30) & (b > 100)
        # Filter red annotations (e.g. red letter 'B' or 'D')
        is_red = (r.astype(np.int32) - np.maximum(b, g).astype(np.int32) > 30) & (r > 100)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        is_blue = np.zeros(image.shape[:2], dtype=bool)
        is_red = np.zeros(image.shape[:2], dtype=bool)
        gray = image.copy()

    if customer:
        # Otsu handles clean screenshots well; a locally adaptive threshold
        # provides a fallback for uneven PDF/photo illumination.
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, otsu = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        )

        def foreground_fraction(mask):
            return float(np.count_nonzero(mask)) / float(mask.size)

        otsu_fraction = foreground_fraction(otsu)
        adaptive_fraction = foreground_fraction(adaptive)
        if 0.001 <= otsu_fraction <= 0.45:
            binary = otsu
        elif 0.001 <= adaptive_fraction <= 0.45:
            binary = adaptive
        else:
            binary = otsu if otsu_fraction <= adaptive_fraction else adaptive
    else:
        mean_val = np.mean(gray)
        if mean_val > 127:  # black lines on white
            _, binary = cv2.threshold(
                gray,
                200,
                255,
                cv2.THRESH_BINARY_INV
            )
        else:  # white lines on black
            _, binary = cv2.threshold(
                gray,
                100,
                255,
                cv2.THRESH_BINARY
            )

    # Mask out colored annotations
    binary[is_blue] = 0
    binary[is_red] = 0

    return binary


# ============================================================
# 2. EXTRACT CLEAN REBAR FROM CATALOG CARD
# ============================================================

def extract_clean_card_rebar(card_image):
    """
    Extracts pristine rebar geometry from a catalog card image by cropping the
    drawing area and isolating neutral dark pixels (rebar) from colored labels.
    """
    h, w = card_image.shape[:2]
    # Crop central drawing area, avoiding headers and footers
    crop = card_image[
        int(h * 0.12) : int(h * 0.75),
        int(w * 0.06) : int(w * 0.94)
    ].copy()

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Rebar is neutral dark: low saturation and low brightness
    rebar_mask = (sat < 50) & (val < 160)

    binary = np.zeros_like(val, dtype=np.uint8)
    binary[rebar_mask] = 255

    # Filter out small disconnected noise / text artifacts
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if num_labels <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1
    max_area = stats[max_idx, cv2.CC_STAT_AREA]

    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if i == max_idx or stats[i, cv2.CC_STAT_AREA] >= max_area * 0.15:
            clean[labels == i] = 255

    return clean


# ============================================================
# 3. KEEP MAIN BAR COMPONENT
# ============================================================

def keep_main_component(binary):
    """
    Keeps the main rebar drawing while removing small isolated text letters.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if count <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1
    max_area = stats[max_idx, cv2.CC_STAT_AREA]

    result = np.zeros_like(binary)
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if i == max_idx or area >= max_area * 0.15 or max(w, h) >= 30:
            result[labels == i] = 255

    return result


def _remove_obvious_border_components(binary):
    """Remove frame-like components without discarding a drawing at an edge."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if count <= 1:
        return binary

    height, width = binary.shape[:2]
    result = binary.copy()
    image_area = float(height * width)
    for label in range(1, count):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        component_width = stats[label, cv2.CC_STAT_WIDTH]
        component_height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        touches = sum(
            (
                x == 0,
                y == 0,
                x + component_width >= width,
                y + component_height >= height,
            )
        )
        spans_frame = (
            component_width >= width * 0.85
            and component_height >= height * 0.85
        )
        is_large_border_artifact = (
            touches >= 2 and area >= image_area * 0.03
        )
        if spans_frame or is_large_border_artifact:
            result[labels == label] = 0
    return result


def _remove_small_isolated_noise(binary):
    """Remove only components too small to be useful geometry."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if count <= 1:
        return binary

    image_area = binary.shape[0] * binary.shape[1]
    min_area = max(8, int(image_area * 0.00001))
    result = np.zeros_like(binary)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if area >= min_area or max(width, height) >= 12:
            result[labels == label] = 255
    return result


def _normalize_customer_gray(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    background = cv2.GaussianBlur(gray, (0, 0), 21)
    normalized = cv2.divide(gray, background, scale=255)
    return gray, normalized


def _customer_foreground_candidates(image):
    """Build independent deterministic foreground hypotheses."""
    gray, normalized = _normalize_customer_gray(image)
    candidates = {}

    candidates["A_existing"] = to_binary(image, customer=True)
    candidates["B_adaptive"] = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )
    _, candidates["C_normalized"] = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Dark strokes are useful when a bright page has shadows or a strong
    # luminance gradient that defeats a global threshold.
    dark_threshold = int(np.percentile(normalized, 35))
    candidates["D_dark_strokes"] = np.where(
        normalized <= dark_threshold, 255, 0
    ).astype(np.uint8)

    # A local black-hat hypothesis recovers dark strokes whose absolute
    # intensity is close to the photographed page background.  The kernel is
    # deliberately broad enough to estimate the page illumination, while the
    # later component decomposition suppresses page-spanning structure.
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    _, candidates["F_local_contrast"] = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        candidates["E_ink"] = np.where(
            ((saturation > 35) & (value < 220)) |
            ((saturation <= 55) & (value < 150)),
            255,
            0
        ).astype(np.uint8)

    return candidates


def _prepare_customer_candidate(binary):
    candidate = np.where(binary > 0, 255, 0).astype(np.uint8)
    candidate = _remove_obvious_border_components(candidate)
    candidate = _remove_small_isolated_noise(candidate)

    # Fill only tiny breaks; this is intentionally weaker than a full
    # dilation so nearby labels and separate bars are not merged.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, close_kernel)
    # Closing must not silently attach a nearby letter to the drawing.
    # Disconnected fragments are joined later using explicit endpoint evidence.
    _, original_labels = cv2.connectedComponents(candidate, connectivity=8)
    count, closed_labels = cv2.connectedComponents(closed, connectivity=8)
    foreground = candidate > 0
    pairs = np.unique(np.column_stack((closed_labels[foreground],
                                      original_labels[foreground])), axis=0)
    source_counts = np.bincount(pairs[:,0], minlength=count)
    merged = source_counts > 1
    merged[0] = False
    restore = merged[closed_labels]
    closed[restore] = candidate[restore]
    return closed


def _customer_candidate_metrics(binary):
    foreground = int(np.count_nonzero(binary))
    total = int(binary.size)
    fraction = foreground / float(total) if total else 1.0
    if foreground == 0 or total == 0:
        return {
            "foreground_fraction": fraction,
            "component_count": 0,
            "skeleton_pixels": 0,
            "endpoint_count": 0,
            "junction_count": 0,
            "bbox_fraction": 0.0,
            "score": -100.0,
            "rejection_penalties": ["empty_mask"],
        }

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    components = count - 1
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max()) if components else 0
    useful_components = sum(
        1 for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= max(8, total * 0.00001)
    )
    ys, xs = np.where(binary > 0)
    bbox_fraction = (
        ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        / float(total)
    )

    if foreground > MAX_GRAPH_PIXELS or fraction > MAX_GRAPH_FOREGROUND_FRACTION:
        return {
            "foreground_fraction": fraction,
            "component_count": components,
            "useful_component_count": useful_components,
            "skeleton_pixels": 0,
            "endpoint_count": 0,
            "junction_count": 0,
            "bbox_fraction": bbox_fraction,
            "score": -8.0,
            "rejection_penalties": ["oversized_or_dense_mask"],
        }

    skeleton = make_skeleton(binary)
    skeleton_pixels = int(np.count_nonzero(skeleton))
    endpoints = 0
    junctions = 0
    if skeleton_pixels <= MAX_GRAPH_PIXELS:
        graph = skeleton_to_pixel_graph(skeleton)
        if graph.number_of_nodes() > 0:
            endpoints = sum(
                1 for node in graph if graph.degree(node) == 1
            )
            junctions = sum(
                1 for node in graph if graph.degree(node) >= 3
            )

    penalties = []
    score = 0.0
    if 0.001 <= fraction <= 0.30:
        score += 2.0
    else:
        score -= 3.0
        penalties.append("foreground_density")
    if bbox_fraction >= 0.015:
        score += 1.5
    else:
        score -= 2.0
        penalties.append("tiny_bbox")
    if useful_components <= 12:
        score += 1.0
    else:
        score -= min(4.0, 0.25 * (useful_components - 12))
        penalties.append("fragmentation")
    if largest_area >= max(20, total * 0.0001):
        score += 1.5
    else:
        score -= 1.5
        penalties.append("no_dominant_geometry")
    if skeleton_pixels >= 20:
        score += 2.0
    else:
        score -= 2.0
        penalties.append("short_skeleton")
    if fraction > 0.45:
        penalties.append("filled_background")

    return {
        "foreground_fraction": fraction,
        "component_count": components,
        "useful_component_count": useful_components,
        "skeleton_pixels": skeleton_pixels,
        "endpoint_count": endpoints,
        "junction_count": junctions,
        "bbox_fraction": bbox_fraction,
        "score": score,
        "rejection_penalties": penalties,
    }


def _component_orientation(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return 0.0
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    return float(math.atan2(vector[1], vector[0]))


def _angle_distance(angle_a, angle_b):
    difference = abs(angle_a - angle_b) % math.pi
    return min(difference, math.pi - difference)


def _component_descriptor(binary, label, stats, labels):
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    width = int(stats[label, cv2.CC_STAT_WIDTH])
    height = int(stats[label, cv2.CC_STAT_HEIGHT])
    area = int(stats[label, cv2.CC_STAT_AREA])
    # Skeletonize only this component's bounding box, with a zero border.
    # Previously every tiny annotation repeatedly skeletonized the entire photo.
    component = np.pad(np.where(labels[y:y+height, x:x+width] == label,
                                255, 0).astype(np.uint8), 1)
    skeleton = make_skeleton(component)
    graph = skeleton_to_pixel_graph(skeleton, canvas_shape=binary.shape,
                                    offset=(y-1, x-1))
    loop_count = sum(
        1 for cycle in nx.cycle_basis(graph) if len(cycle) > 18
    )
    endpoints = [node for node in graph if graph.degree(node) == 1]
    endpoint_directions = []
    for endpoint in endpoints:
        previous, current = None, endpoint
        for _ in range(6):
            options = [n for n in graph[current] if n != previous]
            if len(options) != 1:
                break
            previous, current = current, options[0]
        direction = np.asarray(endpoint, dtype=float) - current
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        endpoint_directions.append(direction.tolist())
    junctions = [node for node in graph if graph.degree(node) >= 3]
    skeleton_length = int(graph.number_of_nodes())
    if endpoints:
        endpoint_separation = max(
            math.hypot(a[1] - b[1], a[0] - b[0])
            for index, a in enumerate(endpoints)
            for b in endpoints[index + 1:]
        ) if len(endpoints) > 1 else 0.0
    else:
        endpoint_separation = 0.0
    diagonal = max(1.0, math.hypot(width, height))
    elongation = max(width, height) / float(max(1, min(width, height)))
    compactness = (area * 4.0 * math.pi) / float(max(1, width * height))
    centroid = (x + width / 2.0, y + height / 2.0)
    return {
        "label": label,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "area": area,
        "centroid": centroid,
        "skeleton_length": skeleton_length,
        "endpoint_count": len(endpoints),
        "endpoint_coordinates": endpoints,
        "endpoint_directions": endpoint_directions,
        "junction_count": len(junctions),
        "loop_count": loop_count,
        "endpoint_separation": endpoint_separation,
        "geodesic_extent": skeleton_length / diagonal,
        "elongation": elongation,
        "compactness": compactness,
        "orientation": _component_orientation(component),
        "stroke_width": area / float(max(1, skeleton_length)),
    }


def _is_page_spanning_structure(descriptor, image_shape):
    """Identify thin edge-to-edge context without relying on its orientation."""
    height, width = image_shape[:2]
    horizontal_span = descriptor["width"] / float(max(1, width))
    vertical_span = descriptor["height"] / float(max(1, height))
    touches_horizontal_edges = (
        descriptor["x"] <= max(2, int(width * 0.01))
        and descriptor["x"] + descriptor["width"] >= width - max(2, int(width * 0.01))
    )
    touches_vertical_edges = (
        descriptor["y"] <= max(2, int(height * 0.01))
        and descriptor["y"] + descriptor["height"] >= height - max(2, int(height * 0.01))
    )
    thin_horizontal = horizontal_span >= 0.78 and vertical_span <= 0.16
    thin_vertical = vertical_span >= 0.78 and horizontal_span <= 0.16
    return (
        (touches_horizontal_edges and thin_horizontal)
        or (touches_vertical_edges and thin_vertical)
    )


def _primary_component_score(descriptor, image_shape, dominant_length):
    """Score one component using geometry, not image position or shape ID."""
    image_diagonal = max(1.0, math.hypot(image_shape[1], image_shape[0]))
    relative_length = descriptor["skeleton_length"] / max(
        1.0, dominant_length
    )
    score = (
        math.log1p(descriptor["skeleton_length"]) * 2.0
        + math.log1p(descriptor["area"]) * 0.35
        + min(3.0, descriptor["geodesic_extent"])
        + min(2.0, descriptor["width"] / image_diagonal)
        + min(2.0, descriptor["height"] / image_diagonal)
        + min(2.0, relative_length * 2.0)
    )
    if 1 <= descriptor["endpoint_count"] <= 2:
        score += 2.5
    elif descriptor["endpoint_count"] == 0:
        score -= 1.0
    else:
        score -= min(3.0, (descriptor["endpoint_count"] - 2) * 0.25)
    score -= min(2.5, descriptor["junction_count"] * 0.35)
    score -= min(2.0, descriptor["loop_count"] * 0.4)
    if _is_page_spanning_structure(descriptor, image_shape):
        score -= 4.0
    return score


def _bbox_gap(first, second):
    horizontal = max(
        first["x"] - (second["x"] + second["width"]),
        second["x"] - (first["x"] + first["width"]),
        0,
    )
    vertical = max(
        first["y"] - (second["y"] + second["height"]),
        second["y"] - (first["y"] + first["height"]),
        0,
    )
    return math.hypot(horizontal, vertical)


def _endpoint_gap(first, second):
    """Distance between possible joins of two disconnected centerlines.

    Overlapping bounding boxes say nothing about stroke connectivity: a letter
    inside a bent bar's box may be far from either terminal. Use actual graph
    endpoints, independent of drawing position, orientation, or segment length.
    A detached closed component has no evidenced continuation into this path.
    """
    return min(
        (math.hypot(a[0] - b[0], a[1] - b[1])
         for a in first.get("endpoint_coordinates", [])
         for b in second.get("endpoint_coordinates", [])),
        default=float("inf"),
    )


def _continuation_join(first, second):
    """Both outward endpoint tangents must point into a proposed gap."""
    pairs = []
    for a, da in zip(first.get('endpoint_coordinates', []), first.get('endpoint_directions', [])):
        for b, db in zip(second.get('endpoint_coordinates', []), second.get('endpoint_directions', [])):
            gap = np.asarray(b, dtype=float)-a
            length = float(np.linalg.norm(gap))
            if length > 0 and np.dot(da,gap/length) >= .25 and np.dot(db,-gap/length) >= .25:
                pairs.append((length,a,b))
    return min(pairs, default=(float('inf'),None,None), key=lambda pair: pair[0])


def _detect_background_families(descriptors, image_shape):
    """Detect repeated, page-spanning component families without orientation rules."""
    if len(descriptors) < 4:
        return [], set()

    height, width = image_shape[:2]
    families = []
    background_labels = set()
    for seed in descriptors:
        family = [
            item for item in descriptors
            if _angle_distance(
                seed["orientation"], item["orientation"]
            ) <= math.radians(8)
            and item["elongation"] >= 4.0
            and item["geodesic_extent"] >= 0.25
        ]
        if len(family) < 4:
            continue
        projections = [
            (-math.sin(seed["orientation"]) * item["centroid"][0] +
             math.cos(seed["orientation"]) * item["centroid"][1])
            for item in family
        ]
        spread = max(projections) - min(projections)
        coverage = max(
            max(item["width"], item["height"]) for item in family
        ) / float(max(width, height))
        if spread < max(12.0, min(width, height) * 0.08) or coverage < 0.35:
            continue
        labels = {item["label"] for item in family}
        if not any(existing["labels"] == labels for existing in families):
            families.append({
                "orientation_degrees": math.degrees(seed["orientation"]),
                "labels": labels,
                "count": len(labels),
                "coverage": coverage,
            })
            background_labels.update(labels)
    return families, background_labels


def _component_neighbors(descriptors, image_shape):
    """Compute the unchanged symmetric proximity rule once per hypothesis."""
    boxes = np.array([[d['x'], d['y'], d['x']+d['width'], d['y']+d['height']]
                      for d in descriptors], dtype=float)
    widths = np.array([d['stroke_width'] for d in descriptors])
    labels = [d['label'] for d in descriptors]
    result = {}
    for i, box in enumerate(boxes):
        dx = np.maximum(0., np.maximum(box[0]-boxes[:,2], boxes[:,0]-box[2]))
        dy = np.maximum(0., np.maximum(box[1]-boxes[:,3], boxes[:,1]-box[3]))
        limit = np.maximum(max(8., math.hypot(*image_shape)*.025),
                           2.5*np.maximum(widths[i],widths))
        result[labels[i]] = [labels[j] for j in np.flatnonzero(np.hypot(dx,dy)<=limit)
                             if j != i]
    return result


def _component_group(descriptors, seed, excluded_labels, image_shape, neighbors=None):
    """Grow a nearby, stroke-compatible structural group from a seed."""
    neighbors = neighbors if neighbors is not None else _component_neighbors(descriptors, image_shape)
    eligible = {d['label'] for d in descriptors if d['label'] not in excluded_labels
                and .25 <= d['stroke_width']/max(seed['stroke_width'],.1) <= 4.}
    group = {seed["label"]}
    pending = [seed['label']]
    while pending:
        for label in neighbors[pending.pop()]:
            if label in eligible and label not in group:
                group.add(label)
                pending.append(label)
    return group


def _decompose_customer_foreground(binary):
    """Separate a selected threshold mask into a conservative primary object."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    descriptors = [
        _component_descriptor(binary, label, stats, labels)
        for label in range(1, count)
    ]
    if not descriptors:
        return np.zeros_like(binary), {
            "component_count_before": 0,
            "retained_count": 0,
            "rejected_count": 0,
            "background_families": [],
            "primary_object_score": -100.0,
            "absolute_plausibility": False,
            "component_reasons": [],
        }

    raw_skeleton = make_skeleton(binary)
    raw_graph = skeleton_to_pixel_graph(raw_skeleton)
    raw_skeleton_length = raw_graph.number_of_nodes()
    raw_endpoints = sum(
        1 for node in raw_graph if raw_graph.degree(node) == 1
    )
    raw_junctions = sum(
        1 for node in raw_graph if raw_graph.degree(node) >= 3
    )
    families, background_labels = _detect_background_families(
        descriptors, binary.shape
    )
    page_structure_labels = {
        descriptor["label"]
        for descriptor in descriptors
        if _is_page_spanning_structure(descriptor, binary.shape)
    }
    background_labels.update(page_structure_labels)
    neighbors = _component_neighbors(descriptors, binary.shape)
    candidates = []
    for seed in descriptors:
        if seed["label"] in background_labels:
            continue
        group = _component_group(
            descriptors, seed, background_labels, binary.shape, neighbors
        )
        members = [item for item in descriptors if item["label"] in group]
        skeleton_length = sum(item["skeleton_length"] for item in members)
        endpoints = sum(item["endpoint_count"] for item in members)
        junctions = sum(item["junction_count"] for item in members)
        area = sum(item["area"] for item in members)
        bbox_width = max(item["x"] + item["width"] for item in members) - min(
            item["x"] for item in members
        )
        bbox_height = max(item["y"] + item["height"] for item in members) - min(
            item["y"] for item in members
        )
        score = (
            math.log1p(skeleton_length) * 2.0
            + math.log1p(area) * 0.5
            + min(3.0, len(group) * 0.5)
            + min(2.0, (bbox_width + bbox_height) /
                  max(1.0, math.hypot(*binary.shape)) * 2.0)
            - min(4.0, endpoints * 0.08)
            - min(2.0, junctions * 0.04)
            - min(3.0, len(group - {seed["label"]}) * 0.05)
        )
        candidates.append((score, group, seed))

    joins = []
    if not candidates:
        selected_group = set()
        primary_score = -100.0
        primary = None
    else:
        _, _, primary = max(
            candidates,
            key=lambda item: _primary_component_score(
                item[2],
                binary.shape,
                max(descriptor["skeleton_length"] for descriptor in descriptors),
            ),
        )
        primary_score = _primary_component_score(
            primary,
            binary.shape,
            max(descriptor["skeleton_length"] for descriptor in descriptors),
        )
        selected_group = {primary["label"]}
        for item in descriptors:
            if item["label"] == primary["label"] or item["label"] in background_labels:
                continue
            gap, join_a, join_b = _continuation_join(item, primary)
            width_ratio = item["stroke_width"] / max(primary["stroke_width"], 0.1)
            related = (
                gap <= max(
                    8.0,
                    math.hypot(*binary.shape) * 0.02,
                    2.0 * max(item["stroke_width"], primary["stroke_width"]),
                )
                and 0.35 <= width_ratio <= 2.8
                and (
                    item["endpoint_count"] > 0
                    or item["elongation"] >= 2.0
                    or item["skeleton_length"] >= primary["skeleton_length"] * 0.15
                )
                and item["junction_count"] <= 2
                and item["loop_count"] <= 1
            )
            if related:
                selected_group.add(item["label"])
                a, b = join_a, join_b
                joins.append((a, b, max(1, int(round(min(
                    item["stroke_width"], primary["stroke_width"]))))))

    # Absolute plausibility is independent from being the best available mask.
    selected_members = [
        item for item in descriptors if item["label"] in selected_group
    ]
    selected_length = sum(item["skeleton_length"] for item in selected_members)
    selected_endpoints = sum(item["endpoint_count"] for item in selected_members)
    primary_metrics = primary if primary is not None else {}
    absolute_plausibility = (
        primary_score >= 7.0
        and selected_length >= 30
        and len(selected_group) <= max(8, len(descriptors) * 0.6)
    )

    retained_labels = set(selected_group) if absolute_plausibility else set()
    retained = np.isin(labels, list(retained_labels))
    retained_mask = np.where(retained, 255, 0).astype(np.uint8)
    # Retained disconnected fragments otherwise vanish when graph cleanup keeps
    # the largest component. Bridge only joins already accepted by endpoint and
    # stroke compatibility, using the thinner observed stroke width.
    if absolute_plausibility:
        for a, b, thickness in joins:
            cv2.line(retained_mask, (a[1], a[0]), (b[1], b[0]), 255, thickness)
    rejected_labels = set(range(1, count)) - retained_labels
    component_reasons = []
    for item in descriptors:
        if item["label"] in retained_labels:
            reason = "retained_as_primary_or_related"
        elif item["label"] in background_labels:
            reason = (
                "repeated_family_or_page_structure"
                if item["label"] in page_structure_labels
                else "repeated_family_or_page_structure"
            )
        elif item["label"] not in selected_group:
            reason = "not_connected_or_structurally_related_to_primary"
        else:
            reason = "primary_group_rejected_by_absolute_plausibility"
        component_reasons.append({
            "label": item["label"],
            "reason": reason,
            "area": item["area"],
            "skeleton_length": item["skeleton_length"],
            "endpoint_count": item["endpoint_count"],
            "junction_count": item["junction_count"],
        })
    return retained_mask, {
        "component_count_before": len(descriptors),
        "retained_count": len(retained_labels),
        "joined_endpoint_gaps": joins if absolute_plausibility else [],
        "rejected_count": len(rejected_labels),
        "background_families": families,
        "background_structure_labels": sorted(background_labels),
        "primary_object_score": primary_score,
        "primary_component_label": (
            primary["label"] if primary is not None else None
        ),
        "primary_component_score": primary_score,
        "primary_component_metrics": primary,
        "absolute_plausibility": absolute_plausibility,
        "raw_skeleton_pixels": raw_skeleton_length,
        "raw_endpoint_count": raw_endpoints,
        "raw_junction_count": raw_junctions,
        "raw_endpoint_limit": None,
        "global_candidate_plausibility_overridden": (
            bool(primary is not None)
            and primary_score >= 7.0
            and raw_endpoints > max(24, raw_skeleton_length * 0.015)
            and absolute_plausibility
        ),
        "reason_primary_accepted": (
            "strong individual connected stroke geometry"
            if absolute_plausibility
            else "individual component did not pass geometry checks"
        ),
        "component_reasons": component_reasons,
    }


def canonicalize_geometry(binary):
    """
    Normalize useful foreground geometry to a centered, aspect-preserving mask.
    """
    if binary is None or binary.ndim != 2:
        raise ValueError("canonicalize_geometry expects a 2-D binary mask")

    mask = np.where(binary > 0, 255, 0).astype(np.uint8)
    mask = _remove_obvious_border_components(mask)
    mask = _remove_small_isolated_noise(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros(
            (CANONICAL_CANVAS_SIZE, CANONICAL_CANVAS_SIZE), dtype=np.uint8
        )

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped = mask[y0:y1, x0:x1]

    # Close only pinholes and 1-2 px breaks; do not bridge separate geometry.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, close_kernel)

    height, width = cropped.shape[:2]
    scale = min(
        CANONICAL_MAX_DIMENSION / float(max(height, width)),
        1.0 if max(height, width) <= CANONICAL_MAX_DIMENSION else float("inf"),
    )
    if max(height, width) > CANONICAL_MAX_DIMENSION:
        scale = CANONICAL_MAX_DIMENSION / float(max(height, width))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        cropped, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    resized = np.where(resized >= 128, 255, 0).astype(np.uint8)

    canvas = np.zeros(
        (CANONICAL_CANVAS_SIZE, CANONICAL_CANVAS_SIZE), dtype=np.uint8
    )
    top = (CANONICAL_CANVAS_SIZE - resized_height) // 2
    left = (CANONICAL_CANVAS_SIZE - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


# ============================================================
# 4. SKELETONIZE
# ============================================================

def make_skeleton(binary):
    skeleton = skeletonize(binary > 0)
    return skeleton.astype(np.uint8) * 255


# ============================================================
# 5. SKELETON -> PIXEL GRAPH
# ============================================================

def skeleton_to_pixel_graph(skeleton, *, canvas_shape=None, offset=(0, 0)):
    G = nx.Graph()
    ys, xs = np.where(skeleton > 0)
    foreground_pixels = len(xs)
    shape = skeleton.shape if canvas_shape is None else canvas_shape
    total_pixels = shape[0] * shape[1]
    if (
        foreground_pixels > MAX_GRAPH_PIXELS
        or (
            total_pixels > 0
            and foreground_pixels / float(total_pixels)
            > MAX_GRAPH_FOREGROUND_FRACTION
        )
    ):
        return G
    pixels = set(zip((ys+offset[0]).tolist(), (xs+offset[1]).tolist()))

    for pixel in pixels:
        G.add_node(pixel)

    neighbours = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    for y, x in pixels:
        for dy, dx in neighbours:
            neighbour = (y + dy, x + dx)
            if neighbour not in pixels:
                continue

            if abs(dx) + abs(dy) == 2:
                p1 = (y, x + dx)
                p2 = (y + dy, x)
                if p1 in pixels or p2 in pixels:
                    continue
                weight = math.sqrt(2)
            else:
                weight = 1.0

            G.add_edge((y, x), neighbour, weight=weight)

    return G


# ============================================================
# 6. TOPOLOGICAL GRAPH CLEANING & THREAD PRUNING
# ============================================================

def clean_skeleton_graph(G, min_cycle_len=18):
    """
    Cleans skeleton graph by:
    1. Breaking tiny cycle artifacts (<= min_cycle_len) caused by skeletonization discretization.
    2. Pruning short spurs relative to the foreground extent, not canvas size.
    3. Returning the clean backbone graph and the detected number of threaded ends.
    """
    # 1. Break tiny cycle artifacts
    cycles = nx.cycle_basis(G)
    for cycle in cycles:
        if len(cycle) <= min_cycle_len:
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                    break

    # Small screenshots are not upscaled by canonicalize_geometry. An absolute
    # 18-pixel cutoff can therefore erase a substantial terminal hook. Measure
    # physical edge length against the same fraction of the occupied extent
    # that the former cutoff used on a full-size canonical shape.
    extent = max((max(n[axis] for n in G) - min(n[axis] for n in G)
                  for axis in (0, 1)), default=0) if G else 0
    spur_limit = 17.0 * min(1.0, extent / float(CANONICAL_MAX_DIMENSION))

    # 2. Prune only branches shorter than the relative cutoff.
    total_spurs = 0
    for _ in range(3):
        endpoints = [n for n in G.nodes if G.degree(n) == 1]
        junctions = [n for n in G.nodes if G.degree(n) >= 3]
        if not junctions:
            break

        pruned_nodes = set()
        for ep in endpoints:
            for j in junctions:
                if nx.has_path(G, ep, j):
                    try:
                        path = nx.shortest_path(G, ep, j, weight="weight")
                    except Exception:
                        continue
                    branch_length = sum(G[a][b].get("weight", math.dist(a, b))
                                        for a, b in zip(path, path[1:]))
                    if branch_length <= spur_limit:
                        is_pure_spur = all(G.degree(n) == 2 for n in path[1:-1])
                        if is_pure_spur:
                            total_spurs += 1
                            for n in path[:-1]:
                                pruned_nodes.add(n)
                            break

        if not pruned_nodes:
            break
        G.remove_nodes_from(pruned_nodes)

    # 3. Remove isolated degree-0 nodes
    isolated = [n for n in G.nodes if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    # Keep largest connected component
    if len(G.nodes) > 0:
        comps = list(nx.connected_components(G))
        largest = max(comps, key=len)
        G = G.subgraph(largest).copy()

    detected_threads = 0
    if total_spurs >= 2:
        detected_threads = 1
    if total_spurs >= 5:
        detected_threads = 2

    # A raster X often skeletonizes to two adjacent degree-3 pixels, not one
    # degree-4 pixel. Contract that one-pixel junction interior before tracing.
    # No shared neighbor: do not collapse a triangle or erase a small cycle.
    for node in sorted(list(G)):
        if node not in G or G.degree(node) != 3:
            continue
        partners = [other for other in G[node] if G.degree(other) == 3
                    and not (set(G[node]) & set(G[other]))]
        if len(partners) == 1:
            other = partners[0]
            for neighbor in list(G[other]):
                if neighbor != node:
                    G.add_edge(node, neighbor, weight=math.dist(node, neighbor))
            G.remove_node(other)

    return G, detected_threads


# ============================================================
# 7. TOPOLOGICAL PATH EXTRACTION (OPEN, LOOPS & STEMS)
# ============================================================

def _crossing_traversal(graph):
    """Trace an open self-crossing bar through opposite arms, using every edge.

    A crossing is not a welded junction. Accept this interpretation only if
    the entire connected graph is traversed continuously exactly once.
    Ambiguous branches retain the existing fallback and coverage diagnostic.
    """
    ends = sorted(n for n in graph if graph.degree(n) == 1)
    if len(ends) != 2 or not any(graph.degree(n) == 4 for n in graph):
        return None
    if any(graph.degree(n) not in (1,2,4) for n in graph):
        return None
    path = [ends[0]]
    used = set()
    while True:
        current = path[-1]
        options = [n for n in graph[current] if frozenset((current,n)) not in used]
        if not options:
            break
        if len(options) > 1 and len(path) > 1:
            incoming = np.asarray(current)-np.asarray(path[max(0,len(path)-6)])
            def alignment(node):
                branch = [current,node]
                while len(branch)<6 and graph.degree(branch[-1])==2:
                    branch.append(next(n for n in graph[branch[-1]] if n!=branch[-2]))
                vector = np.asarray(branch[-1])-np.asarray(current)
                return float(np.dot(incoming,vector))/max(float(np.linalg.norm(vector)),1e-9)
            nxt = max(options,key=alignment)
        else:
            nxt = options[0]
        used.add(frozenset((current,nxt)))
        path.append(nxt)
    return path if len(used)==graph.number_of_edges() and path[-1]==ends[1] else None


def extract_topological_path(G):
    """
    Extracts an ordered, continuous centerline coordinate sequence.
    Handles:
    - Standard open bars (2 endpoints, 0 loops)
    - Loops with attached stems (e.g. Shape 039 - circle + line)
    - Pure closed loops / stirrup ties (0 endpoints, >= 1 loops)
    - Multi-junction / general bars
    """
    if len(G.nodes) == 0:
        return [], 0

    endpoints = [n for n in G.nodes if G.degree(n) == 1]
    cycles = [c for c in nx.cycle_basis(G) if len(c) > 18]
    # The fallback traces one cycle. Never let arbitrary cycle-basis ordering
    # select a tiny attached annotation loop instead of the main contour.
    cycles.sort(key=lambda cycle: (-len(cycle), sorted(cycle)))
    loop_count = len(cycles)

    crossing_path = _crossing_traversal(G)
    if crossing_path is not None:
        return crossing_path, loop_count

    # Case 1: Standard open bar
    if len(endpoints) >= 2 and loop_count == 0:
        best_len = -1
        best_path = None
        for i in range(len(endpoints)):
            source = endpoints[i]
            lengths, paths = nx.single_source_dijkstra(G, source, weight="weight")
            for j in range(i + 1, len(endpoints)):
                target = endpoints[j]
                if target in lengths and lengths[target] > best_len:
                    best_len = lengths[target]
                    best_path = paths[target]
        if best_path is not None:
            return best_path, loop_count

    # Case 2: Loop with attached stem (e.g. Shape 039: circle + horizontal line)
    if loop_count >= 1 and len(endpoints) >= 1:
        cycle_nodes = set(cycles[0])
        stem_ep = endpoints[0]
        lengths, paths = nx.single_source_dijkstra(G, stem_ep, weight="weight")
        best_j = None
        min_dist = float('inf')
        for node in cycle_nodes:
            if node in lengths and lengths[node] < min_dist:
                min_dist = lengths[node]
                best_j = node

        stem_path = paths[best_j]

        cycle_sub = G.subgraph(cycle_nodes).copy()
        cycle_path = []
        curr = best_j
        visited = {curr}
        cycle_path.append(curr)
        while len(visited) < len(cycle_nodes):
            nbrs = [n for n in cycle_sub.neighbors(curr) if n not in visited]
            if not nbrs:
                break
            curr = nbrs[0]
            visited.add(curr)
            cycle_path.append(curr)
        cycle_path.append(best_j)

        # The stem runs endpoint -> junction. Append the loop at that junction,
        # never jump from the junction back to the second pixel of the stem.
        # Choose the smoother junction traversal using scale-relative tangents.
        span = max(1, min(len(stem_path) - 1, len(cycle_path) // 20))
        incoming = np.asarray(stem_path[-1]) - np.asarray(stem_path[-1-span])
        forward = np.asarray(cycle_path[span]) - np.asarray(cycle_path[0])
        backward = np.asarray(cycle_path[-1-span]) - np.asarray(cycle_path[-1])
        def alignment(vector):
            return float(np.dot(incoming, vector)) / max(np.linalg.norm(vector), 1e-9)
        if alignment(backward) > alignment(forward):
            cycle_path = cycle_path[::-1]
        full_path = stem_path + cycle_path[1:]
        return full_path, loop_count

    # Case 3: Pure closed loop (0 endpoints, 1 cycle)
    if loop_count >= 1 and len(endpoints) == 0:
        cycle_nodes = cycles[0]
        start = min(cycle_nodes, key=lambda p: (p[0], p[1]))
        cycle_sub = G.subgraph(set(cycle_nodes)).copy()
        cycle_path = [start]
        curr = start
        visited = {curr}
        while len(visited) < len(cycle_nodes):
            nbrs = [n for n in cycle_sub.neighbors(curr) if n not in visited]
            if not nbrs:
                break
            curr = nbrs[0]
            visited.add(curr)
            cycle_path.append(curr)
        cycle_path.append(start)
        return cycle_path, loop_count

    # Case 4: General fallback (longest shortest path)
    nodes = list(G.nodes)
    start = nodes[0]
    lengths, _ = nx.single_source_dijkstra(G, start, weight="weight")
    far_node = max(lengths, key=lengths.get)
    lengths2, paths2 = nx.single_source_dijkstra(G, far_node, weight="weight")
    far_node2 = max(lengths2, key=lengths2.get)
    return paths2[far_node2], loop_count


# ============================================================
# 8. RESAMPLE PATH (ARC-LENGTH EQUIDISTANT)
# ============================================================

def resample_path(path, number_points=100):
    if len(path) < 2:
        return np.empty((0, 2), dtype=np.float32)

    # Convert (y, x) -> (x, y)
    points = np.array(
        [[x, y] for y, x in path],
        dtype=np.float32
    )

    lengths = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1
    )

    cumulative = np.concatenate(
        [[0], np.cumsum(lengths)]
    )

    total = cumulative[-1]
    if total <= 0:
        return np.empty((0, 2), dtype=np.float32)

    targets = np.linspace(0, total, number_points)
    output = []

    for target in targets:
        index = np.searchsorted(cumulative, target)
        index = min(max(index, 1), len(cumulative) - 1)

        start_distance = cumulative[index - 1]
        end_distance = cumulative[index]

        if end_distance == start_distance:
            t = 0.0
        else:
            t = (target - start_distance) / (end_distance - start_distance)

        point = (
            points[index - 1] * (1 - t)
            + points[index] * t
        )
        output.append(point)

    return np.array(output, dtype=np.float32)


# ============================================================
# 9. SMOOTH PATH
# ============================================================

def smooth_path(points, window=7):
    if len(points) < window:
        return points

    result = points.copy()
    half = window // 2

    for i in range(half, len(points) - half):
        result[i] = np.mean(
            points[i - half : i + half + 1],
            axis=0
        )

    return result


# ============================================================
# 10. TURN ANGLES & BEND EXTRACTION (STAGE-1 RECALL)
# ============================================================

def calculate_turn_angles(points):
    if len(points) < 5:
        return np.array([], dtype=np.float32)

    angles = []
    step = 3

    for i in range(step, len(points) - step):
        # Integrate each local turn once. Overlapping 3-sample secants
        # counted the same corner repeatedly when extract_bends summed them.
        before = points[i] - points[i - 1]
        after = points[i + 1] - points[i]

        norm1 = np.linalg.norm(before)
        norm2 = np.linalg.norm(after)

        if norm1 < 0.001 or norm2 < 0.001:
            angles.append(0.0)
            continue

        before = before / norm1
        after = after / norm2

        cross = before[0] * after[1] - before[1] * after[0]
        dot = np.clip(np.dot(before, after), -1.0, 1.0)
        angle = math.atan2(cross, dot)
        angles.append(angle)

    return np.array(angles, dtype=np.float32)


def extract_bends(turn_angles):
    if len(turn_angles) == 0:
        return []

    threshold = math.radians(5)
    active = np.abs(turn_angles) > threshold

    groups = []
    start = None

    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            groups.append((start, i - 1))
            start = None

    if start is not None:
        groups.append((start, len(active) - 1))

    bends = []
    for start, end in groups:
        region = turn_angles[start : end + 1]
        total_angle = float(np.sum(region))
        if abs(math.degrees(total_angle)) < 20:
            continue
        bends.append(math.degrees(total_angle))

    return bends


def _ordered_path_features(points, turn_angles):
    """Return normalized segment lengths and signed bend positions."""
    if len(points) < 2:
        return [1.0], [], []

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = float(cumulative[-1])
    if total <= 0.0:
        return [1.0], [], []

    active = np.abs(turn_angles) > math.radians(5)
    groups = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            groups.append((start, index - 1))
            start = None
    if start is not None:
        groups.append((start, len(active) - 1))

    bend_positions = []
    bend_values = []
    split_positions = [0.0]
    for start, end in groups:
        value = float(np.sum(turn_angles[start:end + 1]))
        if abs(math.degrees(value)) < 20.0:
            continue
        # Curvature centroid, unlike the start of a bend region, transforms
        # exactly as s -> 1-s when the path traversal is reversed.
        indices = np.arange(start, end + 1) + 3
        weights = np.abs(turn_angles[start:end + 1])
        position = float(np.average(cumulative[indices], weights=weights) / total)
        bend_positions.append(position)
        bend_values.append(math.degrees(value))
        split_positions.append(position)
    split_positions.append(1.0)

    segment_lengths = np.diff(np.array(split_positions, dtype=np.float32))
    segment_lengths = segment_lengths / max(float(segment_lengths.sum()), 1e-6)
    return (
        segment_lengths.astype(np.float32).tolist(),
        bend_positions,
        bend_values,
    )


def _raw_graph_features(graph):
    endpoints = [node for node in graph if graph.degree(node) == 1]
    junctions = [node for node in graph if graph.degree(node) >= 3]
    cycles = [cycle for cycle in nx.cycle_basis(graph) if len(cycle) > 18]
    short_branch_lengths = []
    for endpoint in endpoints:
        for junction in junctions:
            try:
                path = nx.shortest_path(
                    graph, endpoint, junction, weight="weight"
                )
            except nx.NetworkXNoPath:
                continue
            if len(path) <= 30 and all(
                graph.degree(node) == 2 for node in path[1:-1]
            ):
                short_branch_lengths.append(len(path))
                break
    return {
        "component_count": nx.number_connected_components(graph)
        if graph.number_of_nodes() else 0,
        "endpoint_count": len(endpoints),
        "junction_count": len(junctions),
        "loop_count": len(cycles),
        "skeleton_pixels": graph.number_of_nodes(),
        "short_branch_count": len(short_branch_lengths),
        "short_branch_length": float(sum(short_branch_lengths)),
    }


# ============================================================
# 11. HEADING PROFILE & ROTATION INVARIANTS
# ============================================================

def build_heading_profile(points):
    """
    Computes normalized unwrapped heading profile, net rotation,
    and total absolute curvature along the resampled centerline.
    """
    if len(points) < 7:
        return np.array([], dtype=np.float32), 0.0, 0.0

    headings = []
    step = 3

    for i in range(step, len(points) - step):
        vec = points[i + step] - points[i - step]
        if np.linalg.norm(vec) < 0.001:
            continue
        headings.append(math.atan2(vec[1], vec[0]))

    if not headings:
        return np.array([], dtype=np.float32), 0.0, 0.0

    headings = np.unwrap(np.array(headings, dtype=np.float32))

    net_rotation = math.degrees(headings[-1] - headings[0])
    total_curvature = math.degrees(np.sum(np.abs(np.diff(headings))))

    # Rotation invariant (relative to start heading)
    norm_headings = headings - headings[0]

    return norm_headings, net_rotation, total_curvature


# ============================================================
# 12. STRUCTURAL SIGNATURE BUILDER
# ============================================================

def build_signature(binary, preview=None):
    """
    Builds rich structural signature including topology, bend sequence,
    heading profiles, rotation invariants, and thread count.
    """
    if binary is None or binary.ndim != 2 or binary.size == 0:
        return None
    foreground_fraction = np.count_nonzero(binary) / float(binary.size)
    if (
        foreground_fraction > MAX_GRAPH_FOREGROUND_FRACTION
        or np.count_nonzero(binary) > MAX_GRAPH_PIXELS
    ):
        return None

    skeleton = make_skeleton(binary)
    G = skeleton_to_pixel_graph(skeleton)
    if len(G.nodes) == 0:
        return None

    raw_topology = _raw_graph_features(G)
    # Coverage of the cleaned graph alone cannot reveal whole components
    # discarded by cleanup. Ignore tiny noise; retain evidence of other
    # substantial disconnected strokes for query-quality decisions.
    component_sizes = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    significant_sizes = [size for size in component_sizes
                         if size >= max(20, component_sizes[0] * .10)]
    foreground_coverage = component_sizes[0] / max(component_sizes[0], sum(significant_sizes))
    G_clean, detected_threads = clean_skeleton_graph(G, min_cycle_len=18)
    if len(G_clean.nodes) == 0:
        return None

    endpoints = [n for n in G_clean.nodes if G_clean.degree(n) == 1]
    junctions = [n for n in G_clean.nodes if G_clean.degree(n) >= 3]

    path, loop_count = extract_topological_path(G_clean)
    if len(path) < 2:
        return None
    path_points = np.array(
        [[x, y] for y, x in path],
        dtype=np.float32,
    )
    path_length = float(
        np.linalg.norm(np.diff(path_points, axis=0), axis=1).sum()
    )
    sampled = resample_path(path, 100)
    sampled = smooth_path(sampled, window=7)

    turns = calculate_turn_angles(sampled)
    bends = extract_bends(turns)
    segment_ratios, bend_positions, bend_values = _ordered_path_features(
        sampled, turns
    )

    heading, net_rotation, total_curvature = build_heading_profile(sampled)

    rev_sampled = sampled[::-1].copy()
    heading_rev, _, _ = build_heading_profile(rev_sampled)

    visited_edges = {frozenset((a,b)) for a,b in zip(path,path[1:]) if G_clean.has_edge(a,b)}
    coverage = len(visited_edges)/max(1,G_clean.number_of_edges())
    if preview is not None:
        overlay = cv2.cvtColor(255 - binary, cv2.COLOR_GRAY2RGB)
        xy = np.asarray(path_points, dtype=np.int32)
        cv2.polylines(overlay, [xy], False, (220, 45, 45), 1)
        cv2.circle(overlay, tuple(xy[0]), 3, (0, 160, 70), -1)
        cv2.circle(overlay, tuple(xy[-1]), 3, (40, 80, 230), -1)
        preview.update(foreground=255 - binary, skeleton=255 - skeleton,
                       ordered_path=overlay, path_coverage=coverage,
                       foreground_coverage=foreground_coverage)
    crossing_count = sum(G_clean.degree(node)==4 for node in G_clean)

    reliable_threads = (
        detected_threads
        if raw_topology.get("short_branch_count", 0) >= 5
        else 0
    )

    return {
        "endpoint_count": len(endpoints),
        "junction_count": len(junctions),
        "loop_count": loop_count,
        "closed": len(endpoints) == 0,
        "detected_threads": reliable_threads,
        "path_length": path_length,
        "net_rotation": abs(net_rotation),
        "total_curvature": total_curvature,
        "bends": bends,
        "bend_positions": bend_positions,
        "bend_values": bend_values,
        "segment_ratios": segment_ratios,
        "heading": heading,
        "heading_reverse": heading_rev,
        "trajectory": normalize_trajectory(sampled),
        "raw_topology": raw_topology,
        "primitives": primitives(resample_path(path, 100)),
        "crossing_count": crossing_count,
        "path_coverage": coverage,
        "foreground_coverage": foreground_coverage,
    }


def normalize_trajectory(points):
    """Center and normalize RMS radius, preserving all relative proportions."""
    centered = np.asarray(points, dtype=np.float64)
    centered = centered - centered.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    return centered / max(radius, 1e-12)


def trajectory_distance(query, candidate):
    """Rigid 2-D Procrustes distance, in degrees like heading DTW.

    Unit-RMS paths have residual = 2*sin(theta/2). Convert that residual
    to an angular distance; no anisotropic scaling, reflection or warping.
    Both traversals are allowed. Pure loops also allow cyclic start offsets.
    Missing legacy features return None, so old callers remain supported.
    """
    if "trajectory" not in query or "trajectory" not in candidate:
        return None
    a = np.asarray(query["trajectory"], dtype=np.float64)
    b = np.asarray(candidate["trajectory"], dtype=np.float64)
    closed = query["closed"] and candidate["closed"]
    if closed:
        # Drop the repeated closing sample before testing cyclic offsets.
        a, b = normalize_trajectory(a[:-1]), normalize_trajectory(b[:-1])
    if a.shape != b.shape or len(a) < 2:
        return None
    best = float("inf")
    for path in (b, b[::-1]):
        variants = np.stack([np.roll(path, shift, axis=0) for shift in range(len(path))]) if closed else path[None, :, :]
        # Closed form SO(2) fit. Unlike unconstrained SVD, it cannot reflect.
        dot = np.sum(variants * a, axis=(1, 2))
        cross = np.sum(variants[:, :, 0] * a[:, 1] - variants[:, :, 1] * a[:, 0], axis=1)
        correlation = np.hypot(dot, cross) / len(a)
        residual = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * correlation))
        best = min(best, float(residual.min()))
    return math.degrees(2.0 * math.asin(min(1.0, best / 2.0)))


def geometry_fingerprint(binary):
    """Conservative whole-foreground equivalence, including visible branches.

    Remove translation and quarter-turn orientation only. Identical fingerprints
    prove identical extracted pixels; near centerlines alone do not prove that
    two branched/threaded drawings are visually indistinguishable.
    """
    ys, xs = np.where(binary > 0)
    if not len(xs):
        return None
    crop = np.asarray(binary[ys.min():ys.max()+1, xs.min():xs.max()+1] > 0, dtype=np.uint8)
    variants = []
    for rotation in range(4):
        rotated = np.rot90(crop, rotation)
        variants.append(str(rotated.shape).encode("ascii") + rotated.tobytes())
    return hashlib.sha256(min(variants)).hexdigest()


# ============================================================
# 13. BEND SEQUENCE DISTANCE
# ============================================================

def bend_sequence_distance(bends_a, bends_b):
    n = len(bends_a)
    m = len(bends_b)
    insertion_cost = 60.0

    dp = np.zeros((n + 1, m + 1), dtype=np.float32)

    for i in range(1, n + 1):
        dp[i, 0] = i * insertion_cost

    for j in range(1, m + 1):
        dp[0, j] = j * insertion_cost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Scale substitution cost smoothly
            diff = abs(bends_a[i - 1] - bends_b[j - 1])
            substitution = min(80.0, diff * 0.8)
            dp[i, j] = min(
                dp[i - 1, j] + insertion_cost,
                dp[i, j - 1] + insertion_cost,
                dp[i - 1, j - 1] + substitution
            )

    return float(dp[n, m])


def ordered_bend_disagreement(query, candidate):
    """Worst unresolved bend in the best ordered alignment, both traversals.

    Reuse Stage-1 edit costs, but minimize the maximum rather than the sum:
    repeated segmentation errors should not accumulate into false absence.
    Unlike mean heading DTW, short terminal bends retain their evidence.
    """
    if min(query.get('path_coverage', 1.), candidate.get('path_coverage', 1.)) < .95:
        return 0.
    a = query.get('bends', [])
    b = candidate.get('bends', [])
    if not a or not b:
        # Missing bend events are not proof of a straight line (smooth arcs).
        return 0.
    best = float('inf')
    for oriented in (list(b), [-v for v in reversed(b)]):
        shifts = range(len(b)) if query.get('closed') and candidate.get('closed') else (0,)
        for shift in shifts:
            values = oriented[shift:] + oriented[:shift]
            dp = np.full((len(a)+1, len(values)+1), 60., dtype=float)
            dp[0,0] = 0.
            for i, x in enumerate(a, 1):
                for j, y in enumerate(values, 1):
                    dp[i,j] = min(max(dp[i-1,j],60.), max(dp[i,j-1],60.),
                                  max(dp[i-1,j-1], min(80., .8*abs(x-y))))
            best = min(best, float(dp[-1,-1]))
    return best


def _numeric_sequence_distance(values_a, values_b, scale=1.0):
    n = len(values_a)
    m = len(values_b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return scale * max(n, m)

    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    dp[:, 0] = np.arange(n + 1) * scale
    dp[0, :] = np.arange(m + 1) * scale
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = abs(values_a[i - 1] - values_b[j - 1]) * scale
            dp[i, j] = min(
                dp[i - 1, j] + scale,
                dp[i, j - 1] + scale,
                dp[i - 1, j - 1] + substitution,
            )
    return float(dp[n, m])


def _ordered_structure_distance(query, candidate):
    """Compare ordered proportions and signed bend locations both ways."""
    orientations = []
    for reverse in (False, True):
        if reverse:
            ratios = list(reversed(candidate.get("segment_ratios", [1.0])))
            positions = [
                1.0 - value
                for value in reversed(candidate.get("bend_positions", []))
            ]
            bends = [
                -value for value in reversed(
                    candidate.get("bend_values", candidate["bends"])
                )
            ]
        else:
            ratios = candidate.get("segment_ratios", [1.0])
            positions = candidate.get("bend_positions", [])
            bends = candidate.get("bend_values", candidate["bends"])

        ratio_distance = _numeric_sequence_distance(
            query.get("segment_ratios", [1.0]), ratios, scale=35.0
        )
        position_distance = _numeric_sequence_distance(
            query.get("bend_positions", []), positions, scale=18.0
        )
        bend_distance = _numeric_sequence_distance(
            query.get("bend_values", query["bends"]), bends, scale=0.025
        )
        orientations.append(ratio_distance + position_distance + bend_distance)
    return min(orientations)


def _complete_structure_penalty(query, candidate):
    query_raw = query.get("raw_topology", {})
    candidate_raw = candidate.get("raw_topology", {})
    penalty = 0.0

    penalty += abs(
        query["endpoint_count"] - candidate["endpoint_count"]
    ) * 45.0
    penalty += abs(
        query["junction_count"] - candidate["junction_count"]
    ) * 35.0
    penalty += abs(
        query["loop_count"] - candidate["loop_count"]
    ) * 120.0

    q_threads = query.get("detected_threads", 0)
    c_threads = candidate.get("detected_threads", 0)
    if q_threads == 0 and c_threads == 0:
        query_branches = query_raw.get("short_branch_count", 0)
        candidate_branches = candidate_raw.get("short_branch_count", 0)
        penalty += abs(query_branches - candidate_branches) * 1.5
        penalty += max(0, candidate_branches - query_branches) * 2.5

    query_components = query_raw.get("component_count", 1)
    candidate_components = candidate_raw.get("component_count", 1)
    if q_threads == 0 and c_threads == 0:
        penalty += max(0, candidate_components - query_components) * 3.0
    return penalty


# ============================================================
# 14. STAGE 1: TOPOLOGY & CURVATURE CANDIDATE GENERATOR
# ============================================================

def compare_stage1(query, candidate):
    """
    Stage-1 fast coarse candidate filter based on topology, loop count,
    endpoints, curvature difference, and bend edit distance.
    """
    loop_penalty = abs(
        query["loop_count"] - candidate["loop_count"]
    ) * 250

    closed_penalty = (
        350
        if (query["closed"] != candidate["closed"])
        else 0
    )

    endpoint_penalty = abs(
        min(query["endpoint_count"], 2) - min(candidate["endpoint_count"], 2)
    ) * 120

    curvature_penalty = abs(
        query["total_curvature"] - candidate["total_curvature"]
    ) * 0.15

    candidate_forward = candidate["bends"]
    candidate_reverse = [-angle for angle in reversed(candidate["bends"])]

    forward_distance = bend_sequence_distance(
        query["bends"],
        candidate_forward
    )
    reverse_distance = bend_sequence_distance(
        query["bends"],
        candidate_reverse
    )

    bend_distance = min(forward_distance, reverse_distance)
    geometry_distance = trajectory_distance(query, candidate)
    if geometry_distance is not None:
        # Retrieval: continuous agreement can rescue a fragmented bend list.
        # Topology contradictions still apply; shortlist remains broad (35).
        bend_distance = min(bend_distance, geometry_distance)
    return (
        loop_penalty
        + closed_penalty
        + endpoint_penalty
        + curvature_penalty
        + bend_distance
        + 20.0 * abs(query.get("crossing_count",0)-candidate.get("crossing_count",0))
        + _arc_evidence_distance(query, candidate, geometry_distance)
    )


def _arc_evidence_distance(query, candidate, geometry_distance):
    distance = primitive_distance_components(query, candidate)['arc_geometry']
    # The same raster arc can split into arc+short-line after rotation. Keep
    # partition/sweep uncertainty bounded when both paths contain an arc, but
    # never erase the stronger contradiction of an arc versus no arc at all.
    both_arcs = all(any(s['kind']=='arc' for s in sig.get('primitives',{}).get('sequence',[]))
                    for sig in (query,candidate))
    return min(distance, geometry_distance) if both_arcs and geometry_distance is not None else distance


# ============================================================
# 15. ANGULAR DIFFERENCE & HEADING DTW
# ============================================================

def angular_difference(angle_a, angle_b):
    diff = angle_a - angle_b
    return abs(math.atan2(math.sin(diff), math.cos(diff)))


def heading_dtw(profile_a, profile_b):
    if len(profile_a) == 0 or len(profile_b) == 0:
        return 9999.0

    # The stored profiles use the first tangent as their zero direction.
    # Hand-drawn endpoint wobble therefore adds a constant offset to the
    # entire profile. Fit one SO(2) offset over fixed arc-length positions
    # before warping, rather than letting DTW hide it by moving bends.
    # Circular least squares handles angle wrapping without reflection,
    # per-segment rotations, or changes to segment proportions.
    profile_a = np.asarray(profile_a, dtype=np.float64)
    profile_b = np.asarray(profile_b, dtype=np.float64)
    positions = np.linspace(0.0, 1.0, max(len(profile_a), len(profile_b)))
    aligned_a = np.interp(positions, np.linspace(0.0, 1.0, len(profile_a)), profile_a)
    aligned_b = np.interp(positions, np.linspace(0.0, 1.0, len(profile_b)), profile_b)
    differences = aligned_a - aligned_b
    offset = math.atan2(float(np.sin(differences).sum()),
                        float(np.cos(differences).sum()))
    profile_b = profile_b + offset

    n = len(profile_a)
    m = len(profile_b)

    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    steps = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    window = max(20, abs(n - m) + 20)

    for i in range(1, n + 1):
        start_j = max(1, i - window)
        end_j = min(m, i + window)

        for j in range(start_j, end_j + 1):
            cost = angular_difference(profile_a[i - 1], profile_b[j - 1])
            choices = [
                (dp[i - 1, j - 1], steps[i - 1, j - 1]),
                (dp[i - 1, j], steps[i - 1, j]),
                (dp[i, j - 1], steps[i, j - 1])
            ]
            best_cost, best_steps = min(choices, key=lambda x: x[0])
            dp[i, j] = best_cost + cost
            steps[i, j] = best_steps + 1

    if not np.isfinite(dp[n, m]):
        return 9999.0

    path_length = max(1, steps[n, m])
    return math.degrees(dp[n, m] / path_length)


# ============================================================
# 16. STAGE 2: STRUCTURAL VERIFICATION & CALIBRATED SCORING
# ============================================================

def stage2_distance_components(query, candidate, candidate_threads=0):
    """Return the exact additive terms used by Stage-2, in score order.

    Diagnostics must report the applied trajectory/legacy term rather than
    accidentally attributing a score to a feature that is not being used.
    This is also the single scoring source for compare_stage2.
    """
    scores = [
        heading_dtw(query["heading"], candidate["heading"]),
        heading_dtw(query["heading"], candidate["heading_reverse"]),
        heading_dtw(query["heading_reverse"], candidate["heading"]),
        heading_dtw(query["heading_reverse"], candidate["heading_reverse"])
    ]
    raw_dtw = min(scores)

    # Rotation and curvature penalties
    rot_penalty = abs(query["net_rotation"] - candidate["net_rotation"]) / 180.0 * 10.0
    curv_penalty = abs(query["total_curvature"] - candidate["total_curvature"]) / 180.0 * 8.0
    loop_penalty = abs(query["loop_count"] - candidate["loop_count"]) * 30.0

    # Complete topology and ordered geometry are evaluated independently of
    # heading DTW so a similar backbone cannot erase extra structure.
    complete_penalty = _complete_structure_penalty(query, candidate)
    geometry_distance = trajectory_distance(query, candidate)
    if geometry_distance is not None:
        # Fixed arc-length correspondence retains bend spacing and terminal
        # proportions without discontinuous penalties for missing bend groups.
        sequence_penalty = geometry_distance
    else:
        # Compatibility for external callers holding a legacy signature.
        same_thread_state = (
            (query.get("detected_threads", 0) > 0)
            == (candidate.get("detected_threads", 0) > 0)
        )
        sequence_penalty = (
            _ordered_structure_distance(query, candidate)
            if same_thread_state
            and abs(len(query.get("bends", [])) - len(candidate.get("bends", []))) <= 1
            else 0.0
        )

    # Compare visual evidence symmetrically. Metadata can describe threads
    # that are absent/unresolved in the raster; it must not contradict an
    # identical query/candidate extraction (previously a 30-point penalty).
    # Legacy callers without extracted thread evidence retain metadata fallback.
    q_threads = query.get("detected_threads", 0)
    catalog_threads = candidate_threads if candidate_threads is not None else 0
    c_threads = candidate.get("detected_threads", catalog_threads)
    thread_penalty = abs(q_threads - c_threads) * 10.0
    if (q_threads == 0) != (c_threads == 0):
        thread_penalty += 20.0

    primitive_terms = primitive_distance_components(query, candidate)
    return {
        "heading_dtw": raw_dtw,
        # Angular evidence is max(mean heading mismatch, worst ordered bend).
        # Add only the excess, avoiding two full charges for the same turn.
        "ordered_bends": max(0., ordered_bend_disagreement(query, candidate)-raw_dtw),
        "rotation": rot_penalty,
        "curvature": curv_penalty,
        "loop": loop_penalty,
        "complete_structure": complete_penalty,
        "trajectory" if geometry_distance is not None else "legacy_sequence": sequence_penalty,
        "thread": thread_penalty,
        # Discrete model counts can split/merge under drawing noise. Their
        # contribution must not exceed the measured spatial disagreement.
        # Continuous arc location/sweep is independent local evidence. A close
        # global silhouette must not erase an arc-versus-straight mismatch.
        "arc_geometry": _arc_evidence_distance(query, candidate, geometry_distance),
        "line_count": min(primitive_terms['line_count'], geometry_distance)
            if geometry_distance is not None else 0.,
        "crossings": 20.0 * abs(query.get("crossing_count",0)-candidate.get("crossing_count",0)),
    }


def compare_stage2(query, candidate, candidate_threads=0):
    """Return heading DTW, total structural distance, and confidence as before."""
    components = stage2_distance_components(query, candidate, candidate_threads)
    # Retain the original left-to-right floating-point addition order.
    structural_distance = 0.0
    for value in components.values():
        structural_distance += value

    # Keep confidence below automatic certainty and make it decrease with
    # absolute distance; match_shape additionally applies a rank margin.
    confidence = 0.98 * math.exp(-max(0.0, structural_distance) / 30.0)

    _log_step("06.stage2.score", shape_id=_CANDIDATE.get(), components=components,
              total=structural_distance)

    return components["heading_dtw"], structural_distance, confidence


def diagnostic_structural_report(customer_image_path, shape_ids=()):
    """Print a temporary, read-only structural comparison for debugging."""
    image = read_customer_image(customer_image_path)
    if image is None:
        raise FileNotFoundError(customer_image_path)

    catalog = load_or_build_cache()
    query_binary = prepare_customer(image)
    query = build_signature(query_binary)
    if query is None:
        raise RuntimeError("Could not build customer structural signature")

    ranked = []
    for item in catalog:
        ranked.append((compare_stage1(query, item["signature"]), item))
    ranked.sort(key=lambda value: value[0])
    rank_by_id = {
        item["shape_id"]: rank
        for rank, (_, item) in enumerate(ranked, start=1)
    }

    def summarize(signature):
        return {
            "endpoint_count": signature["endpoint_count"],
            "junction_count": signature["junction_count"],
            "loop_count": signature["loop_count"],
            "closed": signature["closed"],
            "detected_threads": signature["detected_threads"],
            "path_length": signature.get("path_length"),
            "segment_ratios": signature.get("segment_ratios", []),
            "bend_positions": signature.get("bend_positions", []),
            "bend_values": signature.get("bend_values", []),
            "net_rotation": signature["net_rotation"],
            "total_curvature": signature["total_curvature"],
            "bends": signature["bends"],
            "bend_count": len(signature["bends"]),
            "heading_summary": {
                "length": len(signature["heading"]),
                "first": float(signature["heading"][0])
                if len(signature["heading"]) else None,
                "last": float(signature["heading"][-1])
                if len(signature["heading"]) else None,
            },
            "raw_topology": signature.get("raw_topology"),
        }

    def stage_terms(candidate, candidate_threads):
        query_forward = candidate["bends"]
        query_reverse = [-angle for angle in reversed(candidate["bends"])]
        bend_forward = bend_sequence_distance(query["bends"], query_forward)
        bend_reverse = bend_sequence_distance(query["bends"], query_reverse)
        geometry_distance = trajectory_distance(query, candidate)
        dtw_scores = [
            heading_dtw(query["heading"], candidate["heading"]),
            heading_dtw(query["heading"], candidate["heading_reverse"]),
            heading_dtw(query["heading_reverse"], candidate["heading"]),
            heading_dtw(query["heading_reverse"], candidate["heading_reverse"]),
        ]
        chosen_dtw_index = int(np.argmin(dtw_scores))
        q_threads = query.get("detected_threads", 0)
        c_threads = candidate_threads or 0
        return {
            "stage1": {
                "loop": abs(query["loop_count"] - candidate["loop_count"]) * 250,
                "closed": 350 if query["closed"] != candidate["closed"] else 0,
                "endpoints": abs(
                    min(query["endpoint_count"], 2)
                    - min(candidate["endpoint_count"], 2)
                ) * 120,
                "curvature": abs(
                    query["total_curvature"] - candidate["total_curvature"]
                ) * 0.15,
                "bend_forward": bend_forward,
                "bend_reverse": bend_reverse,
                "bend_min": min(bend_forward, bend_reverse),
                "trajectory": geometry_distance,
                "bend_or_geometry": min(bend_forward, bend_reverse, geometry_distance)
                if geometry_distance is not None else min(bend_forward, bend_reverse),
                "total": compare_stage1(query, candidate),
            },
            "stage2": {
                "applied_components": stage2_distance_components(query, candidate, candidate_threads),
                "dtw_candidates": dtw_scores,
                "raw_dtw": min(dtw_scores),
                "heading_dtw_forward": dtw_scores[0],
                "heading_dtw_reverse": dtw_scores[1],
                "heading_traversal": (
                    "query-forward/candidate-reverse"
                    if chosen_dtw_index == 1
                    else "query-forward/candidate-forward"
                    if chosen_dtw_index == 0
                    else "query-reverse"
                ),
                "rotation": abs(
                    query["net_rotation"] - candidate["net_rotation"]
                ) / 180.0 * 10.0,
                "curvature": abs(
                    query["total_curvature"] - candidate["total_curvature"]
                ) / 180.0 * 8.0,
                "loop": abs(
                    query["loop_count"] - candidate["loop_count"]
                ) * 30.0,
                "complete_structure": _complete_structure_penalty(
                    query, candidate
                ),
                "ordered_sequence_legacy": _ordered_structure_distance(query, candidate),
                "trajectory": trajectory_distance(query, candidate),
                "thread": (
                    abs(q_threads - max(
                        c_threads,
                        candidate.get("detected_threads", 0),
                    )) * 10.0
                    + (
                        20.0
                        if (q_threads == 0) != (
                            max(c_threads, candidate.get(
                                "detected_threads", 0
                            )) == 0
                        )
                        else 0.0
                    )
                ),
                "result": compare_stage2(
                    query, candidate, candidate_threads
                ),
            },
        }

    report = {"customer": summarize(query), "candidates": {}}
    for shape_id in shape_ids:
        item = next(
            (entry for entry in catalog if entry["shape_id"] == shape_id),
            None,
        )
        if item is None:
            continue
        report["candidates"][shape_id] = {
            "stage1_rank": rank_by_id.get(shape_id),
            "threads_column": item["threads"],
            "signature": summarize(item["signature"]),
            "score_terms": stage_terms(
                item["signature"],
                item["threads"],
            ),
        }

    print(json.dumps(report, indent=2, default=lambda value: value.tolist()
                     if isinstance(value, np.ndarray) else value))
    return report


# ============================================================
# 17. LOAD CATALOG DATABASE
# ============================================================

def load_catalog():
    # Read-only URI prevents an accidental empty database in another cwd.
    connection = sqlite3.connect(Path(DB_PATH).resolve().as_uri() + "?mode=ro", uri=True)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        """
    )
    tables = [row[0] for row in cursor.fetchall()]

    selected_table = None
    columns = None

    for table in tables:
        cursor.execute(f'PRAGMA table_info("{table}")')
        table_columns = [row[1] for row in cursor.fetchall()]
        if "shape_id" in table_columns:
            selected_table = table
            columns = table_columns
            break

    if selected_table is None:
        connection.close()
        raise RuntimeError("Catalog table not found.")

    card_column = "card_path" if "card_path" in columns else "geometry_path"
    category_column = "category" if "category" in columns else "NULL"
    parameters_column = "parameter_labels" if "parameter_labels" in columns else "NULL"
    threads_column = "threads" if "threads" in columns else "0"

    cursor.execute(
        f"""
        SELECT
            shape_id,
            {card_column},
            {category_column},
            {parameters_column},
            {threads_column}
        FROM "{selected_table}"
        ORDER BY shape_id
        """
    )
    rows = cursor.fetchall()
    connection.close()

    return [(shape_id, str(PROJECT_ROOT / card_path), category, parameters, threads)
            for shape_id, card_path, category, parameters, threads in rows]


def _atomic_write_bytes(destination, data):
    """Readers see either the previous complete file or the new complete file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_catalog_cache(catalog):
    """Publish a complete derived cache atomically, never a partial pickle."""
    _atomic_write_bytes(CACHE_PATH, pickle.dumps(
        {"version": CACHE_VERSION, "catalog": catalog}, protocol=pickle.HIGHEST_PROTOCOL))


def _validate_catalog_cache(catalog):
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("Empty or malformed catalog cache")
    ids = [item["shape_id"] for item in catalog]
    expected = {str(row[0]).strip() for row in load_catalog()}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Catalog cache has missing, extra, or duplicate IDs")
    for item in catalog:
        signature = item["signature"]
        for key in ("endpoint_count", "junction_count", "loop_count", "net_rotation", "total_curvature"):
            if not np.isfinite(signature[key]):
                raise ValueError("Non-finite signature value")
        points = np.asarray(signature["trajectory"])
        if points.shape != (100, 2) or not np.isfinite(points).all():
            raise ValueError("Invalid cached centerline")
        for key in ("heading", "heading_reverse", "bends", "bend_positions", "bend_values", "segment_ratios"):
            if not np.isfinite(np.asarray(signature[key], dtype=float)).all():
                raise ValueError(f"Invalid cached {key}")
        for key in ("closed", "detected_threads", "raw_topology", "primitives", "path_coverage", "crossing_count"):
            if key not in signature:
                raise ValueError(f"Missing cached {key}")
        if not 0 <= signature["path_coverage"] <= 1 or signature["crossing_count"] < 0:
            raise ValueError("Invalid cached traversal metadata")
        for key in ("threads", "category", "parameters"):
            if key not in item:
                raise ValueError(f"Missing catalog metadata {key}")


# ============================================================
# 18. BUILD CATALOG CACHE
# ============================================================

def build_catalog_cache():
    print()
    print("=" * 72)
    print("BUILDING DETERMINISTIC TRAJECTORY SIGNATURE CACHE")
    print("=" * 72)

    rows = load_catalog()
    cache = []
    total = len(rows)

    for index, (shape_id, card_path, category, parameters, threads) in enumerate(rows, start=1):
        if not os.path.exists(card_path):
            raise FileNotFoundError(f"Catalog card missing: {card_path}")

        card_img = cv2.imread(card_path)
        if card_img is None:
            raise ValueError(f"Cannot decode catalog card: {card_path}")

        # Extract clean rebar using color filtering
        binary = extract_clean_card_rebar(card_img)
        signature = build_signature(binary)

        if signature is None:
            raise ValueError(f"Cannot extract catalog signature: {shape_id}")

        cache.append({
            "shape_id": str(shape_id).strip(),
            "category": category,
            "parameters": parameters,
            "threads": threads if threads is not None else 0,
            "geometry_fingerprint": geometry_fingerprint(binary),
            "signature": signature
        })

        if index % 200 == 0 or index == total:
            print(f"Processed {index}/{total} catalog shapes...")

    _validate_catalog_cache(cache)
    write_catalog_cache(cache)

    print()
    print("Cache saved:", CACHE_PATH)
    print("Cached shapes:", len(cache))

    return cache


# ============================================================
# 19. LOAD OR BUILD CACHE
# ============================================================

def load_or_build_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as file:
                payload = pickle.load(file)

            if payload.get("version") == CACHE_VERSION:
                catalog = payload["catalog"]
                _validate_catalog_cache(catalog)
                print(f"Loaded {len(catalog)} cached CUBE shapes.")
                return _with_catalog_annotations(catalog)

            print("Old cache version found. Rebuilding cache...")
        except Exception as error:
            print(f"Could not load cache: {error}. Rebuilding cache...")

    return _with_catalog_annotations(build_catalog_cache())


def _with_catalog_annotations(catalog):
    """Optional verified annotation index, separate from original DB/assets."""
    path = PROJECT_ROOT / 'data/catalog_parameters.json'
    if not path.exists():
        return catalog
    records = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records,dict):
        raise ValueError('Catalog parameter index must map IDs to label/source records')
    unknown = set(records)-{item['shape_id'] for item in catalog}
    if unknown:
        raise ValueError(f'Unknown IDs in catalog parameter index: {sorted(unknown)}')
    result = []
    for item in catalog:
        record = records.get(item['shape_id'])
        if record is not None:
            if not isinstance(record,dict) or not record.get('source') or not isinstance(record.get('labels'),list):
                raise ValueError('Catalog parameter records require labels list and source')
            item = {**item,'parameters':record['labels'],'parameter_source':record['source']}
        result.append(item)
    return result


# ============================================================
# 20. CUSTOMER PREPROCESSING
# ============================================================

def read_customer_image(image_path):
    """Decode uploads on a white page, retaining alpha and EXIF orientation."""
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source)
            pixels = np.asarray(image)
            if pixels.dtype == np.uint16:
                image = Image.fromarray(np.round(pixels / 257.0).astype(np.uint8))
            if 'A' in image.getbands() or 'transparency' in image.info:
                rgba = image.convert('RGBA')
                page = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(page, rgba)
            rgb = np.asarray(image.convert('RGB'))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except FileNotFoundError:
        raise
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f'Cannot decode image at: {image_path}') from error


def prepare_customer(image, debug=False, debug_dir=None):
    if image is None:
        raise ValueError("Empty image")

    _log_step("02.foreground.start", image_shape=image.shape)
    candidates = _customer_foreground_candidates(image)
    evaluated = []
    for name, raw_candidate in candidates.items():
        _log_step("02.foreground.evaluate", hypothesis=name)
        candidate = _prepare_customer_candidate(raw_candidate)
        metrics = _customer_candidate_metrics(candidate)
        _log_step("02.foreground.metrics", hypothesis=name, metrics=metrics)
        evaluated.append({
            "name": name,
            "mask": candidate,
            "metrics": metrics,
        })

    evaluated.sort(
        key=lambda item: item["metrics"]["score"],
        reverse=True,
    )
    selected = evaluated[0] if evaluated else None
    attempts = []
    chosen_decomposition = None
    chosen_support = -1
    for item in evaluated:
        if item["metrics"].get("score", -100.0) < 1.0 or item["metrics"].get("skeleton_pixels", 0) < 20:
            continue
        _log_step("02.components.start", hypothesis=item["name"])
        mask, report = _decompose_customer_foreground(item["mask"])
        _log_step("02.components.done", hypothesis=item["name"],
                  plausible=report["absolute_plausibility"],
                  retained=report.get("retained_count"), rejected=report.get("rejected_count"))
        attempts.append({"candidate": item["name"], "plausible": report["absolute_plausibility"]})
        if report["absolute_plausibility"]:
            # Equal global scores are common on annotated/shadowed pages.
            # Prefer the more complete plausible stroke rather than arbitrary
            # threshold insertion order (which can select only half a bar).
            # Whole-page fragmentation penalizes labels around a valid drawing.
            # Compare the extracted primary object across ALL plausible masks;
            # a clean isolated letter must not beat the complete bar merely
            # because its threshold removed most of the drawing.
            graph, _ = clean_skeleton_graph(skeleton_to_pixel_graph(
                make_skeleton(canonicalize_geometry(mask))))
            path, _ = extract_topological_path(graph)
            visited = {frozenset((a,b)) for a,b in zip(path,path[1:]) if graph.has_edge(a,b)}
            coverage = len(visited) / max(1,graph.number_of_edges())
            primary_length = (report.get("primary_component_metrics") or {}).get("skeleton_length",0)
            support = (primary_length * coverage,
                       report.get("primary_component_score", 0.),
                       item["metrics"]["score"])
            if chosen_decomposition is None or support > chosen_support:
                selected = item
                chosen_decomposition = (mask, report)
                chosen_support = support
    selected_mask = (
        selected["mask"] if selected is not None else np.zeros(
            image.shape[:2], dtype=np.uint8
        )
    )
    selected_metrics = selected["metrics"] if selected is not None else {}
    threshold_is_plausible = (
        selected is not None
        and selected_metrics.get("score", -100.0) >= 1.0
        and selected_metrics.get("skeleton_pixels", 0) >= 20
    )
    decomposed_mask, decomposition = chosen_decomposition or _decompose_customer_foreground(
        selected_mask if threshold_is_plausible else np.zeros_like(selected_mask)
    )
    selected_is_plausible = (
        threshold_is_plausible
        and decomposition["absolute_plausibility"]
    )
    isolated_mask = (
        decomposed_mask if selected_is_plausible
        else np.zeros_like(selected_mask)
    )
    canonical = canonicalize_geometry(isolated_mask)
    _log_step("02.foreground.selected", hypothesis=selected["name"] if selected else None,
              plausible=selected_is_plausible, support=chosen_support,
              foreground_pixels=int(np.count_nonzero(canonical)))

    if debug:
        output_dir = debug_dir or "debug_customer"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "original_input.png"), image)
        for item in evaluated:
            cv2.imwrite(
                os.path.join(output_dir, f"candidate_{item['name']}.png"),
                item["mask"],
            )
        cv2.imwrite(
            os.path.join(output_dir, "selected_foreground.png"),
            selected_mask,
        )
        cv2.imwrite(
            os.path.join(output_dir, "selected_raw_candidate.png"),
            selected_mask,
        )
        background_mask = np.zeros_like(selected_mask)
        if decomposition["background_families"] or decomposition.get(
            "background_structure_labels"
        ):
            count, labels, _, _ = cv2.connectedComponentsWithStats(
                selected_mask, connectivity=8
            )
            background_labels = set()
            for family in decomposition["background_families"]:
                background_labels.update(family["labels"])
            background_labels.update(
                decomposition.get("background_structure_labels", [])
            )
            background_mask = np.where(
                np.isin(labels, list(background_labels)), 255, 0
            ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(output_dir, "background_pattern_mask.png"),
            background_mask,
        )
        rejected_mask = cv2.subtract(selected_mask, decomposed_mask)
        cv2.imwrite(
            os.path.join(output_dir, "rejected_components.png"),
            rejected_mask,
        )
        cv2.imwrite(
            os.path.join(output_dir, "retained_components.png"),
            decomposed_mask,
        )
        cv2.imwrite(
            os.path.join(output_dir, "isolated_engineering_foreground.png"),
            isolated_mask,
        )
        cv2.imwrite(
            os.path.join(output_dir, "extracted_foreground.png"),
            isolated_mask,
        )
        cv2.imwrite(os.path.join(output_dir, "canonical_binary.png"), canonical)
        cv2.imwrite(
            os.path.join(output_dir, "final_skeleton.png"),
            make_skeleton(canonical),
        )
        report = {
            "selected_candidate": selected["name"] if selected else None,
            "candidate_attempts": attempts,
            "selected_score": selected_metrics.get("score"),
            "selected_is_plausible": selected_is_plausible,
            "threshold_candidate_plausible": threshold_is_plausible,
            "selection_reason": (
                "highest-ranked threshold candidate whose primary object passed absolute plausibility"
                if selected_is_plausible
                else "best threshold candidate failed absolute primary-object plausibility"
            ),
            "raw_selected_candidate": selected["name"] if selected else None,
            **{
                key: value for key, value in decomposition.items()
                if key != "component_reasons"
            },
            "component_rejection_reasons": decomposition["component_reasons"],
            "background_families": [
                {
                    **family,
                    "labels": sorted(family["labels"]),
                }
                for family in decomposition["background_families"]
            ],
            "candidates": [
                {
                    "strategy": item["name"],
                    **item["metrics"],
                }
                for item in evaluated
            ],
        }
        with open(
            os.path.join(output_dir, "preprocessing_report.json"),
            "w",
            encoding="utf-8",
        ) as report_file:
            json.dump(report, report_file, indent=2)

    return canonical


# ============================================================
# 21. MATCH SHAPE (FULL TWO-STAGE PIPELINE)
# ============================================================

def _update_match_trace(trace, **updates):
    if trace is None:
        return
    trace["report"].update(updates)
    try:
        _atomic_write_bytes(trace["directory"] / "trace.json", json.dumps(
            trace["report"], indent=2, default=lambda value: value.tolist()).encode("utf-8"))
    except OSError as error:
        warnings.warn(f"Could not save local match trace: {error}", RuntimeWarning)


def _begin_match_trace(image_path):
    # Enabled by Start-App.ps1. Opt-in when using the Python API directly.
    destination = os.environ.get("BBS_MATCH_TRACE_DIR")
    if not destination:
        return None
    try:
        source = Path(image_path)
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        code_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        directory = Path(destination) / (digest + "-" + code_digest[:12])
        _atomic_write_bytes(directory / ("input" + source.suffix), data)
        trace = {"directory": directory, "report": {
            "input_sha256": digest, "matcher_sha256": code_digest,
            "project_root": str(PROJECT_ROOT), "cache_version": CACHE_VERSION,
            "python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__,
            "networkx": nx.__version__, "scikit_image": package_version("scikit-image"),
        }}
        _update_match_trace(trace, status="processing")
        return trace
    except OSError as error:
        warnings.warn(f"Could not preserve local query input: {error}", RuntimeWarning)
        return None


def match_shape(image_path, top_k=None, parameter_labels=None, *, preview=None, source_name=None, log_events=None):
    """Return both rankings; an explicit top_k limits only the final output.

    Omitting top_k retains the historical full-shortlist return used by the UI.
    Retrieval and confidence always use all 35 candidates, even for top_k=1.
    parameter_labels is retained for caller compatibility but is ignored:
    matching uses geometry only, including straight-line and arc features.
    """
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer or None")
    # Uploads and their derived results are never persisted by matching.
    trace = None
    # Caller-owned memory only. Keep partial events on failure for downloading.
    if log_events is not None:
        log_events.clear()
    event_token = _EVENTS.set(log_events)
    token = _RUN.set((uuid.uuid4().hex[:12], time.perf_counter()))
    _log_step("01.input", filename=Path(source_name or image_path).name,
              cache_version=CACHE_VERSION, shortlist_limit=STAGE1_SHORTLIST)
    try:
        if preview is not None:
            preview.clear()
            return _match_shape_impl(image_path, top_k, trace, parameter_labels, preview=preview)
        return _match_shape_impl(image_path, top_k, trace, parameter_labels)
    except Exception as error:
        _log_step("error", level=logging.ERROR, error_type=type(error).__name__, message=str(error))
        _update_match_trace(trace, status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _log_step("09.finished")
        _RUN.reset(token)
        _EVENTS.reset(event_token)


def _match_shape_impl(image_path, top_k, trace, parameter_labels=None, *, preview=None):
    """
    Main entry point: Matches a customer bar drawing against the CUBE Shape Catalog.
    Returns:
    - stage1_results: Top shortlist from Stage 1
    - final_results: Top shortlist after Stage 2 structural verification
    """
    image = read_customer_image(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image at: {image_path}")

    # 1. Customer structural signature
    binary = prepare_customer(image)
    _log_step("03.signature.start")
    query = build_signature(binary, preview=preview) if preview is not None else build_signature(binary)

    if trace is not None:
        try:
            for name, mask in (("binary", binary), ("skeleton", make_skeleton(binary))):
                ok, encoded = cv2.imencode(".png", mask)
                if not ok:
                    raise OSError(f"Cannot encode {name}")
                _atomic_write_bytes(trace["directory"] / (name + ".png"), encoded.tobytes())
        except OSError as error:
            warnings.warn(f"Could not save foreground trace: {error}", RuntimeWarning)
        _update_match_trace(trace, query_signature=query)

    if query is None:
        raise RuntimeError("Could not extract structural signature from query image.")
    _log_step("03.signature.done", structure=_log_structure(query))

    print()
    print("=" * 72)
    print("CUSTOMER STRUCTURE")
    print("=" * 72)
    print("Endpoints:", query["endpoint_count"])
    print("Junctions:", query["junction_count"])
    print("Loops:", query["loop_count"])
    print("Detected Threads:", query["detected_threads"])
    print("Net Rotation:", round(query["net_rotation"], 1), "deg")
    print("Total Curvature:", round(query["total_curvature"], 1), "deg")
    print("Stage-1 bends:", [round(x, 1) for x in query["bends"]])

    # 2. Load catalog cache
    _log_step("04.catalog.start")
    catalog = load_or_build_cache()
    _log_step("04.catalog.ready", entries=len(catalog), cache_version=CACHE_VERSION)

    # 3. Stage 1: Fast Candidate Retrieval
    if trace is not None:
        try:
            _update_match_trace(trace, catalog_size=len(catalog),
                cache_sha256=hashlib.sha256(Path(CACHE_PATH).read_bytes()).hexdigest())
        except OSError as error:
            warnings.warn(f"Could not fingerprint catalog cache: {error}", RuntimeWarning)
    stage1 = []
    for item in catalog:
        score = compare_stage1(query, item["signature"])
        if MATCH_LOG.isEnabledFor(logging.DEBUG):
            _log_step("05.stage1.candidate", level=logging.DEBUG,
                      shape_id=item["shape_id"], score=score,
                      structure=_log_structure(item["signature"]))
        stage1.append({
            **item,
            "stage1_score": score
        })

    stage1.sort(key=lambda x: x["stage1_score"])
    for rank, item in enumerate(stage1, start=1):
        item["stage1_rank"] = rank

    # 4. Stage 2: Structural Verification on Shortlist
    shortlist = stage1[:STAGE1_SHORTLIST]
    _log_step("05.stage1.shortlist", evaluated=len(stage1), candidates=[
        dict(shape_id=r["shape_id"], rank=r["stage1_rank"], score=r["stage1_score"])
        for r in shortlist])
    reranked = []

    for item in shortlist:
        _log_step("06.stage2.candidate", shape_id=item["shape_id"],
                  stage1_rank=item["stage1_rank"],
                  catalog_image=f"data/catalog_cards/shape_{item['shape_id']}.png",
                  structure=_log_structure(item["signature"]))
        candidate_token = _CANDIDATE.set(item["shape_id"])
        try:
            raw_dtw, struct_dist, conf = compare_stage2(
                query,
                item["signature"],
                candidate_threads=item["threads"]
            )
        finally:
            _CANDIDATE.reset(candidate_token)
        reranked.append({
            **item,
            "dtw_score": raw_dtw,
            "structural_dist": struct_dist,
            "parameter_distance": 0.0,
            "confidence": conf
        })

    # Stage 1 is retrieval, not a second vote for the same geometric evidence.
    # It breaks exact final-score ties, but does not force its winner to stay #1.
    reranked.sort(key=lambda x: (x["structural_dist"], x["stage1_rank"]))
    for rank, item in enumerate(reranked, start=1):
        item["final_rank"] = rank
    if reranked:
        # Metadata can distinguish identical drawings, but pixels cannot.
        # Count across the entire catalog, including equivalents outside the
        # shortlist; keep the ranking and all existing return fields intact.
        fingerprint = reranked[0].get("geometry_fingerprint")
        equivalents = [item["shape_id"] for item in catalog
                       if fingerprint is not None
                       and item.get("geometry_fingerprint") == fingerprint]
        if len(equivalents) > 1:
            reranked[0]["geometry_equivalents"] = equivalents
        apply_confidence(reranked)

        status, message = match_decision(reranked, query)
        _log_step("07.ranking", ordering="structural_dist ascending; stage1_rank breaks ties",
                  top5=[dict(shape_id=r["shape_id"], stage1_rank=r["stage1_rank"],
                             distance=r["structural_dist"], fit=r["fit_score"],
                             confidence=r["confidence"]) for r in reranked[:5]])
        _log_step("08.decision", status=status, reason=message,
                  winner=reranked[0]["shape_id"],
                  ambiguity_ceiling=reranked[0]["ambiguity_ceiling"],
                  relative_separation=reranked[0]["relative_separation"])
        for item in reranked:
            item["match_status"] = status
            item["match_message"] = message
            item["query_structure"] = query.get("primitives",{})
            item["query_crossings"] = query.get("crossing_count",0)
            item["query_path_coverage"] = query.get("path_coverage",1.)

    if trace is not None:
        _update_match_trace(trace, status="complete",
            stage1=[{key: item[key] for key in ("shape_id", "stage1_rank", "stage1_score")} for item in stage1],
            final=[{key: item[key] for key in ("shape_id", "final_rank", "stage1_rank", "structural_dist", "dtw_score", "confidence")} for item in reranked],
            leading_components={item["shape_id"]: {**stage2_distance_components(query, item["signature"], item["threads"]),
                                 "parameters": item.get("parameter_distance",0.)}
                                for item in reranked[:TOP_K]})
    return stage1, reranked if top_k is None else reranked[:top_k]


# ============================================================
# 22. MAIN / CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Local deterministic CUBE shape matcher")
    parser.add_argument("image_path", help="Customer image to match")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    image_path = args.image_path

    print()
    print("=" * 72)
    print("BBS TWO-STAGE STRUCTURAL SHAPE MATCHER")
    print("=" * 72)
    print(f"Query Image: {image_path}")

    stage1, final = match_shape(image_path, top_k=args.top_k)

    print()
    print("=" * 72)
    print("STAGE 1 — TOPOLOGY SHORTLIST")
    print("=" * 72)
    for rank, item in enumerate(stage1[:TOP_K], start=1):
        print(
            f"#{rank:<2} "
            f"Shape ID: {item['shape_id']:<12} "
            f"Category: {str(item['category']):<26} "
            f"Score: {item['stage1_score']:.2f}"
        )

    print()
    print("=" * 72)
    print("STAGE 2 — STRUCTURAL DISCRIMINATOR")
    print("=" * 72)
    for rank, item in enumerate(final[:TOP_K], start=1):
        print(
            f"#{rank:<2} "
            f"Shape ID: {item['shape_id']:<12} "
            f"Confidence: {item['confidence']*100:<5.1f}% "
            f"StructDist: {item['structural_dist']:<6.2f} "
            f"DTW: {item['dtw_score']:<5.2f} "
            f"S1Rank: {item['stage1_rank']:<3} "
            f"Threads: {item['threads']}"
        )

    if final:
        best = final[0]
        second_best = final[1] if len(final) > 1 else None
        margin = (
            (second_best["structural_dist"] - best["structural_dist"])
            if second_best is not None
            else 999.0
        )

        print()
        print("=" * 72)
        print(f"FINAL PREDICTED SHAPE ID: {best['shape_id']}")
        print(f"Confidence: {best['confidence']*100:.1f}%")
        print(f"Structural Distance: {best['structural_dist']:.2f}")
        print(f"DTW Heading Distance: {best['dtw_score']:.2f}")
        print(f"Category: {best['category']}")
        print(f"Parameters: {best['parameters']}")
        print(f"Catalog Threads: {best['threads']}")

        print(f"Status: [{best['match_status'].upper()}] - {best['match_message']}")

        print("=" * 72)


if __name__ == "__main__":
    main()
