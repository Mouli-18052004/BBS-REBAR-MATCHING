"""Create a small, deterministic catalog containing synthetic shapes only."""

from pathlib import Path
import sqlite3

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "demo_catalog.db"
GEOMETRY_DIR = ROOT / "demo_data" / "geometry"
SYNTHETIC_SHAPES = {
    "DEMO-STRAIGHT": [(120, 220), (520, 220)],
    "DEMO-ANGLE": [(140, 130), (440, 130), (440, 300)],
    "DEMO-U": [(150, 130), (150, 290), (490, 290), (490, 130)],
}


def render_shape(points):
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    polyline = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [polyline], False, (20, 20, 20), 12, cv2.LINE_AA)
    return image


def main():
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for shape_id, points in SYNTHETIC_SHAPES.items():
        filename = f"{shape_id.lower().replace('-', '_')}.png"
        relative_path = Path("demo_data") / "geometry" / filename
        if not cv2.imwrite(str(ROOT / relative_path), render_shape(points)):
            raise OSError(f"Could not write synthetic geometry: {relative_path}")
        rows.append((shape_id, str(relative_path), "synthetic", None, 0))

    with sqlite3.connect(CATALOG_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS shapes (
                shape_id TEXT PRIMARY KEY,
                card_path TEXT NOT NULL,
                category TEXT,
                parameter_labels TEXT,
                threads INTEGER
            )
            """
        )
        connection.execute("DELETE FROM shapes")
        connection.executemany(
            """
            INSERT INTO shapes
                (shape_id, card_path, category, parameter_labels, threads)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

    print(f"Created {len(rows)} synthetic demo shapes in {CATALOG_PATH}")


if __name__ == "__main__":
    main()
