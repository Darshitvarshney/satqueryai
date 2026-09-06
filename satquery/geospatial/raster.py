"""Raster inspection tools for the geo-spatial specialist.

Ports ``get_raster_metadata`` / ``analyze_spatial_properties`` /
``get_pixel_space_info`` from the notebook. rasterio is optional — if it is not
installed the helpers return an ``error`` payload and pixel-space info falls
back to PIL so plain PNG/JPEG uploads still work.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_rasterio():
    try:
        import rasterio  # type: ignore

        return rasterio
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "rasterio is not installed. Install the geospatial extras: "
            "pip install -r requirements-geo.txt"
        ) from exc


def get_raster_metadata(image_path: str | Path) -> dict[str, Any]:
    try:
        rasterio = _load_rasterio()
    except RuntimeError as exc:
        return {"error": str(exc)}
    try:
        with rasterio.open(str(image_path)) as src:
            return {
                "path": str(image_path),
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]) if src.dtypes else None,
                "crs": str(src.crs) if src.crs else None,
                "transform": [round(float(v), 10) for v in tuple(src.transform)],
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
                "resolution": {"x": src.res[0], "y": src.res[1]},
            }
    except Exception as exc:
        return {"error": str(exc)}


def analyze_spatial_properties(image_path: str | Path) -> dict[str, Any]:
    metadata = get_raster_metadata(image_path)
    if "error" in metadata:
        return {"status": "failed", "reason": metadata["error"]}

    has_crs = metadata.get("crs") is not None
    result: dict[str, Any] = {
        "image": str(image_path),
        "width_pixels": metadata["width"],
        "height_pixels": metadata["height"],
        "bands": metadata["count"],
        "crs": metadata["crs"],
        "georeferenced": has_crs,
        "pixel_bounds": metadata["bounds"],
    }
    if has_crs:
        result["resolution"] = metadata["resolution"]
        result["physical_measurements"] = (
            "Physical spatial measurements may be possible subject to CRS units "
            "and raster validity."
        )
    else:
        result["resolution"] = None
        result["physical_measurements"] = "Unavailable: image has no CRS/georeferencing."
    return result


def get_pixel_space_info(image_path: str | Path) -> dict[str, Any]:
    metadata = get_raster_metadata(image_path)
    if "error" in metadata:
        try:
            from PIL import Image

            with Image.open(str(image_path)) as img:
                width, height = img.size
        except Exception as exc:
            return {"error": metadata["error"], "pil_error": str(exc)}
    else:
        width, height = metadata["width"], metadata["height"]

    return {
        "width_pixels": width,
        "height_pixels": height,
        "pixel_count": width * height,
        "pixel_coordinate_bounds": {
            "x_min": 0,
            "y_min": 0,
            "x_max": width,
            "y_max": height,
        },
    }
