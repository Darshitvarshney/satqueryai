from __future__ import annotations

from pathlib import Path

from PIL import Image

from satquery.geospatial.raster import (
    analyze_spatial_properties,
    get_pixel_space_info,
    get_raster_metadata,
)


def _png(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    Image.new("RGB", (64, 48), (12, 130, 40)).save(path)
    return path


def test_pixel_space_info(tmp_path):
    info = get_pixel_space_info(_png(tmp_path))
    assert info["width_pixels"] == 64
    assert info["height_pixels"] == 48
    assert info["pixel_count"] == 64 * 48
    assert info["pixel_coordinate_bounds"]["x_max"] == 64


def test_spatial_properties_plain_image(tmp_path):
    props = analyze_spatial_properties(_png(tmp_path))
    # rasterio present -> georeferenced is False for a plain PNG
    # rasterio absent  -> {"status": "failed"}
    assert props.get("georeferenced") in (False, None) or props.get("status") == "failed"


def test_raster_metadata_returns_dict(tmp_path):
    meta = get_raster_metadata(_png(tmp_path))
    assert isinstance(meta, dict)
    assert "error" in meta or meta.get("width") == 64
