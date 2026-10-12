"""Optional Sentinel-2 imagery lookup through Google Earth Engine.

This module is deliberately separate from the Bio-ORACLE/Random Forest path:
remote sensing is an additional observation layer and does not change model inputs.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    import ee  # Optional: only required when a user requests satellite imagery.
except ImportError:  # Keep the core Bio-ORACLE and ML APIs available without Earth Engine.
    ee = None

COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_LOOKBACK_DAYS = 180
MAX_SCENE_CLOUD_PERCENT = 80
THUMBNAIL_RADIUS_METERS = 3000
SENTINEL3_COLLECTION_ID = "COPERNICUS/S3/OLCI"
# Monthly, atmospherically processed remote-sensing reflectance product.
SENTINEL3_REFLECTANCE_COLLECTION_ID = "COPERNICUS/MARINE/OC_GLO_BGC/REFLECTANCE_OLCI_300M"
SENTINEL3_THUMBNAIL_RADIUS_METERS = 15000

_ee_lock = threading.Lock()
_initialized = False


class RemoteSensingError(RuntimeError):
    """Raised when Sentinel-2 imagery cannot be retrieved or rendered."""


def _initialize_earth_engine() -> None:
    global _initialized
    if ee is None:
        raise RemoteSensingError(
            "The optional earthengine-api package is not installed. Run `py -m pip install -r requirements.txt` "
            "from the backend folder to enable Sentinel imagery. Core GeoSea prediction and image identification can still run."
        )
    if _initialized:
        return
    with _ee_lock:
        if _initialized:
            return
        project = os.getenv("EARTHENGINE_PROJECT", "").strip()
        if not project:
            raise RemoteSensingError(
                "Google Earth Engine is not configured. Set EARTHENGINE_PROJECT and "
                "authenticate the backend account. See REMOTE_SENSING_SETUP.md."
            )
        try:
            # Local development uses credentials created by `earthengine authenticate`.
            ee.Initialize(project=project)
        except Exception as exc:
            raise RemoteSensingError(
                "Google Earth Engine initialization failed. Check EARTHENGINE_PROJECT, "
                "Earth Engine access, and the backend account's authentication."
            ) from exc
        _initialized = True


def _sentinel2_joined_collection(point: Any, start_date: str, end_date: str) -> Any:
    """Get Sentinel-2 surface-reflectance scenes without requiring a cloud-probability join.

    Requiring a second collection to match by system:index can discard otherwise valid
    scenes. The Scene Classification Layer (SCL) is used for the local clear-pixel mask.
    """
    return (
        ee.ImageCollection(COLLECTION_ID)
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD_PERCENT))
        .sort("system:time_start", False)
    )


def _sentinel2_clear_mask(image: Any) -> Any:
    """Mask cloud, shadow, cirrus, snow and invalid pixels using Sentinel-2 SCL."""
    image = ee.Image(image)
    scl = image.select("SCL")
    return (
        scl.mask().gt(0)
        .And(scl.neq(0))
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(7))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

def get_sentinel2_observation(latitude: float, longitude: float, lookback_days: int = 365) -> dict[str, Any]:
    """Return the locally clearest recent Sentinel-2 RGB thumbnail.

    Candidates are ranked using a combined SCL + cloud-probability mask in the
    local thumbnail area. Scene-wide cloud metadata is shown separately and is
    not treated as the cloud percentage of the thumbnail.
    """
    _initialize_earth_engine()
    now = datetime.now(timezone.utc)
    end_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        point = ee.Geometry.Point([longitude, latitude])
        region = point.buffer(THUMBNAIL_RADIUS_METERS).bounds()
        collection = _sentinel2_joined_collection(point, start_date, end_date)
        scene_count = int(collection.size().getInfo())
        if scene_count == 0:
            raise RemoteSensingError(
                f"No Sentinel-2 scenes cover this location within the last {lookback_days} days. "
                "Try a coastal location or a different date window. Sentinel-2 does not cover every offshore point."
            )

        candidates = collection.limit(100)

        def add_local_clear_score(image: Any) -> Any:
            image = ee.Image(image)
            scl = image.select("SCL")
            observed = scl.mask().gt(0).And(scl.neq(0)).And(scl.neq(1))
            clear = _sentinel2_clear_mask(image)
            counts = ee.Image.cat([
                clear.rename("clear").unmask(0),
                observed.rename("observed").unmask(0),
            ]).reduceRegion(
                reducer=ee.Reducer.sum(), geometry=region, scale=60,
                maxPixels=100000, bestEffort=True,
            )
            observed_count = ee.Number(counts.get("observed", 0))
            clear_count = ee.Number(counts.get("clear", 0))
            clear_percent = ee.Number(ee.Algorithms.If(
                observed_count.gt(0), clear_count.divide(observed_count).multiply(100), 0
            ))
            return image.set("LOCAL_CLEAR_PERCENT", clear_percent)

        ranked = candidates.map(add_local_clear_score).sort("LOCAL_CLEAR_PERCENT", False)
        image = ee.Image(ranked.first())
        properties = image.toDictionary([
            "system:time_start", "CLOUDY_PIXEL_PERCENTAGE", "PRODUCT_ID", "LOCAL_CLEAR_PERCENT"
        ]).getInfo()
        timestamp_ms = properties.get("system:time_start")
        if timestamp_ms is None:
            raise RemoteSensingError("The selected Sentinel-2 scene did not include an observation date.")
        local_clear_percent = round(float(properties.get("LOCAL_CLEAR_PERCENT", 0)), 1)
        if local_clear_percent <= 0:
            raise RemoteSensingError(
                f"Sentinel-2 scenes cover this point, but no usable clear pixels were found in the local area "
                f"within the last {lookback_days} days. Try another nearby location or date window."
            )

        observation_date = datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=timezone.utc).date().isoformat()
        clear_mask = _sentinel2_clear_mask(image)
        rgb = image.select(["B4", "B3", "B2"]).updateMask(clear_mask)
        visualized = rgb.visualize(bands=["B4", "B3", "B2"], min=0, max=2500, gamma=1.2)
        thumbnail_url = visualized.getThumbURL({
            "region": region, "dimensions": "800x520", "format": "png",
        })
        cloud_value = properties.get("CLOUDY_PIXEL_PERCENTAGE")

        # Collect local Sentinel-2 surface-reflectance variables from the same
        # selected, cloud-masked scene. These are observational satellite
        # variables and are deliberately separate from GeoSea's Bio-ORACLE/RF inputs.
        local_reflectance = image.select(["B2", "B3", "B4", "B8", "B11"]).multiply(0.0001).updateMask(clear_mask)
        s2_ndvi = local_reflectance.select("B8").subtract(local_reflectance.select("B4")).divide(
            local_reflectance.select("B8").add(local_reflectance.select("B4")).max(1e-6)
        ).rename("NDVI")
        s2_ndwi = local_reflectance.select("B3").subtract(local_reflectance.select("B8")).divide(
            local_reflectance.select("B3").add(local_reflectance.select("B8")).max(1e-6)
        ).rename("NDWI")
        local_sample = ee.Image.cat([local_reflectance, s2_ndvi, s2_ndwi]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=point.buffer(20), scale=20,
            maxPixels=10000, bestEffort=True,
        ).getInfo()
        s2_variable_specs = [
            ("B2", "Blue surface reflectance · 490 nm", "reflectance", "Sentinel-2 blue-band surface reflectance; unitless."),
            ("B3", "Green surface reflectance · 560 nm", "reflectance", "Sentinel-2 green-band surface reflectance; unitless."),
            ("B4", "Red surface reflectance · 665 nm", "reflectance", "Sentinel-2 red-band surface reflectance; unitless."),
            ("B8", "Near-infrared surface reflectance · 842 nm", "reflectance", "Sentinel-2 NIR surface reflectance; unitless."),
            ("B11", "SWIR surface reflectance · 1610 nm", "reflectance", "Sentinel-2 SWIR surface reflectance; unitless."),
            ("NDVI", "Normalized Difference Vegetation Index", "index", "NIR/red vegetation-response index; not seaweed-specific."),
            ("NDWI", "Normalized Difference Water Index", "index", "Green/NIR water-contrast index used as contextual information."),
        ]
        variables = [
            {"key": key, "label": label, "value": round(float(local_sample[key]), 7) if local_sample.get(key) is not None else None,
             "unit": unit, "description": description}
            for key, label, unit, description in s2_variable_specs
        ]
        return {
            "provider": "European Union / ESA Copernicus",
            "platform": "Sentinel-2",
            "collection": COLLECTION_ID,
            "observation_date": observation_date,
            "scene_cloud_percentage": round(float(cloud_value), 1) if cloud_value is not None else None,
            "local_clear_percentage": local_clear_percent,
            "local_data_percentage": None,
            "scenes_considered": scene_count,
            "scenes_quality_checked": min(scene_count, 100),
            "search_start_date": start_date,
            "search_end_date": end_date,
            "thumbnail_url": thumbnail_url,
            "thumbnail_radius_meters": THUMBNAIL_RADIUS_METERS,
            "visualization": "Cloud-masked natural-colour RGB (B4/B3/B2; Sentinel-2 SCL mask)",
            "variables": variables,
            "interpretation_note": (
                "Pixels classified as cloud, shadow, cirrus, snow or invalid by the Sentinel-2 Scene Classification Layer are masked. "
                "Scene cloud percentage describes the full satellite scene; clear-pixel percentage describes the local thumbnail area. "
                "Some haze, glint, or cloud-mask errors can remain. The spectral variables are local surface-reflectance measurements from the same scene; NDVI/NDWI are contextual indices, not seaweed detections. This is visual evidence only and does not confirm seaweed presence."
            ),
        }
    except RemoteSensingError:
        raise
    except Exception as exc:
        raise RemoteSensingError(f"Sentinel-2 imagery request failed: {exc}") from exc


def get_sentinel3_observation(latitude: float, longitude: float, lookback_days: int = 365) -> dict[str, Any]:
    """Return a Sentinel-3 OLCI natural-colour fallback image.

    This endpoint returns imagery only. It does not calculate extra NOAA-derived
    indicators and does not alter the Bio-ORACLE variables used by the Random Forest.
    """
    _initialize_earth_engine()
    now = datetime.now(timezone.utc)
    end_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    try:
        point = ee.Geometry.Point([longitude, latitude])
        region = point.buffer(SENTINEL3_THUMBNAIL_RADIUS_METERS).bounds()
        collection = (
            ee.ImageCollection(SENTINEL3_COLLECTION_ID)
            .filterBounds(point).filterDate(start_date, end_date)
            .sort("system:time_start", False)
        )
        scene_count = int(collection.size().getInfo())
        if scene_count == 0:
            raise RemoteSensingError(
                f"No Sentinel-3 OLCI scenes cover this location within the last {lookback_days} days."
            )
        recent_start = (now - timedelta(days=min(45, lookback_days))).strftime("%Y-%m-%d")
        recent = collection.filterDate(recent_start, end_date)
        recent_count = int(recent.size().getInfo())
        selected = recent if recent_count > 0 else collection
        image = ee.Image(selected.first())
        timestamp_ms = image.get("system:time_start").getInfo()
        if timestamp_ms is None:
            raise RemoteSensingError("Sentinel-3 scene found without observation date metadata.")

        # Natural-colour rendering only; this is top-of-atmosphere radiance, not
        # a water-quality retrieval. Keep this as a fallback visual, not analysis.
        rgb_names = ["Oa08_radiance", "Oa06_radiance", "Oa04_radiance"]
        rgb_scales = [0.00876539, 0.0123538, 0.0115198]
        rgb = image.select(rgb_names).multiply(ee.Image.constant(rgb_scales))
        rgb = rgb.updateMask(rgb.reduce(ee.Reducer.min()).gt(0))
        visualized = rgb.visualize(bands=rgb_names, min=0, max=6, gamma=1.5)
        thumbnail_url = visualized.getThumbURL({
            "region": region, "dimensions": "800x520", "format": "png"
        })
        observation_date = datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=timezone.utc).date().isoformat()
        return {
            "provider": "ESA / Copernicus",
            "platform": "Sentinel-3 OLCI",
            "collection": SENTINEL3_COLLECTION_ID,
            "observation_date": observation_date,
            "scene_cloud_percentage": None,
            "local_clear_percentage": 0,
            "local_data_percentage": None,
            "scenes_considered": scene_count,
            "scenes_quality_checked": min(scene_count, 40),
            "search_start_date": start_date,
            "search_end_date": end_date,
            "thumbnail_url": thumbnail_url,
            "thumbnail_radius_meters": SENTINEL3_THUMBNAIL_RADIUS_METERS,
            "visualization": "Sentinel-3 OLCI natural-colour radiance image (~300 m pixels; not atmospherically corrected)",
            "variables": [],
            "interpretation_note": (
                "Sentinel-3 is shown only when usable Sentinel-2 imagery is unavailable. This RGB is a visual context layer, "
                "not a validated water-quality or seaweed product. Cloud, haze and sun-glint effects can remain. "
                "The satellite image is a visual context layer only; it is not used as an input to the Random Forest."
            ),
        }
    except RemoteSensingError:
        raise
    except Exception as exc:
        raise RemoteSensingError(f"Sentinel-3 imagery request failed: {exc}") from exc
