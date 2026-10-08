"""Bio-ORACLE point lookup for GeoSea inference.

The production Random Forest was trained from these exact Bio-ORACLE v3 baseline
layers. This service therefore retrieves the same baseline layer values at a
user-selected coordinate before inference, rather than borrowing the nearest
GeoSea occurrence record.
"""

from __future__ import annotations

import csv
import io
import math
import threading
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple

import requests

ERDDAP_BASE = "https://erddap.bio-oracle.org/erddap/griddap"
# The original GeoSea extraction script selected the first time slice.
BASELINE_TIME = "2000-01-01T00:00:00Z"
RESOLUTION_DEG = 0.05
CACHE_ROUND_DIGITS = 4
REQUEST_TIMEOUT_SECONDS = 20

VARIABLES = {
    "temperature": ("thetao_baseline_2000_2019_depthsurf", "thetao_mean"),
    "salinity": ("so_baseline_2000_2019_depthsurf", "so_mean"),
    "chlorophyll": ("chl_baseline_2000_2018_depthsurf", "chl_mean"),
    "nitrate": ("no3_baseline_2000_2018_depthsurf", "no3_mean"),
    "phosphate": ("po4_baseline_2000_2018_depthsurf", "po4_mean"),
    "pH": ("ph_baseline_2000_2018_depthsurf", "ph_mean"),
    "PAR": ("par_mean_baseline_2000_2020_depthsurf", "par_mean_mean"),
    "kdpar": ("kdpar_mean_baseline_2000_2020_depthsurf", "kdpar_mean_mean"),
}

SOURCE_METADATA = {
    "provider": "Bio-ORACLE consortium",
    "version": "Bio-ORACLE v3.0",
    "temporal_basis": "baseline climatology",
    "baseline_time_slice": BASELINE_TIME,
    "spatial_resolution": "0.05°",
    "spatial_resolution_km_equator": "~5.5 km",
    "access": "Bio-ORACLE ERDDAP griddap",
}

_session = requests.Session()
_cache: Dict[Tuple[float, float], dict] = {}
_cache_lock = threading.Lock()


class BioOracleError(RuntimeError):
    pass


def _cache_key(lat: float, lon: float) -> Tuple[float, float]:
    return (round(lat, CACHE_ROUND_DIGITS), round(lon, CACHE_ROUND_DIGITS))


def _validate_coordinate(lat: float, lon: float) -> None:
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise BioOracleError("Latitude and longitude must be finite numbers.")
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise BioOracleError("Latitude/longitude are outside valid geographic ranges.")


def _request_csv(dataset_id: str, variable_name: str, lat: float, lon: float, radius: float = 0.0) -> str:
    # Parenthesized coordinate values tell ERDDAP to use the closest grid cell.
    # This mirrors xarray's method="nearest" used by the original extraction.
    if radius <= 0:
        lat_part = f"[({lat:.6f})]"
        lon_part = f"[({lon:.6f})]"
    else:
        lat_part = f"[({max(-90, lat - radius):.6f}):1:({min(90, lat + radius):.6f})]"
        lon_part = f"[({max(-180, lon - radius):.6f}):1:({min(180, lon + radius):.6f})]"

    query = f"{variable_name}[({BASELINE_TIME})]{lat_part}{lon_part}"
    url = f"{ERDDAP_BASE}/{dataset_id}.csv"
    request_url = url + "?" + quote(query, safe="[]():,.-+TZ")
    response = _session.get(request_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def _parse_rows(csv_text: str) -> list[tuple[float, float, float]]:
    rows = list(csv.reader(io.StringIO(csv_text)))
    if len(rows) < 3:
        return []

    # ERDDAP CSV normally contains header, units, then data rows.
    parsed: list[tuple[float, float, float]] = []
    for row in rows[2:]:
        if len(row) < 4:
            continue
        try:
            grid_lat = float(row[1])
            grid_lon = float(row[2])
            value = float(row[3])
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in (grid_lat, grid_lon, value)):
            continue
        if value <= -9990:
            continue
        parsed.append((grid_lat, grid_lon, value))
    return parsed


def _nearest_value(csv_text: str, lat: float, lon: float) -> float | None:
    rows = _parse_rows(csv_text)
    if not rows:
        return None

    cos_lat = math.cos(math.radians(lat))
    return min(
        rows,
        key=lambda item: (item[0] - lat) ** 2 + ((item[1] - lon) * cos_lat) ** 2,
    )[2]


def _fetch_variable(name: str, lat: float, lon: float) -> float:
    dataset_id, variable_name = VARIABLES[name]
    try:
        value = _nearest_value(
            _request_csv(dataset_id, variable_name, lat, lon), lat, lon
        )
        if value is None:
            # Coastal clicks can land on a masked/land grid cell. Retry with a
            # small neighborhood and select the nearest valid ocean cell.
            value = _nearest_value(
                _request_csv(dataset_id, variable_name, lat, lon, radius=0.15),
                lat,
                lon,
            )
    except requests.RequestException as exc:
        raise BioOracleError(f"Bio-ORACLE request failed for {name}: {exc}") from exc
    except Exception as exc:
        raise BioOracleError(f"Could not parse Bio-ORACLE {name}: {exc}") from exc

    if value is None:
        raise BioOracleError(
            f"Bio-ORACLE returned no valid {name} value near the selected coordinate."
        )
    return value

def get_environment(lat: float, lon: float) -> dict:
    """Retrieve the exact eight training variables for a clicked coordinate."""
    _validate_coordinate(lat, lon)
    key = _cache_key(lat, lon)

    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached

    values: dict[str, float] = {}
    errors: list[str] = []

    # Eight independent ERDDAP datasets. Parallel requests keep click latency
    # reasonable while retaining one authoritative source per variable.
    with ThreadPoolExecutor(max_workers=len(VARIABLES)) as executor:
        futures = {
            executor.submit(_fetch_variable, name, lat, lon): name
            for name in VARIABLES
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                values[name] = float(future.result())
            except Exception as exc:
                errors.append(str(exc))

    if errors:
        raise BioOracleError("; ".join(errors))

    ordered = {name: values[name] for name in VARIABLES}
    result = {
        "values": ordered,
        "source": SOURCE_METADATA.copy(),
    }

    with _cache_lock:
        # Keep the cache bounded without adding an external dependency.
        if len(_cache) >= 512:
            _cache.pop(next(iter(_cache)))
        _cache[key] = result

    return result
