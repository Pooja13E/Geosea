/**
 * Data access layer — now backed by the real GeoSea_8_Variables.csv dataset.
 *
 * WHY COLUMNAR STORAGE
 * The dataset has 336,924 rows. Holding it as an array of plain JS objects
 * (~13 fields each) costs far more memory and GC pressure than a
 * structure-of-arrays layout using typed arrays. So the parsed dataset is
 * kept internally as a set of Float32Array/Uint8Array/Int32Array columns,
 * and a single row is only "hydrated" into an OccurrenceRecord (see
 * src/types/environmental.ts) when a component actually needs to display
 * one record (e.g. the location info panel).
 *
 * No component should touch these columns directly — everything goes
 * through the functions exported below, which is also the seam later
 * stages (ML predictions, environmental layers) will read through.
 *
 * but is not imported anywhere in this file or by any component.
 */
import Papa from "papaparse";
import { canonicalBinomialName, normalizeTaxonLabel } from "@/utils/scientificNames";
import {
  DEFAULT_FILTERS,
  ENV_VARIABLE_KEYS,
  PSEUDO_ABSENCE_LABEL,
  type DatasetSummary,
  type EnvGridCell,
  type EnvironmentalStats,
  type EnvironmentalVariables,
  type FilterState,
  type GridCell,
  type OccurrenceRecord,
  type SpeciesProfile,
  type SpatialFold,
} from "@/types/environmental";

/**
 * ROOT CAUSE OF THE LOADING BUG
 * Papa.parse() below is called with `download: true` AND `worker: true`.
 * When `worker: true`, PapaParse offloads the HTTP request itself to a Web
 * Worker built from a Blob. Inside that worker, `self.location` is the
 * blob: URL the worker script was created from — not this page's origin —
 * so a root-relative URL like "/data/GeoSea_8_Variables.csv" cannot be
 * resolved against it. The worker's XMLHttpRequest.open() then throws
 * "Failed to execute 'open' on 'XMLHttpRequest': Invalid URL" synchronously,
 * the parse never starts, and the app is stuck on the loading screen
 * forever (the promise never resolves or rejects).
 *
 * FIX: resolve the public asset path to a fully-qualified absolute URL
 * (scheme + host + path) up front, using the page's own origin. An absolute
 * URL resolves the same way regardless of the worker's blob: base, in dev
 * and in a production build/preview, and regardless of Vite's configured
 * `base` (we still route through import.meta.env.BASE_URL so this keeps
 * working if base is ever changed from the default "/").
 */
const CSV_PATH = `${import.meta.env.BASE_URL}data/GeoSea_8_Variables.csv`.replace(/\/{2,}/g, "/");
const CSV_URL = new URL(CSV_PATH, window.location.origin).toString();

/** Row-count headroom above the documented 336,924 so a slightly larger file still loads. */
const INITIAL_CAPACITY = 350_000;

interface Columns {
  count: number;
  speciesIndex: Int32Array; // index into `speciesTable`
  latitude: Float32Array;
  longitude: Float32Array;
  presence: Uint8Array; // 0 | 1
  temperature: Float32Array;
  salinity: Float32Array;
  chlorophyll: Float32Array;
  nitrate: Float32Array;
  phosphate: Float32Array;
  pH: Float32Array;
  PAR: Float32Array;
  kdpar: Float32Array;
  spatialFold: Uint8Array;
}

let columns: Columns | null = null;
let speciesTable: string[] = [];
let speciesToIndex = new Map<string, number>();
let pseudoAbsenceIndex = -1;

let datasetSummaryCache: DatasetSummary | null = null;
let speciesListCache: string[] | null = null;

let loadPromise: Promise<DatasetSummary> | null = null;

function allocateColumns(capacity: number): Columns {
  return {
    count: 0,
    speciesIndex: new Int32Array(capacity),
    latitude: new Float32Array(capacity),
    longitude: new Float32Array(capacity),
    presence: new Uint8Array(capacity),
    temperature: new Float32Array(capacity),
    salinity: new Float32Array(capacity),
    chlorophyll: new Float32Array(capacity),
    nitrate: new Float32Array(capacity),
    phosphate: new Float32Array(capacity),
    pH: new Float32Array(capacity),
    PAR: new Float32Array(capacity),
    kdpar: new Float32Array(capacity),
    spatialFold: new Uint8Array(capacity),
  };
}

function internSpecies(name: string): number {
  const existing = speciesToIndex.get(name);
  if (existing !== undefined) return existing;
  const idx = speciesTable.length;
  speciesTable.push(name);
  speciesToIndex.set(name, idx);
  return idx;
}

const REQUIRED_COLUMNS = [
  "scientific_name",
  "latitude",
  "longitude",
  "presence",
  "temperature",
  "salinity",
  "chlorophyll",
  "nitrate",
  "phosphate",
  "pH",
  "PAR",
  "kdpar",
  "spatial_fold",
];

/**
 * Loads and parses the real dataset exactly once (subsequent calls reuse
 * the same in-flight/resolved promise). Resolves with the dataset summary,
 * or rejects with a readable error message on malformed input.
 */
export function loadEnvironmentalData(): Promise<DatasetSummary> {
  if (loadPromise) return loadPromise;

  loadPromise = new Promise<DatasetSummary>((resolve, reject) => {
    let cols = allocateColumns(INITIAL_CAPACITY);
    let capacity = INITIAL_CAPACITY;
    let validated = false;

    function growIfNeeded() {
      if (cols.count < capacity) return;
      const nextCapacity = Math.ceil(capacity * 1.5);
      const grown = allocateColumns(nextCapacity);
      grown.count = cols.count;
      (Object.keys(cols) as Array<keyof Columns>).forEach((key) => {
        if (key === "count") return;
        (grown[key] as Float32Array | Uint8Array | Int32Array).set(
          cols[key] as Float32Array | Uint8Array | Int32Array
        );
      });
      cols = grown;
      capacity = nextCapacity;
    }

    Papa.parse<Record<string, string>>(CSV_URL, {
      download: true,
      header: true,
      skipEmptyLines: true,
      worker: true,
      step: (result) => {
        const row = result.data;

        if (!validated) {
          const missing = REQUIRED_COLUMNS.filter((col) => !(col in row));
          if (missing.length > 0) {
            reject(
              new Error(
                `GeoSea_8_Variables.csv is missing expected column(s): ${missing.join(", ")}`
              )
            );
            return;
          }
          validated = true;
        }

        growIfNeeded();
        const i = cols.count;

        const lat = Number.parseFloat(row.latitude);
        const lon = Number.parseFloat(row.longitude);
        const presenceValue = Number.parseFloat(row.presence);
        const presence = presenceValue === 1 ? 1 : 0;
        const fold = Number.parseInt(row.spatial_fold, 10);

        if (Number.isNaN(lat) || Number.isNaN(lon)) {
          // Skip malformed geographic rows rather than failing the whole load.
          return;
        }

        const sourceName = (row.scientific_name ?? PSEUDO_ABSENCE_LABEL).trim();
        // Normalize author/rank suffix variants for search, filtering, map labels,
        // and profiles. Genus-only labels stay genus-only. The source CSV remains
        // unchanged; this is display normalization, not synonym resolution.
        cols.speciesIndex[i] = internSpecies(
          sourceName.toLowerCase() === PSEUDO_ABSENCE_LABEL.toLowerCase()
            ? PSEUDO_ABSENCE_LABEL
            : normalizeTaxonLabel(sourceName)
        );
        cols.latitude[i] = lat;
        cols.longitude[i] = lon;
        cols.presence[i] = presence;
        cols.temperature[i] = Number.parseFloat(row.temperature);
        cols.salinity[i] = Number.parseFloat(row.salinity);
        cols.chlorophyll[i] = Number.parseFloat(row.chlorophyll);
        cols.nitrate[i] = Number.parseFloat(row.nitrate);
        cols.phosphate[i] = Number.parseFloat(row.phosphate);
        cols.pH[i] = Number.parseFloat(row.pH);
        cols.PAR[i] = Number.parseFloat(row.PAR);
        cols.kdpar[i] = Number.parseFloat(row.kdpar);
        cols.spatialFold[i] = Number.isNaN(fold) ? 0 : (fold as SpatialFold);
        cols.count += 1;
      },
      complete: () => {
        if (cols.count === 0) {
          reject(new Error("GeoSea_8_Variables.csv loaded but contained no usable rows."));
          return;
        }
        pseudoAbsenceIndex = speciesToIndex.get(PSEUDO_ABSENCE_LABEL) ?? -1;
        columns = cols;
        datasetSummaryCache = computeSummary(allIndices());
        speciesListCache = computeSpeciesList();
        resolve(datasetSummaryCache);
      },
      error: (err) => {
        reject(new Error(`Failed to load GeoSea_8_Variables.csv: ${err.message}`));
      },
    });
  }).catch((err) => {
    // Allow a retry after a failure instead of caching a rejected promise forever.
    loadPromise = null;
    throw err;
  });

  return loadPromise;
}

function requireColumns(): Columns {
  if (!columns) {
    throw new Error("Dataset not loaded yet — call loadEnvironmentalData() first.");
  }
  return columns;
}

/** All row indices in the loaded dataset, unfiltered. */
export function getAllIndices(): Int32Array {
  return allIndices();
}

function allIndices(): Int32Array {
  const cols = requireColumns();
  const idx = new Int32Array(cols.count);
  for (let i = 0; i < cols.count; i++) idx[i] = i;
  return idx;
}

function isPseudoAbsenceRow(cols: Columns, i: number): boolean {
  return cols.presence[i] === 0 || cols.speciesIndex[i] === pseudoAbsenceIndex;
}

function computeSpeciesList(): string[] {
  const cols = requireColumns();
  const seen = new Set<number>();
  for (let i = 0; i < cols.count; i++) {
    if (!isPseudoAbsenceRow(cols, i)) seen.add(cols.speciesIndex[i]);
  }
  return Array.from(seen)
    .map((idx) => speciesTable[idx])
    // Species Explorer lists canonical binomials only; genus/higher-rank records
    // remain in the underlying map and overall record counts but are not mislabeled
    // as species in the search control.
    .filter((name) => canonicalBinomialName(name) === name)
    .sort((a, b) => a.localeCompare(b));
}

/** All real species names (excludes Pseudo_Absence), sorted alphabetically. Cached after load. */
export function getSpeciesList(): string[] {
  if (!speciesListCache) speciesListCache = computeSpeciesList();
  return speciesListCache;
}

/**
 * Case-insensitive substring search over the species list. Kept cheap and
 * capped so the sidebar never has to render thousands of options at once.
 */
export function searchSpecies(query: string, maxResults = 200): string[] {
  const list = getSpeciesList();
  const q = query.trim().toLowerCase();
  if (!q) return list.slice(0, maxResults);
  const results: string[] = [];
  for (const name of list) {
    if (name.toLowerCase().includes(q)) {
      results.push(name);
      if (results.length >= maxResults) break;
    }
  }
  return results;
}


function computeSummary(indices: ArrayLike<number>): DatasetSummary {
  const cols = requireColumns();

  let presenceCount = 0;
  let absenceCount = 0;
  let latMin = Infinity;
  let latMax = -Infinity;
  let lonMin = Infinity;
  let lonMax = -Infinity;
  const speciesSeen = new Set<string>();
  const foldCounts: Record<SpatialFold, number> = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };

  const ranges: Record<string, [number, number]> = {};
  for (const key of ENV_VARIABLE_KEYS) ranges[key] = [Infinity, -Infinity];

  for (let k = 0; k < indices.length; k++) {
    const i = indices[k];
    const pseudo = isPseudoAbsenceRow(cols, i);
    if (pseudo) absenceCount += 1;
    else {
      presenceCount += 1;
      const rawSpecies = speciesTable[cols.speciesIndex[i]];
      const canonical = canonicalBinomialName(rawSpecies);
      if (canonical) speciesSeen.add(canonical);
    }

    if (cols.latitude[i] < latMin) latMin = cols.latitude[i];
    if (cols.latitude[i] > latMax) latMax = cols.latitude[i];
    if (cols.longitude[i] < lonMin) lonMin = cols.longitude[i];
    if (cols.longitude[i] > lonMax) lonMax = cols.longitude[i];

    const fold = cols.spatialFold[i] as SpatialFold;
    foldCounts[fold] = (foldCounts[fold] ?? 0) + 1;

    for (const key of ENV_VARIABLE_KEYS) {
      const v = cols[key as keyof Columns] as Float32Array;
      const value = v[i];
      const range = ranges[key];
      if (value < range[0]) range[0] = value;
      if (value > range[1]) range[1] = value;
    }
  }

  return {
    recordCount: indices.length,
    presenceCount,
    absenceCount,
    speciesCount: speciesSeen.size,
    latitudeRange: [latMin, latMax],
    longitudeRange: [lonMin, lonMax],
    environmentalRanges: ranges as Record<keyof EnvironmentalVariables, [number, number]>,
    foldCounts,
  };
}

/** Summary for the whole loaded dataset. Computed once at load time and cached. */
export function getDatasetSummary(): DatasetSummary {
  if (!datasetSummaryCache) datasetSummaryCache = computeSummary(allIndices());
  return datasetSummaryCache;
}

/** Summary recomputed for an arbitrary subset of row indices (e.g. the current filter). */
export function getSummaryForIndices(indices: ArrayLike<number>): DatasetSummary {
  return computeSummary(indices);
}

/**
 * Returns row indices matching the given filters. A single pass over typed
 * arrays — fast enough (a few ms for 336k rows) to call directly on every
 * filter change rather than maintaining per-filter caches.
 */
export function getFilteredRecords(filters: FilterState): Int32Array {
  const cols = requireColumns();
  const speciesIdx = filters.species === "all" ? -1 : speciesToIndex.get(filters.species) ?? -2;
  const envRange = filters.envRange;
  const envColumn = envRange ? (cols[envRange.variable as keyof Columns] as Float32Array) : null;

  const out = new Int32Array(cols.count);
  let n = 0;

  for (let i = 0; i < cols.count; i++) {
    if (filters.presence === "presence" && cols.presence[i] !== 1) continue;
    if (filters.presence === "absence" && cols.presence[i] !== 0) continue;
    if (filters.fold !== "all" && cols.spatialFold[i] !== filters.fold) continue;
    if (speciesIdx !== -1 && cols.speciesIndex[i] !== speciesIdx) continue;
    if (envColumn && envRange) {
      const v = envColumn[i];
      if (v < envRange.min || v > envRange.max) continue;
    }
    out[n++] = i;
  }

  return out.subarray(0, n);
}

/** Further restricts a set of row indices to a lat/lng bounding box. */
export function getIndicesInBounds(
  indices: ArrayLike<number>,
  bounds: { south: number; north: number; west: number; east: number }
): Int32Array {
  const cols = requireColumns();
  const out = new Int32Array(indices.length);
  let n = 0;
  for (let k = 0; k < indices.length; k++) {
    const i = indices[k];
    const lat = cols.latitude[i];
    const lon = cols.longitude[i];
    if (lat >= bounds.south && lat <= bounds.north && lon >= bounds.west && lon <= bounds.east) {
      out[n++] = i;
    }
  }
  return out.subarray(0, n);
}

/** Hydrates a single row into a display-ready OccurrenceRecord. */
export function getRecordAt(index: number): OccurrenceRecord {
  const cols = requireColumns();
  const speciesIdx = cols.speciesIndex[index];
  return {
    index,
    scientific_name: speciesTable[speciesIdx],
    isPseudoAbsence: isPseudoAbsenceRow(cols, index),
    latitude: cols.latitude[index],
    longitude: cols.longitude[index],
    presence: cols.presence[index] as 0 | 1,
    temperature: cols.temperature[index],
    salinity: cols.salinity[index],
    chlorophyll: cols.chlorophyll[index],
    nitrate: cols.nitrate[index],
    phosphate: cols.phosphate[index],
    pH: cols.pH[index],
    PAR: cols.PAR[index],
    kdpar: cols.kdpar[index],
    spatial_fold: cols.spatialFold[index] as SpatialFold,
  };
}

/** All hydrated presence records for one species (used for species-detail views). */
export function getRecordsForSpecies(species: string): OccurrenceRecord[] {
  const indices = getFilteredRecords({ ...DEFAULT_FILTERS, species, presence: "presence" });
  const records: OccurrenceRecord[] = [];
  for (let k = 0; k < indices.length; k++) records.push(getRecordAt(indices[k]));
  return records;
}

/** Compact profile for the selected real species, computed without hydrating every row. */
export function getSpeciesProfile(species: string): SpeciesProfile | null {
  const cols = requireColumns();
  const speciesIdx = speciesToIndex.get(species);
  if (speciesIdx === undefined || species === PSEUDO_ABSENCE_LABEL) return null;

  let count = 0;
  let latMin = Infinity;
  let latMax = -Infinity;
  let lonMin = Infinity;
  let lonMax = -Infinity;
  const ranges: Record<string, [number, number]> = {};
  for (const key of ENV_VARIABLE_KEYS) ranges[key] = [Infinity, -Infinity];

  for (let i = 0; i < cols.count; i++) {
    if (cols.speciesIndex[i] !== speciesIdx || isPseudoAbsenceRow(cols, i)) continue;
    count += 1;
    const lat = cols.latitude[i];
    const lon = cols.longitude[i];
    latMin = Math.min(latMin, lat);
    latMax = Math.max(latMax, lat);
    lonMin = Math.min(lonMin, lon);
    lonMax = Math.max(lonMax, lon);
    for (const key of ENV_VARIABLE_KEYS) {
      const value = (cols[key as keyof Columns] as Float32Array)[i];
      const range = ranges[key];
      if (value < range[0]) range[0] = value;
      if (value > range[1]) range[1] = value;
    }
  }

  if (count === 0) return null;
  return {
    scientificName: species,
    occurrenceCount: count,
    latitudeRange: [latMin, latMax],
    longitudeRange: [lonMin, lonMax],
    environmentalRanges: ranges as Record<keyof EnvironmentalVariables, [number, number]>,
  };
}

/** Min/max per environmental variable across the whole dataset. */
export function getEnvironmentalRanges(): Record<keyof EnvironmentalVariables, [number, number]> {
  return getDatasetSummary().environmentalRanges;
}

/**
 * Spatially samples a set of row indices into a small number of
 * REPRESENTATIVE POINTS so the map can render a manageable number of
 * markers instead of one per record. `cellSizeDeg` (which shrinks as the
 * user zooms in) only controls how many bins — and therefore how many
 * points — are produced; it never controls a rendered shape's size. Each
 * bin keeps the real coordinate of one actual observation from within it
 * (biased toward the bin's majority class) so the map still plots genuine
 * observation locations, never an artificial cell center or rectangle.
 */
export function getGridAggregation(indices: ArrayLike<number>, cellSizeDeg: number): GridCell[] {
  const cols = requireColumns();

  interface Accumulator {
    id: string;
    latCenter: number;
    lngCenter: number;
    bounds: { south: number; north: number; west: number; east: number };
    count: number;
    presenceCount: number;
    absenceCount: number;
    // First actual observation seen of each class, used to source the
    // representative marker's real coordinate once the majority class for
    // the bin is known.
    firstPresenceLat: number | null;
    firstPresenceLng: number | null;
    firstPresenceIndex: number | null;
    firstAbsenceLat: number | null;
    firstAbsenceLng: number | null;
    firstAbsenceIndex: number | null;
  }
  const cells = new Map<string, Accumulator>();

  for (let k = 0; k < indices.length; k++) {
    const i = indices[k];
    const lat = cols.latitude[i];
    const lon = cols.longitude[i];
    const cellLat = Math.floor(lat / cellSizeDeg);
    const cellLon = Math.floor(lon / cellSizeDeg);
    const id = `${cellLat}:${cellLon}`;

    let cell = cells.get(id);
    if (!cell) {
      const south = cellLat * cellSizeDeg;
      const west = cellLon * cellSizeDeg;
      cell = {
        id,
        latCenter: south + cellSizeDeg / 2,
        lngCenter: west + cellSizeDeg / 2,
        bounds: { south, north: south + cellSizeDeg, west, east: west + cellSizeDeg },
        count: 0,
        presenceCount: 0,
        absenceCount: 0,
        firstPresenceLat: null,
        firstPresenceLng: null,
        firstPresenceIndex: null,
        firstAbsenceLat: null,
        firstAbsenceLng: null,
        firstAbsenceIndex: null,
      };
      cells.set(id, cell);
    }

    cell.count += 1;
    if (isPseudoAbsenceRow(cols, i)) {
      cell.absenceCount += 1;
      if (cell.firstAbsenceLat === null) {
        cell.firstAbsenceLat = lat;
        cell.firstAbsenceLng = lon;
        cell.firstAbsenceIndex = i;
      }
    } else {
      cell.presenceCount += 1;
      if (cell.firstPresenceLat === null) {
        cell.firstPresenceLat = lat;
        cell.firstPresenceLng = lon;
        cell.firstPresenceIndex = i;
      }
    }
  }

  const result: GridCell[] = [];
  cells.forEach((acc) => {
    // Majority class decides which real observation's coordinate the
    // representative point is drawn at (and its presence/absence color),
    // falling back to whichever class is actually present in the bin.
    const presenceIsMajority = acc.presenceCount >= acc.absenceCount;
    const useAbsence =
      (!presenceIsMajority && acc.firstAbsenceLat !== null) || acc.firstPresenceLat === null;
    const representativeLat = useAbsence ? acc.firstAbsenceLat! : acc.firstPresenceLat!;
    const representativeLng = useAbsence ? acc.firstAbsenceLng! : acc.firstPresenceLng!;
    const representativeIndex = useAbsence ? acc.firstAbsenceIndex! : acc.firstPresenceIndex!;

    result.push({
      id: acc.id,
      latCenter: acc.latCenter,
      lngCenter: acc.lngCenter,
      bounds: acc.bounds,
      count: acc.count,
      presenceCount: acc.presenceCount,
      absenceCount: acc.absenceCount,
      representativeLat,
      representativeLng,
      representativeIndex,
      secondaryRepresentativeLat: useAbsence ? acc.firstPresenceLat : acc.firstAbsenceLat,
      secondaryRepresentativeLng: useAbsence ? acc.firstPresenceLng : acc.firstAbsenceLng,
      secondaryRepresentativeIndex: useAbsence ? acc.firstPresenceIndex : acc.firstAbsenceIndex,
      presence: useAbsence ? 0 : 1,
    });
  });

  return result;
}

/**
 * Environmental counterpart to getGridAggregation(). Bins a set of row
 * indices into the same lat/lng grid, but each cell carries the mean/min/max
 * of the selected environmental variable across the *observations* that
 * fall in it — i.e. spatially aggregated observed values, not an
 * interpolated raster. Reuses the same columnar storage; no per-row objects
 * are allocated.
 */
export function getEnvironmentalGridAggregation(
  indices: ArrayLike<number>,
  cellSizeDeg: number,
  variable: keyof EnvironmentalVariables
): EnvGridCell[] {
  const cols = requireColumns();
  const values = cols[variable as keyof Columns] as Float32Array;

  interface Accumulator {
    latCell: number;
    lonCell: number;
    count: number;
    sum: number;
    min: number;
    max: number;
    representativeIndex: number;
  }
  const cells = new Map<string, Accumulator>();

  for (let k = 0; k < indices.length; k++) {
    const i = indices[k];
    const value = values[i];
    if (!Number.isFinite(value)) continue;

    const lat = cols.latitude[i];
    const lon = cols.longitude[i];
    const cellLat = Math.floor(lat / cellSizeDeg);
    const cellLon = Math.floor(lon / cellSizeDeg);
    const id = `${cellLat}:${cellLon}`;

    let acc = cells.get(id);
    if (!acc) {
      acc = { latCell: cellLat, lonCell: cellLon, count: 0, sum: 0, min: Infinity, max: -Infinity, representativeIndex: i };
      cells.set(id, acc);
    }
    acc.count += 1;
    acc.sum += value;
    if (value < acc.min) acc.min = value;
    if (value > acc.max) acc.max = value;
  }

  const result: EnvGridCell[] = [];
  cells.forEach((acc, id) => {
    const south = acc.latCell * cellSizeDeg;
    const west = acc.lonCell * cellSizeDeg;
    result.push({
      id,
      latCenter: south + cellSizeDeg / 2,
      lngCenter: west + cellSizeDeg / 2,
      bounds: { south, north: south + cellSizeDeg, west, east: west + cellSizeDeg },
      count: acc.count,
      mean: acc.sum / acc.count,
      min: acc.min,
      max: acc.max,
      representativeIndex: acc.representativeIndex,
    });
  });

  return result;
}

/**
 * Descriptive statistics (min/max/mean/median/count) for one environmental
 * variable over a set of row indices. Callers pass either the full dataset
 * (allIndices()) or a filtered subset — the caller is responsible for
 * labeling which one is shown, so this dashboard never presents filtered
 * statistics as if they were global.
 */
export function getEnvironmentalStats(
  indices: ArrayLike<number>,
  variable: keyof EnvironmentalVariables
): EnvironmentalStats {
  const cols = requireColumns();
  const values = cols[variable as keyof Columns] as Float32Array;
  const n = indices.length;

  if (n === 0) {
    return { variable, min: NaN, max: NaN, mean: NaN, median: NaN, count: 0 };
  }

  const sorted = new Float32Array(n);
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let k = 0; k < n; k++) {
    const v = values[indices[k]];
    sorted[k] = v;
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  sorted.sort();

  const mid = Math.floor(n / 2);
  const median = n % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;

  return { variable, min, max, mean: sum / n, median, count: n };
}

/** Convenience wrapper: environmental stats over the entire loaded dataset. */
export function getGlobalEnvironmentalStats(variable: keyof EnvironmentalVariables): EnvironmentalStats {
  return getEnvironmentalStats(allIndices(), variable);
}

/** Current dataset load status, useful for debugging / non-React contexts. */
export function isDatasetLoaded(): boolean {
  return columns !== null;
}
