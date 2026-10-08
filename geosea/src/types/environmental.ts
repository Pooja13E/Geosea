/**
 * Core data types for GeoSea.
 *
 * These mirror the columns of GeoSea_8_Variables.csv so the dashboard's
 * data layer can be pointed at the real dataset without changing UI
 * components. The 336k-row dataset itself is held in a columnar
 * (structure-of-arrays) form internally for memory/perf reasons — see
 * src/services/dataService.ts — and only "hydrated" into an
 * OccurrenceRecord when a single row needs to be displayed (e.g. the
 * location info panel).
 */

export const SPATIAL_FOLDS = [0, 1, 2, 3, 4] as const;
export type SpatialFold = (typeof SPATIAL_FOLDS)[number];

/** Sentinel scientific_name used by the dataset for background/pseudo-absence rows. */
export const PSEUDO_ABSENCE_LABEL = "Pseudo_Absence";

/** The 8 environmental variables used across the project. */
export interface EnvironmentalVariables {
  temperature: number; // deg C
  salinity: number; // PSU
  chlorophyll: number; // mg/m^3
  nitrate: number; // umol/L
  phosphate: number; // umol/L
  pH: number;
  PAR: number; // Photosynthetically Active Radiation
  kdpar: number; // diffuse attenuation coefficient of PAR
}

/** Keys of EnvironmentalVariables, and the columnar field each maps to. */
export const ENV_VARIABLE_KEYS: Array<keyof EnvironmentalVariables> = [
  "temperature",
  "salinity",
  "chlorophyll",
  "nitrate",
  "phosphate",
  "pH",
  "PAR",
  "kdpar",
];

/** One row of GeoSea_8_Variables.csv, hydrated for display. */
export interface OccurrenceRecord extends EnvironmentalVariables {
  /** Row index into the columnar dataset — stable identity for a record. */
  index: number;
  scientific_name: string;
  /** True when this row is a background/pseudo-absence sample, not a species. */
  isPseudoAbsence: boolean;
  latitude: number;
  longitude: number;
  presence: 0 | 1;
  spatial_fold: SpatialFold;
}

/** A location the user has selected on the map (click or marker). */
export interface MapLocation {
  latitude: number;
  longitude: number;
  label?: string;
}

/**
 * Suitability classes for the 5-band scheme (see src/data/suitabilityConfig.ts),
 * plus "unknown" for when no live prediction is available yet.
 */
export type SuitabilityClass = "very_low" | "low" | "moderate" | "high" | "very_high" | "unknown";

/**
 * Shape of a single model's prediction for one location.
 * `probability` stays null until a real trained model is connected — see
 * src/services/predictionService.ts. The frontend must never invent a value
 * here.
 */
export interface HabitatPrediction {
  modelId: ModelId;
  location: MapLocation;
  probability: number | null; // 0-1 Random Forest positive-class score
  suitabilityClass: SuitabilityClass;
  confidence: number | null; // Reserved for future calibrated uncertainty; currently null
  /** The 8 environmental values used as model input, retrieved for the clicked coordinate. */
  environmentalValues: EnvironmentalVariables | null;
  /** Provenance for the environmental values used by the live prediction. */
  environmentalSource: {
    provider: string;
    version: string;
    temporal_basis: string;
    baseline_time_slice: string;
    spatial_resolution: string;
    spatial_resolution_km_equator: string;
    access: string;
  } | null;
  /** True once a real model has produced `probability`. */
  isLive: boolean;
  /** Human-readable service error retained for the UI when a live lookup fails. */
  errorMessage?: string;
  /** Variable keys whose looked-up value fell outside the training data's observed range. */
  outOfRangeVariables: Array<keyof EnvironmentalVariables>;
}

/** The model families planned for later stages. */
export type ModelId =
  | "lightgbm"
  | "xgboost"
  | "random_forest"
  | "gam_maxent"
  | "tabular_mlp"
  | "spatial_gnn";

export interface ModelMeta {
  id: ModelId;
  label: string;
  status: "pending" | "training" | "ready";
}

/**
 * One row of the Model Performance section — spatial-CV evaluation metrics
 * for a model family. This describes past evaluation results, not a live,
 * connected prediction (see predictionService.ts / AVAILABLE_MODELS).
 */
export interface ModelPerformanceMetrics {
  modelId: ModelId;
  label: string;
  spatialRocAuc: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

/** One band of the habitat-suitability classification scheme. */
export interface SuitabilityBand {
  id: SuitabilityClass;
  label: string;
  min: number;
  max: number;
  color: string;
}

/**
 * Which top-level page of the dashboard is showing. Only "map" is
 * user-facing today — Model Performance and Reports were removed from the
 * dashboard's navigation/routing (see Header.tsx / App.tsx). The type stays
 * a union (rather than collapsing to a bare string literal) so a future
 * view can be reintroduced without redesigning the state shape.
 */
export type AppView = "map";

/** Whether clicking the map browses observed records or runs a suitability prediction. */
export type InteractionMode = "browse" | "predict" | "density";

export type BaseMapStyle = "standard" | "satellite";

/* ==========================================================================
   Real-dataset filtering, summary and map-aggregation types
   ========================================================================== */

export type PresenceFilter = "all" | "presence" | "absence";
export type SpatialFoldFilter = "all" | SpatialFold;

/** Restricts the currently-selected environmental variable to a [min, max] range. */
export interface EnvRangeFilter {
  variable: keyof EnvironmentalVariables;
  min: number;
  max: number;
}

export interface FilterState {
  species: string | "all";
  presence: PresenceFilter;
  fold: SpatialFoldFilter;
  /** Only meaningful for the currently-selected environmental variable; reset on mode switch. */
  envRange: EnvRangeFilter | null;
}

export const DEFAULT_FILTERS: FilterState = {
  species: "all",
  presence: "all",
  fold: "all",
  envRange: null,
};

/* ==========================================================================
   Environmental map-mode types (Stage 1.5)
   ========================================================================== */

/**
 * What the map is currently rendering. "occurrences" is the original
 * presence/absence point/cluster view; every other value is one of the 8
 * environmental variables, rendered as a spatially aggregated grid of
 * *observed* values (never an interpolated/predicted surface).
 */
export type MapMode = "occurrences" | keyof EnvironmentalVariables;

/**
 * One cell of the environmental spatial-aggregation grid. Unlike GridCell
 * (occurrence counts), this carries the descriptive statistics needed to
 * color and label a cell for a single environmental variable.
 */
export interface EnvGridCell {
  id: string;
  latCenter: number;
  lngCenter: number;
  bounds: { south: number; north: number; west: number; east: number };
  count: number;
  mean: number;
  min: number;
  max: number;
  /** Index of a real occurrence used when the user opens a cell. */
  representativeIndex: number;
}

/** Descriptive statistics for one environmental variable over a set of records. */
export interface EnvironmentalStats {
  variable: keyof EnvironmentalVariables;
  min: number;
  max: number;
  mean: number;
  median: number;
  count: number;
}

/** Aggregate statistics for the full dataset or any filtered subset of it. */

export interface SpeciesProfile {
  scientificName: string;
  occurrenceCount: number;
  latitudeRange: [number, number];
  longitudeRange: [number, number];
  environmentalRanges: Record<keyof EnvironmentalVariables, [number, number]>;
}

export interface DatasetSummary {
  recordCount: number;
  presenceCount: number;
  absenceCount: number;
  /** Distinct real scientific names among presence records, excluding Pseudo_Absence. */
  speciesCount: number;
  latitudeRange: [number, number];
  longitudeRange: [number, number];
  environmentalRanges: Record<keyof EnvironmentalVariables, [number, number]>;
  foldCounts: Record<SpatialFold, number>;
}

export type DatasetStatus = "idle" | "loading" | "ready" | "error";

/**
 * One representative occurrence POINT used to summarize many records when
 * zoomed out. This is spatial sampling, not a rendered rectangle/cell: the
 * map draws a single fixed-size CircleMarker at (representativeLat,
 * representativeLng) — the real coordinate of an actual observation drawn
 * from the bin — never a bounding-box shape. `bounds` is kept only as the
 * internal binning key (used to fit/zoom when the user clicks the point),
 * it is never rendered.
 */
export interface GridCell {
  id: string;
  latCenter: number;
  lngCenter: number;
  bounds: { south: number; north: number; west: number; east: number };
  count: number;
  presenceCount: number;
  absenceCount: number;
  /** Real coordinate of an actual observation in this bin, used as the marker position. */
  representativeLat: number;
  representativeLng: number;
  /** Row index of the actual observation represented by this point. */
  representativeIndex: number;
  /** Real observation for the opposite class when this bin contains both classes. */
  secondaryRepresentativeLat: number | null;
  secondaryRepresentativeLng: number | null;
  secondaryRepresentativeIndex: number | null;
  /** Presence/absence of the representative observation (majority class of the bin). */
  presence: 0 | 1;
}
