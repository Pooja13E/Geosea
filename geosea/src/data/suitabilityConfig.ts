/**
 * Habitat suitability classification scheme.
 *
 * Kept as a single configurable table (rather than thresholds hard-coded
 * throughout components) so the bands can be tuned later without touching
 * MapView, LocationInfoPanel, or the legend. Colors follow a single-hue
 * sequential ramp on the same warm "sand/attention" accent used elsewhere
 * for prediction, per the design tokens in src/index.css.
 */
import type { SuitabilityBand, SuitabilityClass } from "@/types/environmental";

export const SUITABILITY_BANDS: SuitabilityBand[] = [
  { id: "very_low", label: "Very Low", min: 0.0, max: 0.2, color: "#eef2f1" },
  { id: "low", label: "Low", min: 0.2, max: 0.4, color: "#cfe0dc" },
  { id: "moderate", label: "Moderate", min: 0.4, max: 0.6, color: "#e0a85c" },
  { id: "high", label: "High", min: 0.6, max: 0.8, color: "#c96a4e" },
  { id: "very_high", label: "Very High", min: 0.8, max: 1.0, color: "#9c3d27" },
];

/** Maps the Random Forest positive-class score to a suitability band. */
export function classifySuitability(probability: number | null): SuitabilityClass {
  if (probability === null || !Number.isFinite(probability)) return "unknown";
  const clamped = Math.min(1, Math.max(0, probability));
  for (const band of SUITABILITY_BANDS) {
    // Last band is inclusive of the upper bound (1.0); others use [min, max).
    if (clamped >= band.min && (clamped < band.max || band.max === 1)) return band.id;
  }
  return "unknown";
}

export function suitabilityBand(id: SuitabilityClass): SuitabilityBand | null {
  return SUITABILITY_BANDS.find((b) => b.id === id) ?? null;
}

export const UNKNOWN_SUITABILITY_COLOR = "#b7c2c6";
