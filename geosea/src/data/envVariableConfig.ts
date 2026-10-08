/**
 * Display metadata for the 8 environmental variables: label, unit, decimal
 * precision, and a single-hue sequential color ramp (light -> dark) used to
 * shade the environmental grid-aggregation cells on the map.
 *
 * Each variable gets its own hue so switching Map Layer is visually
 * distinct, but within any one legend the ramp is a plain light-to-dark
 * sequential scale — never a rainbow/jet scale — per the project's
 * scientific-visualization requirement.
 */
import type { EnvironmentalVariables, MapMode } from "@/types/environmental";
import { ENV_VARIABLE_KEYS } from "@/types/environmental";

export interface EnvVariableMeta {
  key: keyof EnvironmentalVariables;
  label: string;
  unit: string;
  decimals: number;
  /** Sequential ramp endpoints, low -> high value. */
  colorLow: string;
  colorHigh: string;
}

export const ENV_VARIABLE_META: Record<keyof EnvironmentalVariables, EnvVariableMeta> = {
  temperature: {
    key: "temperature",
    label: "Temperature",
    unit: "°C",
    decimals: 1,
    colorLow: "#fef0d9",
    colorHigh: "#9c2c1c",
  },
  salinity: {
    key: "salinity",
    label: "Salinity",
    unit: "PSU",
    decimals: 2,
    colorLow: "#e3edf7",
    colorHigh: "#0b4f8a",
  },
  chlorophyll: {
    key: "chlorophyll",
    label: "Chlorophyll",
    unit: "mg/m³",
    decimals: 2,
    colorLow: "#eef6e4",
    colorHigh: "#2e6b34",
  },
  nitrate: {
    key: "nitrate",
    label: "Nitrate",
    unit: "µmol/L",
    decimals: 2,
    colorLow: "#f1eaf6",
    colorHigh: "#5f2d78",
  },
  phosphate: {
    key: "phosphate",
    label: "Phosphate",
    unit: "µmol/L",
    decimals: 3,
    colorLow: "#faf1e4",
    colorHigh: "#8a5518",
  },
  pH: {
    key: "pH",
    label: "pH",
    unit: "",
    decimals: 2,
    colorLow: "#e6f4f2",
    colorHigh: "#0f6b62",
  },
  PAR: {
    key: "PAR",
    label: "PAR",
    unit: "mol/m²/d",
    decimals: 1,
    colorLow: "#fdf6de",
    colorHigh: "#a3790a",
  },
  kdpar: {
    key: "kdpar",
    label: "kdPAR",
    unit: "m⁻¹",
    decimals: 3,
    colorLow: "#eceef4",
    colorHigh: "#33395c",
  },
};

/** Ordered list of map modes for the sidebar selector: Occurrences first, then the 8 variables. */
export const MAP_MODE_ORDER: MapMode[] = ["occurrences", ...ENV_VARIABLE_KEYS];

export function isEnvironmentalMode(
  mode: MapMode
): mode is keyof EnvironmentalVariables {
  return mode !== "occurrences";
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace("#", "");
  const num = Number.parseInt(clean, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

/** Interpolates a variable's ramp at t in [0, 1] (0 = low value, 1 = high value). */
export function envColorScale(variable: keyof EnvironmentalVariables, t: number): string {
  const meta = ENV_VARIABLE_META[variable];
  const clamped = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
  const [r1, g1, b1] = hexToRgb(meta.colorLow);
  const [r2, g2, b2] = hexToRgb(meta.colorHigh);
  const r = Math.round(r1 + (r2 - r1) * clamped);
  const g = Math.round(g1 + (g2 - g1) * clamped);
  const b = Math.round(b1 + (b2 - b1) * clamped);
  return `rgb(${r}, ${g}, ${b})`;
}
