import type {
  EnvironmentalVariables,
  HabitatPrediction,
  MapLocation,
  ModelMeta,
} from "@/types/environmental";
import { ENV_VARIABLE_KEYS } from "@/types/environmental";
import { classifySuitability } from "@/data/suitabilityConfig";
import { getEnvironmentalRanges } from "@/services/dataService";
import { FINAL_MODEL } from "@/services/modelConfig";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export const AVAILABLE_MODELS: ModelMeta[] = [
  { id: "random_forest", label: "Random Forest", status: "ready" },
];

export function findOutOfRangeVariables(
  values: EnvironmentalVariables
): Array<keyof EnvironmentalVariables> {
  const ranges = getEnvironmentalRanges();
  const out: Array<keyof EnvironmentalVariables> = [];
  for (const key of ENV_VARIABLE_KEYS) {
    const [min, max] = ranges[key];
    const v = values[key];
    if (!Number.isFinite(v) || v < min || v > max) out.push(key);
  }
  return out;
}

export interface EnvironmentalSourceMeta {
  provider: string;
  version: string;
  temporal_basis: string;
  baseline_time_slice: string;
  spatial_resolution: string;
  spatial_resolution_km_equator: string;
  access: string;
}

/**
 * Runs the production Random Forest through FastAPI. The backend retrieves
 * the eight Bio-ORACLE baseline variables for the clicked coordinate and
 * runs the saved model. The browser never loads the model artifact.
 */
export async function getPrediction(
  location: MapLocation,
  signal?: AbortSignal
): Promise<HabitatPrediction> {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        latitude: location.latitude,
        longitude: location.longitude,
      }),
      signal,
    });

    let data: any = null;
    try {
      data = await response.json();
    } catch {
      throw new Error(`Prediction service returned HTTP ${response.status}.`);
    }

    if (!response.ok) {
      throw new Error(data?.detail || "Prediction service returned an error.");
    }

    const probability = Number(data.probability);
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
      throw new Error("Prediction service returned an invalid suitability score.");
    }

    const environmentalValues = data.environment as EnvironmentalVariables;
    const outOfRangeVariables = findOutOfRangeVariables(environmentalValues);

    return {
      modelId: FINAL_MODEL,
      location,
      probability,
      suitabilityClass: classifySuitability(probability),
      confidence: null,
      environmentalValues,
      environmentalSource: data.environmental_source as EnvironmentalSourceMeta,
      isLive: true,
      outOfRangeVariables,
    };
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    console.error("GeoSea prediction request failed:", error);
    return {
      modelId: FINAL_MODEL,
      location,
      probability: null,
      suitabilityClass: classifySuitability(null),
      confidence: null,
      environmentalValues: null,
      environmentalSource: null,
      isLive: false,
      errorMessage: error instanceof Error ? error.message : "Prediction service unavailable",
      outOfRangeVariables: [],
    };
  }
}
