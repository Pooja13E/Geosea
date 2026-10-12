import type { MapLocation } from "@/types/environmental";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export interface SatelliteObservation {
  latitude: number;
  longitude: number;
  provider: string;
  platform: string;
  collection: string;
  observation_date: string;
  scene_cloud_percentage: number | null;
  local_clear_percentage: number;
  local_data_percentage?: number | null;
  scenes_considered: number;
  scenes_quality_checked: number;
  search_start_date: string;
  search_end_date: string;
  thumbnail_url: string;
  thumbnail_radius_meters: number;
  visualization: string;
  interpretation_note: string;
  variables?: Array<{ key: string; label: string; value: number | null; unit: string; description: string }>;
  reflectance_observation_date?: string | null;
}

async function getObservation(
  endpoint: "sentinel2" | "sentinel3",
  location: MapLocation,
  lookbackDays = 365,
  signal?: AbortSignal
): Promise<SatelliteObservation> {
  const response = await fetch(`${API_URL}/remote-sensing/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      latitude: location.latitude,
      longitude: location.longitude,
      lookback_days: lookbackDays,
    }),
    signal,
  });

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(data?.detail || `Satellite imagery service returned HTTP ${response.status}.`);
  }
  if (!data?.thumbnail_url || !data?.observation_date) {
    throw new Error("The satellite service returned incomplete observation metadata.");
  }
  return data as SatelliteObservation;
}

export function getSentinel2Observation(location: MapLocation, lookbackDays = 365, signal?: AbortSignal) {
  return getObservation("sentinel2", location, lookbackDays, signal);
}

export function getSentinel3Observation(location: MapLocation, lookbackDays = 365, signal?: AbortSignal) {
  return getObservation("sentinel3", location, lookbackDays, signal);
}

