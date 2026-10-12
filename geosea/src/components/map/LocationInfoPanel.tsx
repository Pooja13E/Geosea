import { useEffect, useRef, useState } from "react";
import type { HabitatPrediction, MapLocation, OccurrenceRecord } from "@/types/environmental";
import { ENV_VARIABLE_KEYS } from "@/types/environmental";
import { ENV_VARIABLE_META } from "@/data/envVariableConfig";
import { suitabilityBand, UNKNOWN_SUITABILITY_COLOR } from "@/data/suitabilityConfig";
import { formatCoordinate, formatVariable } from "@/utils/formatters";
import { getEnvironmentalRanges } from "@/services/dataService";
import { getSentinel2Observation, getSentinel3Observation } from "@/services/remoteSensingService";
import type { SatelliteObservation } from "@/services/remoteSensingService";
import "./LocationInfoPanel.css";

const VARIABLE_ROWS: Array<{
  key: keyof Pick<
    OccurrenceRecord,
    "temperature" | "salinity" | "chlorophyll" | "nitrate" | "phosphate" | "pH" | "PAR" | "kdpar"
  >;
  label: string;
  unit: string;
}> = [
  { key: "temperature", label: "Temperature", unit: "°C" },
  { key: "salinity", label: "Salinity", unit: "PSU" },
  { key: "chlorophyll", label: "Chlorophyll", unit: "mg/m³" },
  { key: "nitrate", label: "Nitrate", unit: "µmol/L" },
  { key: "phosphate", label: "Phosphate", unit: "µmol/L" },
  { key: "pH", label: "pH", unit: "" },
  { key: "PAR", label: "PAR", unit: "mol/m²/d" },
  { key: "kdpar", label: "kdPAR", unit: "m⁻¹" },
];

type LocationInfoPanelProps =
  | { kind: "record"; record: OccurrenceRecord; onClose: () => void }
  | {
      kind: "prediction";
      location: MapLocation;
      status: "loading" | "loaded";
      prediction: HabitatPrediction | null;
      onClose: () => void;
    };

export function LocationInfoPanel(props: LocationInfoPanelProps) {
  if (props.kind === "record") {
    return <ObservedRecordPanel record={props.record} onClose={props.onClose} />;
  }
  return (
    <PredictedLocationPanel
      location={props.location}
      status={props.status}
      prediction={props.prediction}
      onClose={props.onClose}
    />
  );
}

function ObservedRecordPanel({ record, onClose }: { record: OccurrenceRecord; onClose: () => void }) {
  const displayName = record.isPseudoAbsence ? "Pseudo-absence / absence" : record.scientific_name;

  return (
    <div className="gs-info-panel" role="dialog" aria-label="Observation information">
      <div className="gs-info-panel__header">
        <div>
          <p className="gs-info-panel__eyebrow">Observed Record</p>
          <h3 className={record.isPseudoAbsence ? "" : "gs-info-panel__scientific-name"}>{displayName}</h3>
        </div>
        <button type="button" className="gs-info-panel__close" onClick={onClose} aria-label="Close panel">
          ×
        </button>
      </div>

      <div
        className={
          "gs-info-panel__badge-row" +
          (record.presence ? " gs-info-panel__badge-row--presence" : " gs-info-panel__badge-row--absence")
        }
      >
        <span className="gs-info-panel__badge-dot" />
        <span>{record.presence ? "Presence recorded" : "Absence / background sample"}</span>
      </div>

      <p className="gs-info-panel__coords">
        {formatCoordinate(record.latitude)}, {formatCoordinate(record.longitude)}
      </p>

      <div className="gs-info-panel__divider" />

      <dl className="gs-info-panel__variables">
        {VARIABLE_ROWS.map((row) => (
          <div className="gs-info-panel__variable" key={row.key}>
            <dt>{row.label}</dt>
            <dd>{formatVariable(record[row.key], row.unit, row.key === "pH" ? 2 : 1)}</dd>
          </div>
        ))}
      </dl>

      <details className="gs-info-panel__details">
        <summary>Technical details</summary>
        <div className="gs-info-panel__variable">
          <dt>Spatial fold</dt>
          <dd>{record.spatial_fold}</dd>
        </div>
      </details>

      <p className="gs-info-panel__footnote">Values are read directly from the loaded dataset.</p>
    </div>
  );
}

function PredictedLocationPanel({
  location,
  status,
  prediction,
  onClose,
}: {
  location: MapLocation;
  status: "loading" | "loaded";
  prediction: HabitatPrediction | null;
  onClose: () => void;
}) {
  const band = prediction ? suitabilityBand(prediction.suitabilityClass) : null;
  const trainingRanges = getEnvironmentalRanges();
  const hasWarnings = (prediction?.outOfRangeVariables.length ?? 0) > 0;
  const [satelliteStatus, setSatelliteStatus] = useState<"idle" | "loading" | "loaded" | "error">("idle");
  const [satelliteObservation, setSatelliteObservation] = useState<SatelliteObservation | null>(null);
  const [satelliteSource, setSatelliteSource] = useState<"sentinel2" | "sentinel3">("sentinel2");
  const [satelliteError, setSatelliteError] = useState<string | null>(null);
  const [satelliteFallback, setSatelliteFallback] = useState(false);
  const satelliteAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    satelliteAbortRef.current?.abort();
    const controller = new AbortController();
    satelliteAbortRef.current = controller;
    setSatelliteStatus("loading");
    setSatelliteObservation(null);
    setSatelliteError(null);
    setSatelliteFallback(false);

    async function loadSatellite() {
      try {
        const observation = await getSentinel2Observation(location, 365, controller.signal);
        if (controller.signal.aborted) return;
        setSatelliteSource("sentinel2");
        setSatelliteObservation(observation);
        setSatelliteStatus("loaded");
        return;
      } catch (sentinel2Error) {
        if (controller.signal.aborted) return;
        try {
          const observation = await getSentinel3Observation(location, 365, controller.signal);
          if (controller.signal.aborted) return;
          setSatelliteSource("sentinel3");
          setSatelliteFallback(true);
          setSatelliteObservation(observation);
          setSatelliteStatus("loaded");
        } catch (sentinel3Error) {
          if (controller.signal.aborted) return;
          // Keep raw backend/proxy errors out of the UI; explain the issue clearly.
          console.warn("Satellite imagery lookup failed", { sentinel2Error, sentinel3Error });
          setSatelliteError("No usable Sentinel-2 or Sentinel-3 image could be retrieved for this location and date range. Check that the backend is running and Google Earth Engine is configured, or try a nearby coastal location.");
          setSatelliteStatus("error");
        }
      }
    }

    void loadSatellite();
    return () => controller.abort();
  }, [location.latitude, location.longitude]);

  useEffect(() => () => satelliteAbortRef.current?.abort(), []);

  return (
    <div className="gs-info-panel" role="dialog" aria-label="Habitat suitability prediction">
      <div className="gs-info-panel__header">
        <div>
          <p className="gs-info-panel__eyebrow">Predicted Location</p>
          <h3>Habitat Suitability</h3>
        </div>
        <button type="button" className="gs-info-panel__close" onClick={onClose} aria-label="Close panel">
          ×
        </button>
      </div>

      <div className="gs-info-panel__divider" />

      {status === "loading" ? (
        <p className="gs-info-panel__loading">Looking up environmental conditions…</p>
      ) : (
        <>
          <div className="gs-suitability-result">
            <div
              className="gs-suitability-result__dot"
              style={{ background: band?.color ?? UNKNOWN_SUITABILITY_COLOR }}
            />
            <div>
              <p className="gs-suitability-result__value">
                {prediction?.probability !== null && prediction?.probability !== undefined
                  ? `${Math.round(prediction.probability * 100)}%`
                  : "Unavailable"}
              </p>
              <p className="gs-suitability-result__label">{band ? `${band.label} · model score` : "Prediction unavailable"}</p>
            </div>
          </div>

          {!prediction?.isLive && (
            <p className="gs-info-panel__status-note">
              Live prediction unavailable. {prediction?.errorMessage || "Check that the GeoSea inference server is running."}
            </p>
          )}

          {hasWarnings && (
            <p className="gs-info-panel__warning">
              Warning: environmental conditions at this location are outside the model training range
              for {prediction!.outOfRangeVariables.map((k) => ENV_VARIABLE_META[k].label).join(", ")}.
              Prediction may be unreliable.
            </p>
          )}

          <div className="gs-info-panel__divider" />

          <p className="gs-info-panel__section-label">Environmental Profile</p>
          {prediction?.environmentalValues ? (
            <div className="gs-env-profile">
              {ENV_VARIABLE_KEYS.map((key) => {
                const meta = ENV_VARIABLE_META[key];
                const outOfRange = prediction.outOfRangeVariables.includes(key);
                const value = prediction.environmentalValues![key];
                const [min, max] = trainingRanges[key];
                const percent = Math.max(4, Math.min(100, ((value - min) / (max - min || 1)) * 100));
                return (
                  <div className={"gs-env-profile__row" + (outOfRange ? " gs-env-profile__row--warn" : "")} key={key}>
                    <div className="gs-env-profile__top">
                      <span>{meta.label}</span>
                      <strong>{formatVariable(value, meta.unit, meta.decimals)}</strong>
                    </div>
                    <div className="gs-env-profile__track" aria-hidden="true">
                      <span style={{ width: `${percent}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="gs-info-panel__empty">Environmental data could not be retrieved for this coordinate.</p>
          )}

          <div className="gs-info-panel__divider" />
          <section className="gs-satellite-card" aria-label="Satellite observation">
            <div className="gs-satellite-card__heading">
              <div>
                <p className="gs-info-panel__section-label">Satellite observation</p>
                <h4>{satelliteSource === "sentinel3" ? "Sentinel-3 fallback imagery" : "Sentinel-2 imagery"}</h4>
              </div>
              <span className="gs-satellite-card__icon" aria-hidden="true">⌁</span>
            </div>
            {satelliteStatus === "loading" && (
              <p className="gs-info-panel__loading" role="status">Searching recent Sentinel-2 imagery; Sentinel-3 will be tried automatically if needed…</p>
            )}
            {satelliteStatus === "error" && (
              <div className="gs-satellite-card__error" role="alert">
                <p>{satelliteError}</p>
                <button type="button" className="gs-satellite-card__retry" onClick={() => window.location.reload()}>Try again</button>
              </div>
            )}
            {satelliteStatus === "loaded" && satelliteObservation && (
              <div className="gs-satellite-card__result">
                {satelliteFallback && (
                  <p className="gs-satellite-card__fallback">Sentinel-2 was not usable here, so Sentinel-3 is being shown as the broader-area fallback.</p>
                )}
                <img
                  className="gs-satellite-card__image"
                  src={satelliteObservation.thumbnail_url}
                  alt={`${satelliteObservation.platform} satellite image around ${formatCoordinate(location.latitude)}, ${formatCoordinate(location.longitude)} captured ${satelliteObservation.observation_date}`}
                  loading="lazy"
                  onError={() => { setSatelliteError("The satellite image could not be displayed. Please try selecting the location again."); setSatelliteStatus("error"); setSatelliteObservation(null); }}
                />
                <div className="gs-satellite-card__metadata">
                  <span>Source</span>
                  <strong>{satelliteSource === "sentinel2" ? "Sentinel-2" : "Sentinel-3 OLCI"}</strong>
                  <span>Observation date</span>
                  <strong>{satelliteObservation.observation_date}</strong>
                  <span>Scene cloud cover</span>
                  <strong>{satelliteObservation.scene_cloud_percentage === null ? "Not reported" : `${satelliteObservation.scene_cloud_percentage}%`}</strong>
                  <span>{satelliteSource === "sentinel3" ? "Spatial resolution" : "Clear pixels near point"}</span>
                  <strong>{satelliteSource === "sentinel3" ? "~300 m" : `${satelliteObservation.local_clear_percentage}%`}</strong>
                  <span>Scenes searched</span>
                  <strong>{satelliteObservation.scenes_considered.toLocaleString()}</strong>
                </div>
                <p className="gs-satellite-card__caption">{satelliteObservation.visualization} · approx. {satelliteObservation.thumbnail_radius_meters / 1000} km around selected point.</p>
              </div>
            )}
          </section>


          {prediction?.environmentalSource && (
            <p className="gs-info-panel__footnote">
              <strong>Source:</strong> {prediction.environmentalSource.provider} {prediction.environmentalSource.version}.
              <br />
              <strong>Temporal basis:</strong> {prediction.environmentalSource.temporal_basis}.
              <br />
              <strong>Resolution:</strong> {prediction.environmentalSource.spatial_resolution}.
              <br />
              Environmental values are baseline climatology values, not real-time observations.
            </p>
          )}
        </>
      )}
    </div>
  );
}
