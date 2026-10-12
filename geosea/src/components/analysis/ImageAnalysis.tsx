import { useEffect, useMemo, useState } from "react";
import { MapView } from "@/components/map/MapView";
import { DEFAULT_FILTERS, type BaseMapStyle, type MapLocation } from "@/types/environmental";
import { getAllIndices, getFilteredRecords } from "@/services/dataService";
import "./ImageAnalysis.css";

interface SpeciesCandidate {
  raw_class: string;
  scientific_name: string;
  algae_group: string;
  phylum: string;
  common_name: string;
  confidence_score: number;
  percentage: number;
  habitat_summary: string;
  known_range: string;
}

interface SeaweedAnalysisResult {
  model_name: string;
  model_architecture: string;
  model_classes_count: number;
  is_uncertain: boolean;
  uncertainty_note?: string;
  top_candidate?: SpeciesCandidate;
  alternative_candidates: SpeciesCandidate[];
  classifier_note: string;
  scientific_advice_note: string;
}

interface VerificationVariable {
  key: string;
  label: string;
  unit: string;
  current_value: number;
  consistency_score: number;
  median?: number | null;
  q25?: number | null;
  q75?: number | null;
}

interface VerificationResult {
  environmental_consistency: {
    score: number;
    level: "high" | "moderate" | "low";
    label: string;
    variables: VerificationVariable[];
  };
}


const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

function explainRequestError(error: unknown, action: string): string {
  if (error instanceof TypeError && /fetch/i.test(error.message)) {
    return `Cannot connect to the GeoSea backend at ${API_URL}. Start the backend in a terminal from the backend folder with: py -m uvicorn app:app --reload --host 127.0.0.1 --port 8000. Then open ${API_URL}/health to confirm it is running. If the backend is running, check its terminal for a startup/import error and make sure the frontend URL is allowed by GEOSEA_CORS_ORIGINS.`;
  }
  return error instanceof Error ? error.message : `${action} failed.`;
}

const GEOSEA_BOUNDS = {
  south: -60,
  north: 30,
  west: 20,
  east: 147,
};

function formatValue(value: number | null | undefined, unit: string) {
  if (value == null || !Number.isFinite(value)) return "—";
  const decimals = unit === "°C" || unit === "PSU" ? 2 : 3;
  return `${value.toFixed(decimals)}${unit ? ` ${unit}` : ""}`;
}

function consistencyBand(score: number) {
  if (score >= 80) return { label: "High consistency", short: "HIGH", tone: "high" } as const;
  if (score >= 50) return { label: "Medium consistency", short: "MEDIUM", tone: "medium" } as const;
  return { label: "Low consistency", short: "LOW", tone: "low" } as const;
}

export function ImageAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SeaweedAnalysisResult | null>(null);
  const [location, setLocation] = useState<MapLocation | null>(null);
  const [verification, setVerification] = useState<VerificationResult | null>(null);
  const [mapStyle, setMapStyle] = useState<BaseMapStyle>("standard");

  const allIndices = useMemo(() => getAllIndices(), []);
  const mapIndices = useMemo(() => {
    if (!result?.top_candidate?.scientific_name) return allIndices;
    return getFilteredRecords({
      ...DEFAULT_FILTERS,
      species: result.top_candidate.scientific_name,
      presence: "presence",
    });
  }, [allIndices, result?.top_candidate?.scientific_name]);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Please select a JPG, PNG, or WebP image.");
      return;
    }
    if (selected.size > 15 * 1024 * 1024) {
      setError("File size exceeds the 15 MB limit.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setLocation(null);
    setVerification(null);
    setError(null);
  }

  async function identifyImage() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setLocation(null);
    setVerification(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("source_type", "underwater");

      const response = await fetch(`${API_URL}/analyze-image`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json().catch(() => null);
      if (!response.ok) throw new Error(data?.detail || `Image analysis failed (HTTP ${response.status}).`);
      setResult(data as SeaweedAnalysisResult);
    } catch (err) {
      setError(explainRequestError(err, "Image analysis"));
    } finally {
      setLoading(false);
    }
  }

  function handleMapLocation(next: MapLocation | null) {
    setLocation(next);
    setVerification(null);
    setError(null);
  }

  async function verifyAtLocation() {
    if (!result?.top_candidate || !location) return;
    setVerifying(true);
    setError(null);
    setVerification(null);

    try {
      const params = new URLSearchParams({
        scientific_name: result.top_candidate.scientific_name,
        latitude: String(location.latitude),
        longitude: String(location.longitude),
      });
      const response = await fetch(`${API_URL}/species-verification?${params.toString()}`, { method: "POST" });
      const data = await response.json().catch(() => null);
      if (!response.ok) throw new Error(data?.detail || `GeoSea verification failed (HTTP ${response.status}).`);
      setVerification(data as VerificationResult);
    } catch (err) {
      setError(explainRequestError(err, "GeoSea verification"));
    } finally {
      setVerifying(false);
    }
  }

  return (
    <div className="si-page">
      <div className="si-workspace">
        <section className="si-map-panel">
          <div className="si-map-toolbar">
            <div className="si-map-toolbar__copy">
              <p className="si-eyebrow">Step 2 · Observation location</p>
              <strong>{result?.top_candidate ? `GeoSea occurrences — ${result.top_candidate.scientific_name}` : "GeoSea observation map"}</strong>
              <span>
                {result?.top_candidate
                  ? "Known GeoSea presence records are highlighted. Click the actual observation location to verify the image prediction."
                  : "This is the same occurrence map used on the main GeoSea dashboard."}
              </span>
            </div>
            <div className="si-map-actions">
              <button type="button" className={mapStyle === "standard" ? "active" : ""} onClick={() => setMapStyle("standard")}>Map</button>
              <button type="button" className={mapStyle === "satellite" ? "active" : ""} onClick={() => setMapStyle("satellite")}>Satellite</button>
            </div>
          </div>
          <div className="si-map-wrap">
            <MapView
              filteredIndices={mapIndices}
              mapMode="occurrences"
              envDomain={null}
              dataBounds={GEOSEA_BOUNDS}
              baseMapStyle={mapStyle}
              fitSignal={0}
              interactionMode="verification"
              onSelectedLocationChange={handleMapLocation}
            />
          </div>
          <div className="si-map-status">
            {location ? (
              <>
                <strong>Selected observation:</strong>
                <span>{location.latitude.toFixed(4)}° {location.latitude >= 0 ? "N" : "S"}</span>
                <span>{location.longitude.toFixed(4)}° {location.longitude >= 0 ? "E" : "W"}</span>
              </>
            ) : (
              <span>Click anywhere on the map to select the specimen's observation location.</span>
            )}
          </div>
        </section>

        <aside className="si-side-panel">
          <section className="si-card si-step1-card">
            <div className="si-card-title">
              <div>
                <p className="si-eyebrow">Step 1 · Image identification</p>
                <h3>Upload seaweed image</h3>
              </div>
              <span className="si-model">EfficientNet-B0 V2</span>
            </div>

            <label className="si-upload">
              <input type="file" accept="image/png,image/jpeg,image/webp" onChange={handleFileChange} />
              {preview ? <img src={preview} alt="Uploaded seaweed specimen" /> : <span className="si-upload-icon">📷</span>}
              <strong>{file ? file.name : "Choose a seaweed image"}</strong>
              <small>JPG, PNG or WebP · maximum 15 MB</small>
            </label>

            <button type="button" className="si-primary" disabled={!file || loading} onClick={identifyImage}>
              {loading ? "Classifying image…" : result ? "Re-identify image" : "Identify Seaweed Species"}
            </button>

            {result?.top_candidate && (
              <div className="si-visual-result">
                <div className="si-visual-result__heading">
                  <div>
                    <span className="si-eyebrow">Visual confidence</span>
                    <h4><em>{result.top_candidate.scientific_name}</em></h4>
                    <p>{result.top_candidate.common_name}</p>
                  </div>
                  <div className="si-score-stack">
                    <span className="si-big-score">{result.top_candidate.percentage.toFixed(0)}%</span>
                    <span className={`si-score-label si-score-label--${consistencyBand(result.top_candidate.percentage).tone}`}>{consistencyBand(result.top_candidate.percentage).short}</span>
                  </div>
                </div>
                <div className="si-progress"><i style={{ width: `${Math.max(0, Math.min(100, result.top_candidate.percentage))}%` }} /></div>
                <p className="si-note">{result.classifier_note}</p>
                {result.is_uncertain && <div className="si-warning"><strong>Model uncertainty</strong><span>{result.uncertainty_note}</span></div>}
              </div>
            )}
          </section>

          <section className="si-card si-step3-card">
            <div className="si-card-title">
              <div>
                <p className="si-eyebrow">Step 3 · GeoSea environmental evidence</p>
                <h3>Environmental variables &amp; consistency</h3>
              </div>
              {verification ? (
                <div className="si-score-stack">
                  <span className="si-big-score si-big-score--secondary">{verification.environmental_consistency.score.toFixed(0)}/100</span>
                  <span className={`si-score-label si-score-label--${consistencyBand(verification.environmental_consistency.score).tone}`}>{consistencyBand(verification.environmental_consistency.score).short}</span>
                </div>
              ) : <span className="si-pending-badge">Waiting for location</span>}
            </div>

            {!result?.top_candidate ? (
              <p className="si-muted">Complete Step 1 first to identify the specimen.</p>
            ) : !location ? (
              <p className="si-muted">Complete Step 2 by selecting the specimen's actual observation location on the map.</p>
            ) : (
              <>
                <div className="si-location"><strong>Observation selected</strong><span>{location.latitude.toFixed(4)}°, {location.longitude.toFixed(4)}°</span></div>
                <button type="button" className="si-primary si-primary--verify" disabled={verifying} onClick={verifyAtLocation}>
                  {verifying ? "Checking Bio-ORACLE + GeoSea…" : "Check GeoSea Variables"}
                </button>
              </>
            )}

            {verification && (
              <div className="si-variable-result">
                <div className="si-environment-summary">
                  <div>
                    <span>Overall environmental consistency</span>
                    <strong>{verification.environmental_consistency.score.toFixed(0)}/100 · {consistencyBand(verification.environmental_consistency.score).label}</strong>
                  </div>
                  <p>These scores compare the selected location's Bio-ORACLE baseline conditions with GeoSea presence records for the predicted species. They are descriptive consistency measures, not species-identification probabilities.</p>
                </div>

                <div className="si-section-label">8 environmental variables</div>
                <div className="si-variable-list">
                  {verification.environmental_consistency.variables.map((item) => (
                    <div className="si-variable" key={item.key}>
                      <div className="si-variable-top"><strong>{item.label}</strong><span>{formatValue(item.current_value, item.unit)}</span></div>
                      <div className="si-progress si-progress--small"><i style={{ width: `${Math.max(0, Math.min(100, item.consistency_score))}%` }} /></div>
                      <small><strong className={`si-inline-band si-inline-band--${consistencyBand(item.consistency_score).tone}`}>{consistencyBand(item.consistency_score).short}</strong> · {item.consistency_score.toFixed(0)}/100 consistency{item.median != null ? ` · GeoSea median ${formatValue(item.median, item.unit)}` : ""}</small>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>


          {error && <div className="si-error">{error}</div>}
        </aside>
      </div>

    </div>
  );
}
