import { useState, useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
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

interface OccurrenceRecord {
  latitude: number;
  longitude: number;
  country: string;
  event_date: string;
  dataset: string;
  basis_of_record: string;
}

interface WoRMSData {
  source?: string;
  aphia_id?: number;
  scientific_name?: string;
  valid_name?: string;
  authority?: string;
  taxonomic_status?: string;
  rank?: string;
  taxonomy?: {
    kingdom?: string;
    phylum?: string;
    class?: string;
    order?: string;
    family?: string;
    genus?: string;
  };
  url?: string;
  citation?: string;
  status?: string;
}

interface GBIFData {
  source?: string;
  taxon_key?: number;
  matched_scientific_name?: string;
  taxonomy?: {
    kingdom?: string;
    phylum?: string;
    class?: string;
    order?: string;
    family?: string;
    genus?: string;
    species?: string;
  };
  total_occurrences_documented?: number;
  occurrences_sample?: OccurrenceRecord[];
  url?: string;
  data_limitation_note?: string;
}

interface SeaweedAnalysisResult {
  model_name: string;
  model_architecture: string;
  model_classes_count: number;
  is_uncertain: boolean;
  uncertainty_note?: string;
  top_candidate?: SpeciesCandidate;
  alternative_candidates: SpeciesCandidate[];
  worms: WoRMSData;
  gbif: GBIFData;
  occurrences: OccurrenceRecord[];
  total_documented_occurrences: number;
  classifier_note: string;
  scientific_advice_note: string;
}

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

// Helper component to adjust Leaflet bounds to occurrences
function DistributionMapBounds({ occurrences }: { occurrences: OccurrenceRecord[] }) {
  const map = useMap();
  useEffect(() => {
    if (occurrences.length > 0) {
      const validPoints = occurrences.filter(
        (o) => !isNaN(o.latitude) && !isNaN(o.longitude)
      );
      if (validPoints.length > 0) {
        const bounds: [number, number][] = validPoints.map((o) => [o.latitude, o.longitude]);
        map.fitBounds(bounds, { padding: [40, 40], maxZoom: 8 });
      }
    }
  }, [occurrences, map]);
  return null;
}

export function ImageAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SeaweedAnalysisResult | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<SpeciesCandidate | null>(null);
  const [distributionLoading, setDistributionLoading] = useState(false);

  // Active biodiversity data (can switch if user selects an alternative candidate)
  const [activeWoRMS, setActiveWoRMS] = useState<WoRMSData>({});
  const [activeGBIF, setActiveGBIF] = useState<GBIFData>({});
  const [activeOccurrences, setActiveOccurrences] = useState<OccurrenceRecord[]>([]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      if (f.size > 15 * 1024 * 1024) {
        setError("File size exceeds 15 MB limit. Please select a smaller image.");
        setFile(null);
        setPreview(null);
        return;
      }
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
      setError(null);
      setSelectedCandidate(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("source_type", "underwater");

    try {
      const response = await fetch(`${API_URL}/analyze-image`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Analysis failed with HTTP ${response.status}`);
      }

      const data: SeaweedAnalysisResult = await response.json();
      setResult(data);
      if (data.top_candidate) {
        setSelectedCandidate(data.top_candidate);
        setActiveWoRMS(data.worms || {});
        setActiveGBIF(data.gbif || {});
        setActiveOccurrences(data.occurrences || []);
      }
    } catch (err: any) {
      setError(err.message || "An error occurred during seaweed identification.");
    } finally {
      setLoading(false);
    }
  };

  // Switch inspected candidate to an alternative species
  const handleSelectCandidate = async (candidate: SpeciesCandidate) => {
    setSelectedCandidate(candidate);
    setDistributionLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/species-distribution?scientific_name=${encodeURIComponent(candidate.scientific_name)}`
      );
      if (response.ok) {
        const data = await response.json();
        setActiveWoRMS(data.worms || {});
        setActiveGBIF(data.gbif || {});
        const occList = data.gbif?.occurrences_sample || [];
        setActiveOccurrences(occList);
      }
    } catch (err) {
      console.error("Failed to fetch alternative species distribution:", err);
    } finally {
      setDistributionLoading(false);
    }
  };

  const getGroupBadgeColor = (group: string) => {
    if (group.toLowerCase().includes("green")) return "#2e7d32";
    if (group.toLowerCase().includes("brown")) return "#8d6e63";
    if (group.toLowerCase().includes("red")) return "#c62828";
    return "#007791";
  };

  return (
    <div className="seaweed-analysis-container">
      {/* Header Banner */}
      <div className="seaweed-header-banner">
        <div className="seaweed-header-text">
          <h2>Seaweed Species Identification &amp; Global Distribution System</h2>
          <p className="seaweed-header-subtitle">
            Automated image classification across 53 marine macroalgae species (Brown, Green &amp; Red
            Algae) integrated with live WoRMS taxonomic validation and GBIF occurrence records.
          </p>
        </div>
      </div>

      <div className="seaweed-main-layout">
        {/* Left Column: Upload & Image */}
        <div className="seaweed-upload-card">
          <h3>1. Upload Seaweed Image</h3>
          <p className="seaweed-card-desc">
            Select a field photo, underwater observation, or herbarium specimen (JPEG, PNG, max 15 MB).
          </p>

          <div className="seaweed-dropzone">
            <input
              type="file"
              accept="image/png, image/jpeg, image/webp"
              id="seaweed-file-input"
              onChange={handleFileChange}
            />
            <label htmlFor="seaweed-file-input" className="seaweed-file-label">
              <span className="seaweed-upload-icon">📷</span>
              <span>{file ? file.name : "Choose an image or drop file here"}</span>
            </label>
          </div>

          {preview && (
            <div className="seaweed-preview-wrapper">
              <img src={preview} alt="Uploaded Seaweed" className="seaweed-preview-img" />
              <div className="seaweed-preview-meta">
                <span>{file ? `${(file.size / 1024).toFixed(1)} KB` : ""}</span>
              </div>
            </div>
          )}

          <button
            className="seaweed-analyze-btn"
            onClick={handleAnalyze}
            disabled={!file || loading}
          >
            {loading ? "Analyzing Seaweed Morphology…" : "Identify Seaweed Species"}
          </button>

          {error && <div className="seaweed-error-banner">{error}</div>}

          {/* Model info footnote */}
          <div className="seaweed-model-info">
            <strong>Model:</strong> EfficientNet-B0 transfer learning (53 macroalgal taxa)
            <br />
            <strong>Taxonomy:</strong> WoRMS (World Register of Marine Species)
            <br />
            <strong>Occurrences:</strong> GBIF (Global Biodiversity Information Facility)
          </div>
        </div>

        {/* Right Column: Identification Results & Distribution */}
        <div className="seaweed-results-area">
          {!result && !loading && (
            <div className="seaweed-empty-state">
              <div className="seaweed-empty-icon">🌿</div>
              <h4>No Image Analyzed Yet</h4>
              <p>Upload a seaweed photograph on the left and click "Identify Seaweed Species" to view predictions, taxonomic verification, and global occurrence mapping.</p>
            </div>
          )}

          {loading && (
            <div className="seaweed-loading-state">
              <div className="seaweed-spinner"></div>
              <h4>Processing Image &amp; Querying Biodiversity Repositories</h4>
              <p>Evaluating convolutional morphological features across 53 seaweed taxa and checking live WoRMS / GBIF databases…</p>
            </div>
          )}

          {result && (
            <div className="seaweed-results-container">
              {/* Uncertainty Warning */}
              {result.is_uncertain && (
                <div className="seaweed-uncertain-alert">
                  <strong>⚠️ Model Uncertainty Warning:</strong>
                  <p>{result.uncertainty_note}</p>
                </div>
              )}

              {/* Identification Candidates */}
              <div className="seaweed-card">
                <div className="seaweed-card-header">
                  <h3>Top Species Predictions</h3>
                  <span className="seaweed-model-tag">{result.model_name}</span>
                </div>

                <div className="seaweed-candidates-list">
                  {result.top_candidate && (
                    <div
                      className={`seaweed-candidate-card seaweed-candidate-card--top ${
                        selectedCandidate?.scientific_name === result.top_candidate.scientific_name
                          ? "seaweed-candidate-card--selected"
                          : ""
                      }`}
                      onClick={() => handleSelectCandidate(result.top_candidate!)}
                    >
                      <div className="seaweed-candidate-rank">#1 Primary Match</div>
                      <div className="seaweed-candidate-info">
                        <div className="seaweed-candidate-name-row">
                          <span className="seaweed-sci-name">{result.top_candidate.scientific_name}</span>
                          <span
                            className="seaweed-group-badge"
                            style={{
                              backgroundColor: getGroupBadgeColor(result.top_candidate.algae_group),
                            }}
                          >
                            {result.top_candidate.algae_group}
                          </span>
                        </div>
                        <div className="seaweed-common-name">{result.top_candidate.common_name}</div>
                        <div className="seaweed-confidence-bar-wrapper">
                          <div
                            className="seaweed-confidence-bar"
                            style={{ width: `${result.top_candidate.percentage}%` }}
                          />
                        </div>
                        <div className="seaweed-confidence-meta">
                          <span>Model Confidence Score: {result.top_candidate.percentage}%</span>
                          <span className="seaweed-click-hint">Click to view distribution</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Alternative Candidates */}
                  {result.alternative_candidates.length > 0 && (
                    <div className="seaweed-alternatives-grid">
                      <h4>Alternative Species Candidates:</h4>
                      <div className="seaweed-alternatives-list">
                        {result.alternative_candidates.map((cand, idx) => (
                          <div
                            key={idx}
                            className={`seaweed-alt-card ${
                              selectedCandidate?.scientific_name === cand.scientific_name
                                ? "seaweed-alt-card--selected"
                                : ""
                            }`}
                            onClick={() => handleSelectCandidate(cand)}
                          >
                            <div className="seaweed-alt-top">
                              <span className="seaweed-alt-rank">#{idx + 2}</span>
                              <span className="seaweed-alt-sci">{cand.scientific_name}</span>
                              <span className="seaweed-alt-score">{cand.percentage}%</span>
                            </div>
                            <div className="seaweed-alt-bottom">
                              <span>{cand.common_name}</span>
                              <span
                                className="seaweed-alt-badge"
                                style={{ backgroundColor: getGroupBadgeColor(cand.algae_group) }}
                              >
                                {cand.algae_group}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="seaweed-classifier-disclaimer">
                  <em>{result.classifier_note}</em>
                </div>
              </div>

              {/* Selected Species Information Card */}
              {selectedCandidate && (
                <div className="seaweed-card">
                  <div className="seaweed-card-header">
                    <h3>
                      Species Information &amp; Taxonomy: <em>{selectedCandidate.scientific_name}</em>
                    </h3>
                    <span
                      className="seaweed-group-badge"
                      style={{ backgroundColor: getGroupBadgeColor(selectedCandidate.algae_group) }}
                    >
                      {selectedCandidate.algae_group}
                    </span>
                  </div>

                  <div className="seaweed-bio-grid">
                    <div className="seaweed-bio-item">
                      <strong>Common Name:</strong>
                      <p>{selectedCandidate.common_name}</p>
                    </div>
                    <div className="seaweed-bio-item">
                      <strong>Algal Division / Phylum:</strong>
                      <p>{selectedCandidate.phylum}</p>
                    </div>
                    <div className="seaweed-bio-item">
                      <strong>Ecological Habitat:</strong>
                      <p>{selectedCandidate.habitat_summary}</p>
                    </div>
                    <div className="seaweed-bio-item">
                      <strong>Known Natural Range:</strong>
                      <p>{selectedCandidate.known_range}</p>
                    </div>
                  </div>

                  {/* WoRMS & GBIF Links */}
                  <div className="seaweed-taxonomy-section">
                    <h4>Taxonomic Verification &amp; Biodiversity Databases:</h4>
                    <div className="seaweed-taxonomy-boxes">
                      <div className="seaweed-db-box">
                        <div className="seaweed-db-title">WoRMS (MarineSpecies.org)</div>
                        {activeWoRMS.aphia_id ? (
                          <>
                            <div className="seaweed-db-detail">
                              <strong>AphiaID:</strong> {activeWoRMS.aphia_id} (Status: {activeWoRMS.taxonomic_status || "accepted"})
                            </div>
                            <div className="seaweed-db-detail">
                              <strong>Authority:</strong> {activeWoRMS.authority || "N/A"}
                            </div>
                            {activeWoRMS.taxonomy && (
                              <div className="seaweed-db-detail seaweed-tax-crumb">
                                {activeWoRMS.taxonomy.phylum} &gt; {activeWoRMS.taxonomy.class} &gt;{" "}
                                {activeWoRMS.taxonomy.order} &gt; {activeWoRMS.taxonomy.family} &gt;{" "}
                                {activeWoRMS.taxonomy.genus}
                              </div>
                            )}
                            <a
                              href={activeWoRMS.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="seaweed-db-link"
                            >
                              View Official WoRMS Species Profile ↗
                            </a>
                          </>
                        ) : (
                          <p className="seaweed-db-unavailable">Information unavailable from WoRMS</p>
                        )}
                      </div>

                      <div className="seaweed-db-box">
                        <div className="seaweed-db-title">GBIF Backbone Taxonomy</div>
                        {activeGBIF.taxon_key ? (
                          <>
                            <div className="seaweed-db-detail">
                              <strong>GBIF Taxon Key:</strong> {activeGBIF.taxon_key}
                            </div>
                            <div className="seaweed-db-detail">
                              <strong>Documented Occurrences:</strong>{" "}
                              {(activeGBIF.total_occurrences_documented || 0).toLocaleString()} records
                            </div>
                            <a
                              href={activeGBIF.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="seaweed-db-link"
                            >
                              Explore GBIF Global Dataset ↗
                            </a>
                          </>
                        ) : (
                          <p className="seaweed-db-unavailable">Information unavailable from GBIF</p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Interactive Global Distribution Map */}
              {selectedCandidate && (
                <div className="seaweed-card">
                  <div className="seaweed-card-header">
                    <div>
                      <h3>
                        Global Distribution Map: <em>{selectedCandidate.scientific_name}</em>
                      </h3>
                      <p className="seaweed-map-subtitle">
                        Plotting genuine documented observation coordinates retrieved from GBIF.
                      </p>
                    </div>
                    <div className="seaweed-occ-badge">
                      {(activeGBIF.total_occurrences_documented || activeOccurrences.length).toLocaleString()}{" "}
                      Documented Records
                    </div>
                  </div>

                  {distributionLoading ? (
                    <div className="seaweed-map-loading">Loading distribution coordinates…</div>
                  ) : activeOccurrences.length > 0 ? (
                    <div className="seaweed-leaflet-wrapper">
                      <MapContainer
                        center={[20, 0]}
                        zoom={2}
                        scrollWheelZoom={true}
                        style={{ height: "380px", width: "100%", borderRadius: "6px" }}
                      >
                        <TileLayer
                          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        />
                        <DistributionMapBounds occurrences={activeOccurrences} />

                        {activeOccurrences.map((occ, idx) => (
                          <CircleMarker
                            key={idx}
                            center={[occ.latitude, occ.longitude]}
                            radius={5}
                            pathOptions={{
                              color: getGroupBadgeColor(selectedCandidate.algae_group),
                              fillColor: getGroupBadgeColor(selectedCandidate.algae_group),
                              fillOpacity: 0.7,
                              weight: 1.5,
                            }}
                          >
                            <Popup>
                              <div className="seaweed-map-popup">
                                <strong><em>{selectedCandidate.scientific_name}</em></strong>
                                <p><strong>Location:</strong> {occ.country}</p>
                                <p><strong>Date:</strong> {occ.event_date}</p>
                                <p><strong>Coordinates:</strong> {occ.latitude.toFixed(4)}, {occ.longitude.toFixed(4)}</p>
                                <p><strong>Record Type:</strong> {occ.basis_of_record}</p>
                              </div>
                            </Popup>
                          </CircleMarker>
                        ))}
                      </MapContainer>
                    </div>
                  ) : (
                    <div className="seaweed-no-map-alert">
                      <p>No georeferenced coordinate records currently available for this species in the GBIF query.</p>
                    </div>
                  )}

                  <div className="seaweed-map-disclaimer">
                    <p>
                      <strong>Scientific Data Notice:</strong> Documented occurrence records represent verified
                      museum, herbarium, and citizen-science sightings. Missing dots in an oceanic basin do not prove
                      absence of the species. Documented records are distinct from the full theoretical natural range.
                    </p>
                  </div>
                </div>
              )}

              {/* Taxonomic Confirmation Notice */}
              <div className="seaweed-advisory-box">
                <strong>Taxonomic Advice:</strong> {result.scientific_advice_note}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
