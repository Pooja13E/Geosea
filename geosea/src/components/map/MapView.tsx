import { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Rectangle, Tooltip, useMap, useMapEvents } from "react-leaflet";
import type {
  BaseMapStyle,
  EnvGridCell,
  GridCell,
  HabitatPrediction,
  InteractionMode,
  MapLocation,
  MapMode,
  OccurrenceRecord,
} from "@/types/environmental";
import { MapLegend } from "./MapLegend";
import { LocationInfoPanel } from "./LocationInfoPanel";
import { ENV_VARIABLE_META, envColorScale, isEnvironmentalMode } from "@/data/envVariableConfig";
import {
  getEnvironmentalGridAggregation,
  getGridAggregation,
  getIndicesInBounds,
  getSummaryForIndices,
  getRecordAt,
} from "@/services/dataService";
import { getPrediction } from "@/services/predictionService";
import { FINAL_MODEL } from "@/services/modelConfig";
import { formatCoordinate } from "@/utils/formatters";
import "./MapView.css";

const TILE_LAYERS: Record<BaseMapStyle, { url: string; attribution: string }> = {
  standard: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri — Source: Esri, HERE, Garmin, USGS, NGA, EPA, NPS",
  },
  satellite: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics",
  },
};

/** Above this many visible records, switch from individual markers to representative points. */
const POINT_RENDER_THRESHOLD = 4000;
/** Hard cap on individual markers rendered even below the threshold, as a safety net. */
const MAX_INDIVIDUAL_MARKERS = 6000;

/**
 * FIXED screen-space radius (px) for every occurrence point — individual
 * observations AND representative/aggregated points alike. This is the
 * "Google Maps, not grid cells" requirement: point size never changes with
 * zoom and never encodes record count. Aggregation controls how MANY points
 * are drawn (fewer, spatially-sampled points at low zoom; more, eventually
 * individual, points as the user zooms in) — never how big any one point is.
 */
const OCCURRENCE_POINT_RADIUS = 4;

/** Presence/absence colors shared by both individual and representative occurrence points. */
const PRESENCE_COLOR = { stroke: "#1c6e63", fill: "#278f80" };
const ABSENCE_COLOR = { stroke: "#7c8e93", fill: "#a7b3b7" };

export interface MapBounds {
  south: number;
  north: number;
  west: number;
  east: number;
}

interface MapViewProps {
  filteredIndices: Int32Array;
  mapMode: MapMode;
  /** [min, max] used to color environmental cells; computed by the caller from the current filters. */
  envDomain: [number, number] | null;
  dataBounds: MapBounds;
  baseMapStyle: BaseMapStyle;
  fitSignal: number;
  /** "browse" reads observed records on click (existing behavior); "predict" runs a habitat suitability lookup instead. */
  interactionMode: InteractionMode;
}

function cellSizeForZoom(zoom: number): number {
  if (zoom <= 2) return 10;
  if (zoom <= 4) return 5;
  if (zoom <= 6) return 2;
  if (zoom <= 8) return 1;
  if (zoom <= 10) return 0.5;
  return 0.25;
}

/**
 * Cell size (in degrees) used specifically for the environmental grid.
 * Deliberately coarser than cellSizeForZoom() at low zoom: environmental
 * layers cover the whole viewport (unlike occurrence points, which thin out
 * naturally), so without extra coarsening a global view still produces
 * hundreds of adjacent cells.
 */
function envCellSizeForZoom(zoom: number): number {
  if (zoom <= 2) return 12;
  if (zoom <= 4) return 6;
  if (zoom <= 6) return 3;
  if (zoom <= 8) return 1;
  if (zoom <= 10) return 0.5;
  return 0.25;
}

interface AggregatedCellStyle {
  /** Fraction of the cell shrunk on each side, so adjacent cells no longer touch. */
  inset: number;
  weight: number;
  fillOpacity: number;
}

/**
 * At low zoom, a fully-tiled grid of bordered rectangles reads as a solid
 * "wall" no matter how big or small the individual cells are — the density
 * comes from every edge being drawn, not from cell size. Insetting each
 * cell (and dropping the border) turns the same grid into a clean,
 * legible pattern of spatially-aggregated values. As the user zooms in,
 * the inset and border are relaxed until, at high zoom, cells render
 * exactly as they did before (fully tiled, thin border) for the detailed view.
 *
 * Used only by the environmental grid — the occurrence layer never renders
 * rectangles (see the fixed-radius CircleMarker rendering further down);
 * this style only applies to the environmental Rectangle overlay.
 */
function aggregatedCellStyleForZoom(zoom: number): AggregatedCellStyle {
  if (zoom <= 3) return { inset: 0.22, weight: 0, fillOpacity: 0.85 };
  if (zoom <= 5) return { inset: 0.14, weight: 0.5, fillOpacity: 0.82 };
  if (zoom <= 6) return { inset: 0.06, weight: 0.75, fillOpacity: 0.82 };
  return { inset: 0, weight: 1, fillOpacity: 0.82 };
}

function insetBounds(bounds: MapBounds, ratio: number): MapBounds {
  if (ratio <= 0) return bounds;
  const latPad = (bounds.north - bounds.south) * ratio;
  const lngPad = (bounds.east - bounds.west) * ratio;
  return {
    south: bounds.south + latPad,
    north: bounds.north - latPad,
    west: bounds.west + lngPad,
    east: bounds.east - lngPad,
  };
}

/** Fits the map to `bounds` whenever the bounds values or `signal` change. */
function FitBoundsController({ bounds, signal }: { bounds: MapBounds; signal: number }) {
  const map = useMap();
  useEffect(() => {
    if (!Number.isFinite(bounds.south) || !Number.isFinite(bounds.north)) return;
    const padded: [[number, number], [number, number]] = [
      [bounds.south, bounds.west],
      [bounds.north, bounds.east],
    ];
    map.fitBounds(padded, { padding: [24, 24] });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bounds.south, bounds.north, bounds.west, bounds.east, signal]);
  return null;
}

function stopMapPropagation(event: { originalEvent?: Event }) {
  event.originalEvent?.stopPropagation();
}

/** In prediction mode, a plain map click (not on a marker/cell) picks a location. */
function MapClickController({ onClick }: { onClick: (lat: number, lng: number) => void }) {
  useMapEvents({
    click: (e) => onClick(e.latlng.lat, e.latlng.lng),
  });
  return null;
}

/** Adds explicit pan controls so users can move the map vertically/horizontally
 * without relying only on mouse dragging.
 */
function MapPanControls() {
  const map = useMap();
  const panBy = (dx: number, dy: number) => {
    const size = map.getSize();
    map.panBy([dx * size.x * 0.35, dy * size.y * 0.35], { animate: true, duration: 0.25 });
  };

  return (
    <div className="gs-map__pan-controls" aria-label="Map pan controls">
      <button type="button" onClick={() => panBy(0, -1)} aria-label="Pan map up">↑</button>
      <div className="gs-map__pan-row">
        <button type="button" onClick={() => panBy(-1, 0)} aria-label="Pan map left">←</button>
        <button type="button" onClick={() => panBy(1, 0)} aria-label="Pan map right">→</button>
      </div>
      <button type="button" onClick={() => panBy(0, 1)} aria-label="Pan map down">↓</button>
    </div>
  );
}

/** Tracks the current viewport bounds/zoom so rendering can restrict to what's visible. */
function ViewportController({ onChange }: { onChange: (bounds: MapBounds, zoom: number) => void }) {
  const map = useMapEvents({
    moveend: () => {
      const b = map.getBounds();
      onChange(
        { south: b.getSouth(), north: b.getNorth(), west: b.getWest(), east: b.getEast() },
        map.getZoom()
      );
    },
  });

  useEffect(() => {
    const b = map.getBounds();
    onChange(
      { south: b.getSouth(), north: b.getNorth(), west: b.getWest(), east: b.getEast() },
      map.getZoom()
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return null;
}

function DensityPanel({ cell, filteredIndices, onClose }: { cell: GridCell; filteredIndices: Int32Array; onClose: () => void }) {
  const localIndices = getIndicesInBounds(filteredIndices, cell.bounds);
  const summary = getSummaryForIndices(localIndices);

  return (
    <div className="gs-density-panel" role="dialog" aria-label="Occurrence density details">
      <div className="gs-density-panel__header">
        <div>
          <p className="gs-info-panel__eyebrow">Occurrence Density</p>
          <h3>Local observation coverage</h3>
        </div>
        <button type="button" className="gs-info-panel__close" onClick={onClose} aria-label="Close density details">×</button>
      </div>
      <p className="gs-info-panel__coords">{formatCoordinate(cell.representativeLat)}, {formatCoordinate(cell.representativeLng)}</p>
      <div className="gs-density-panel__hero">
        <strong>{cell.count.toLocaleString()}</strong>
        <span>records in this area</span>
      </div>
      <div className="gs-density-panel__grid">
        <div><span>Presence</span><strong>{cell.presenceCount.toLocaleString()}</strong></div>
        <div><span>Pseudo-absence</span><strong>{cell.absenceCount.toLocaleString()}</strong></div>
        <div><span>Species</span><strong>{summary.speciesCount.toLocaleString()}</strong></div>
      </div>
      <p className="gs-density-panel__note">Density is calculated from the currently active filters and observed GeoSea records.</p>
    </div>
  );
}

export function MapView({
  filteredIndices,
  mapMode,
  envDomain,
  dataBounds,
  baseMapStyle,
  fitSignal,
  interactionMode,
}: MapViewProps) {
  const [viewport, setViewport] = useState<{ bounds: MapBounds; zoom: number } | null>(null);
  const [selectedRecord, setSelectedRecord] = useState<OccurrenceRecord | null>(null);
  const [fitToCell, setFitToCell] = useState<{ bounds: MapBounds; signal: number } | null>(null);

  const [predictedLocation, setPredictedLocation] = useState<MapLocation | null>(null);
  const [predictionStatus, setPredictionStatus] = useState<"loading" | "loaded">("loaded");
  const [prediction, setPrediction] = useState<HabitatPrediction | null>(null);
  const predictionRequestRef = useRef(0);
  const predictionAbortRef = useRef<AbortController | null>(null);
  const [selectedDensityCell, setSelectedDensityCell] = useState<GridCell | null>(null);

  function handlePredictClick(lat: number, lng: number) {
    const location: MapLocation = { latitude: lat, longitude: lng };
    const requestId = ++predictionRequestRef.current;
    predictionAbortRef.current?.abort();
    const controller = new AbortController();
    predictionAbortRef.current = controller;

    setPredictedLocation(location);
    setPrediction(null);
    setPredictionStatus("loading");
    setSelectedRecord(null);

    getPrediction(location, controller.signal)
      .then((result) => {
        if (requestId !== predictionRequestRef.current) return;
        setPrediction(result);
        setPredictionStatus("loaded");
      })
      .catch((error) => {
        if (requestId !== predictionRequestRef.current) return;
        if (error instanceof DOMException && error.name === "AbortError") return;
        setPredictionStatus("loaded");
        setPrediction({
          modelId: FINAL_MODEL,
          location,
          probability: null,
          suitabilityClass: "unknown",
          confidence: null,
          environmentalValues: null,
          environmentalSource: null,
          isLive: false,
          errorMessage: error instanceof Error ? error.message : "Prediction service unavailable",
          outOfRangeVariables: [],
        });
      });
  }

  // Keep only one active analysis panel at a time. Prediction mode uses the
  // live Bio-ORACLE/Random Forest path; density mode uses the loaded occurrence data.
  useEffect(() => {
    setSelectedRecord(null);
    setSelectedDensityCell(null);
    if (interactionMode !== "predict") {
      predictionAbortRef.current?.abort();
      predictionAbortRef.current = null;
      predictionRequestRef.current += 1;
      setPredictedLocation(null);
      setPrediction(null);
      setPredictionStatus("loaded");
    }
  }, [interactionMode]);

  useEffect(() => () => {
    predictionAbortRef.current?.abort();
  }, []);

  const tiles = TILE_LAYERS[baseMapStyle];
  const envMode = isEnvironmentalMode(mapMode);
  const densityMode = interactionMode === "density";

  const visibleIndices = useMemo(() => {
    if (!viewport) return filteredIndices;
    return getIndicesInBounds(filteredIndices, viewport.bounds);
  }, [filteredIndices, viewport]);

  const shouldAggregate = !envMode && !densityMode && visibleIndices.length > POINT_RENDER_THRESHOLD;

  const gridCells: GridCell[] = useMemo(() => {
    if (envMode || densityMode || !shouldAggregate) return [];
    const cellSize = cellSizeForZoom(viewport?.zoom ?? 2);
    return getGridAggregation(visibleIndices, cellSize);
  }, [envMode, shouldAggregate, visibleIndices, viewport?.zoom]);

  const densityCells: GridCell[] = useMemo(() => {
    if (!densityMode) return [];
    const cellSize = cellSizeForZoom(viewport?.zoom ?? 2);
    return getGridAggregation(visibleIndices, cellSize);
  }, [densityMode, visibleIndices, viewport?.zoom]);

  const densityMax = useMemo(() => {
    let max = 1;
    for (const cell of densityCells) max = Math.max(max, cell.count);
    return max;
  }, [densityCells]);

  // Environmental mode always aggregates into a grid (never one marker per
  // record) so the map stays responsive regardless of how many observations
  // are visible.
  const envGridCells: EnvGridCell[] = useMemo(() => {
    if (!envMode) return [];
    const cellSize = envCellSizeForZoom(viewport?.zoom ?? 2);
    return getEnvironmentalGridAggregation(visibleIndices, cellSize, mapMode);
  }, [envMode, visibleIndices, viewport?.zoom, mapMode]);

  const envCellStyle = useMemo(() => aggregatedCellStyleForZoom(viewport?.zoom ?? 2), [viewport?.zoom]);

  const pointRecords = useMemo(() => {
    if (envMode || densityMode || shouldAggregate) return [];
    const cap = Math.min(visibleIndices.length, MAX_INDIVIDUAL_MARKERS);
    const records: OccurrenceRecord[] = [];
    for (let i = 0; i < cap; i++) records.push(getRecordAt(visibleIndices[i]));
    return records;
  }, [envMode, shouldAggregate, visibleIndices]);

  // Selected record can go stale if filters change underneath it; clear it then.
  useEffect(() => {
    setSelectedRecord(null);
    setSelectedDensityCell(null);
  }, [filteredIndices]);

  // Individual-record selection only makes sense in occurrence mode.
  useEffect(() => {
    if (envMode) setSelectedRecord(null);
  }, [envMode]);

  const envMeta = envMode ? ENV_VARIABLE_META[mapMode] : null;
  // Fall back to the visible cells' own range if no filtered-dataset domain
  // is available yet (e.g. right after switching variables).
  const fallbackDomain = useMemo<[number, number]>(() => {
    if (envGridCells.length === 0) return [0, 1];
    let min = Infinity;
    let max = -Infinity;
    for (const cell of envGridCells) {
      if (cell.mean < min) min = cell.mean;
      if (cell.mean > max) max = cell.mean;
    }
    return [min, max];
  }, [envGridCells]);
  const colorDomain = envDomain ?? fallbackDomain;
  const domainSpan = colorDomain[1] - colorDomain[0] || 1;

  return (
    <div className="gs-map">
      <div className="gs-map__mode-badge">
        <p className="gs-map__mode-title">{
          densityMode ? "Occurrence Density" :
          envMode ? envMeta!.label : "Observed Occurrence Distribution"
        }</p>
        <p className="gs-map__mode-subtitle">
          {densityMode ? "Spatial concentration of the loaded occurrence records" :
           envMode ? "Observed values — spatially aggregated observations" : "Presence / absence observations"}
        </p>
        {interactionMode === "predict" && (
          <p className="gs-map__mode-subtitle gs-map__mode-subtitle--predict">Prediction mode — click the map to evaluate a location</p>
        )}
      </div>

      <MapContainer
        center={[(dataBounds.south + dataBounds.north) / 2, (dataBounds.west + dataBounds.east) / 2]}
        zoom={3}
        className="gs-map__container"
        zoomControl={true}
        preferCanvas
        dragging={true}
        scrollWheelZoom={true}
        doubleClickZoom={true}
        touchZoom={true}
        boxZoom={true}
        keyboard={true}
      >
        <TileLayer url={tiles.url} attribution={tiles.attribution} />
        <MapPanControls />
        <FitBoundsController bounds={dataBounds} signal={fitSignal} />
        {fitToCell && <FitBoundsController bounds={fitToCell.bounds} signal={fitToCell.signal} />}
        <ViewportController onChange={(bounds, zoom) => setViewport({ bounds, zoom })} />
        {interactionMode === "predict" && <MapClickController onClick={handlePredictClick} />}

        {envMode &&
          envMeta &&
          envGridCells.map((cell) => {
            const t = (cell.mean - colorDomain[0]) / domainSpan;
            const fill = envColorScale(mapMode, t);
            const renderBounds = insetBounds(cell.bounds, envCellStyle.inset);
            return (
              <Rectangle
                key={cell.id}
                bounds={[
                  [renderBounds.south, renderBounds.west],
                  [renderBounds.north, renderBounds.east],
                ]}
                pathOptions={{
                  color: "rgba(16, 38, 46, 0.25)",
                  weight: envCellStyle.weight,
                  fillColor: fill,
                  fillOpacity: envCellStyle.fillOpacity,
                }}
                eventHandlers={{
                  click: (event) => {
                    stopMapPropagation(event);
                    if (interactionMode === "predict") {
                      handlePredictClick(cell.latCenter, cell.lngCenter);
                    } else {
                      // Even when observations are aggregated, clicking a cell
                      // should immediately show the actual observation behind it.
                      setSelectedRecord(getRecordAt(cell.representativeIndex));
                    }
                  },
                }}
              >
                <Tooltip direction="top" offset={[0, -4]}>
                  {envMeta.label}: {cell.mean.toFixed(envMeta.decimals)}
                  {envMeta.unit ? ` ${envMeta.unit}` : ""} (mean)
                  <br />
                  Range {cell.min.toFixed(envMeta.decimals)} – {cell.max.toFixed(envMeta.decimals)}
                  <br />
                  {cell.count.toLocaleString()} observations
                  <br />
                  <em>Click for observation details</em>
                </Tooltip>
              </Rectangle>
            );
          })}

        {densityMode &&
          densityCells.map((cell) => {
            const ratio = Math.log1p(cell.count) / Math.log1p(densityMax);
            const radius = 5 + Math.round(ratio * 11);
            const fill = ratio >= 0.72 ? "#c96a4e" : ratio >= 0.42 ? "#e0a85c" : "#278f80";
            return (
              <CircleMarker
                key={`density-${cell.id}`}
                center={[cell.representativeLat, cell.representativeLng]}
                radius={radius}
                pathOptions={{ color: fill, fillColor: fill, fillOpacity: 0.42, weight: 1.5 }}
                eventHandlers={{
                  click: (event) => {
                    stopMapPropagation(event);
                    setSelectedRecord(null);
                    setSelectedDensityCell(cell);
                  },
                }}
              >
                <Tooltip direction="top" offset={[0, -6]}>
                  <strong>{cell.count.toLocaleString()} records</strong><br />
                  Presence: {cell.presenceCount.toLocaleString()}<br />
                  Pseudo-absence: {cell.absenceCount.toLocaleString()}<br />
                  <em>Click for local details</em>
                </Tooltip>
              </CircleMarker>
            );
          })}

        {/*
          Representative occurrence points: a spatial SAMPLE of the visible
          records, drawn at fixed screen-space radius (OCCURRENCE_POINT_RADIUS)
          — never a rectangle, never sized by count. Zoom controls how many
          of these points exist (smaller bins -> more points as the user
          zooms in), not how big any single point is. Each point sits at the
          real coordinate of an actual observation in its bin.
        */}
        {!envMode &&
          shouldAggregate &&
          gridCells.flatMap((cell) => {
            const markers = [
              {
                key: `${cell.id}:primary`,
                lat: cell.representativeLat,
                lng: cell.representativeLng,
                index: cell.representativeIndex,
                presence: cell.presence,
              },
            ];
            if (cell.secondaryRepresentativeIndex !== null && cell.secondaryRepresentativeLat !== null && cell.secondaryRepresentativeLng !== null) {
              markers.push({
                key: `${cell.id}:secondary`,
                lat: cell.secondaryRepresentativeLat,
                lng: cell.secondaryRepresentativeLng,
                index: cell.secondaryRepresentativeIndex,
                presence: cell.presence === 1 ? 0 : 1,
              });
            }
            return markers.map((marker) => {
              const color = marker.presence ? PRESENCE_COLOR : ABSENCE_COLOR;
              const classLabel = marker.presence ? "Presence" : "Absence";
              return (
                <CircleMarker
                  key={marker.key}
                  center={[marker.lat, marker.lng]}
                  radius={OCCURRENCE_POINT_RADIUS}
                  pathOptions={{
                    color: color.stroke,
                    fillColor: color.fill,
                    fillOpacity: 0.9,
                    weight: 1.5,
                  }}
                  eventHandlers={{
                    click: (event) => {
                      stopMapPropagation(event);
                      if (interactionMode === "predict") {
                        handlePredictClick(marker.lat, marker.lng);
                      } else {
                        setSelectedRecord(getRecordAt(marker.index));
                      }
                    },
                  }}
                >
                  <Tooltip direction="top" offset={[0, -6]}>
                    <strong>{classLabel}</strong> observation
                    <br />
                    Records represented in area: {cell.count.toLocaleString()}
                    <br />
                    Presence: {cell.presenceCount.toLocaleString()}
                    <br />
                    Absence: {cell.absenceCount.toLocaleString()}
                    <br />
                    <em>Click for observation details</em>
                  </Tooltip>
                </CircleMarker>
              );
            });
          })}

        {/* Individual observations, revealed once zoom brings the visible count below the aggregation threshold. Same fixed radius as the representative points above. */}
        {!envMode &&
          !shouldAggregate &&
          pointRecords.map((record) => (
            <CircleMarker
              key={record.index}
              center={[record.latitude, record.longitude]}
              radius={OCCURRENCE_POINT_RADIUS}
              pathOptions={{
                color: record.presence ? PRESENCE_COLOR.stroke : ABSENCE_COLOR.stroke,
                fillColor: record.presence ? PRESENCE_COLOR.fill : ABSENCE_COLOR.fill,
                fillOpacity: 0.85,
                weight: 1.5,
              }}
              eventHandlers={{
                click: (event) => {
                  stopMapPropagation(event);
                  if (interactionMode === "predict") {
                    handlePredictClick(record.latitude, record.longitude);
                  } else {
                    setSelectedRecord(record);
                  }
                },
              }}
            >
              <Tooltip direction="top" offset={[0, -6]}>
                {formatCoordinate(record.latitude)}, {formatCoordinate(record.longitude)}
                <br />
                {record.presence ? "Presence" : "Absence"}
                {!record.isPseudoAbsence && (
                  <>
                    <br />
                    <em>{record.scientific_name}</em>
                  </>
                )}
              </Tooltip>
            </CircleMarker>
          ))}

        {interactionMode === "predict" && predictedLocation && (
          <CircleMarker
            center={[predictedLocation.latitude, predictedLocation.longitude]}
            radius={9}
            pathOptions={{
              color: "#e0a85c",
              fillColor: "#e0a85c",
              fillOpacity: 0.35,
              weight: 2.5,
            }}
          />
        )}
      </MapContainer>

      {!densityMode && (
        <MapLegend
          mode={mapMode}
          aggregated={shouldAggregate}
          visibleCount={visibleIndices.length}
          envDomain={envMode ? colorDomain : null}
        />
      )}

      {densityMode && (
        <div className="gs-density-legend">
          <p>Occurrence density</p>
          <span><i className="gs-density-legend__dot gs-density-legend__dot--low" /> Lower concentration</span>
          <span><i className="gs-density-legend__dot gs-density-legend__dot--mid" /> Medium concentration</span>
          <span><i className="gs-density-legend__dot gs-density-legend__dot--high" /> Higher concentration</span>
        </div>
      )}

      {selectedRecord && (
        <LocationInfoPanel kind="record" record={selectedRecord} onClose={() => setSelectedRecord(null)} />
      )}

      {densityMode && selectedDensityCell && (
        <DensityPanel
          cell={selectedDensityCell}
          filteredIndices={filteredIndices}
          onClose={() => setSelectedDensityCell(null)}
        />
      )}

      {interactionMode === "predict" && predictedLocation && (
        <LocationInfoPanel
          kind="prediction"
          location={predictedLocation}
          status={predictionStatus}
          prediction={prediction}
          onClose={() => {
            setPredictedLocation(null);
            setPrediction(null);
          }}
        />
      )}
    </div>
  );
}
