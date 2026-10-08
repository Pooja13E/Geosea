import { useEffect, useMemo, useState } from "react";
import { Header } from "@/components/layout/Header";
import { StatsBanner } from "@/components/layout/StatsBanner";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { MapView } from "@/components/map/MapView";
import {
  getAllIndices,
  getEnvironmentalStats,
  getFilteredRecords,
  loadEnvironmentalData,
} from "@/services/dataService";
import { isEnvironmentalMode } from "@/data/envVariableConfig";
import { DEFAULT_FILTERS } from "@/types/environmental";
import type {
  BaseMapStyle,
  DatasetSummary,
  EnvironmentalStats,
  FilterState,
  InteractionMode,
  MapMode,
} from "@/types/environmental";
import "./App.css";

type LoadStatus = "loading" | "ready" | "error";

const INDIA_STUDY_BOUNDS = {
  south: 4,
  north: 26,
  west: 67,
  east: 98,
};


function App() {
  const [status, setStatus] = useState<LoadStatus>("loading");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [fullSummary, setFullSummary] = useState<DatasetSummary | null>(null);

  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);
  const [mapMode, setMapMode] = useState<MapMode>("occurrences");
  const [baseMapStyle, setBaseMapStyle] = useState<BaseMapStyle>("standard");
  const [fitSignal, setFitSignal] = useState(0);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("browse");

  useEffect(() => {
    let cancelled = false;
    loadEnvironmentalData()
      .then((summary) => {
        if (cancelled) return;
        setFullSummary(summary);
        setStatus("ready");
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setErrorMessage(err.message);
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredIndices = useMemo(() => {
    if (status !== "ready") return new Int32Array(0);
    return getFilteredRecords(filters);
  }, [status, filters]);


  // Reset any active environmental range filter whenever the map mode
  // changes — a range that made sense for the previous variable (or for no
  // variable at all, in Occurrence mode) is not valid for the new one.
  useEffect(() => {
    setFilters((prev) => (prev.envRange ? { ...prev, envRange: null } : prev));
  }, [mapMode]);

  const envStatsFiltered: EnvironmentalStats | null = useMemo(() => {
    if (status !== "ready" || !isEnvironmentalMode(mapMode)) return null;
    return getEnvironmentalStats(filteredIndices, mapMode);
  }, [status, mapMode, filteredIndices]);

  const envStatsGlobal: EnvironmentalStats | null = useMemo(() => {
    if (status !== "ready" || !isEnvironmentalMode(mapMode)) return null;
    // Whole-dataset stats never change once loaded, so recompute only when
    // the selected variable changes, not on every filter tweak.
    return getEnvironmentalStats(getAllIndices(), mapMode);
  }, [status, mapMode]);

  // GeoSea is an India-coast decision-support dashboard. The underlying
  // marine dataset spans a much larger oceanic region, so the map should
  // open on the intended Indian coastal study area instead of fitting to
  // the full dataset extent.
  const dataBounds = INDIA_STUDY_BOUNDS;

  // Re-fit the map whenever the active filters change the data extent.
  useEffect(() => {
    setFitSignal((n) => n + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    filters.species,
    filters.presence,
    filters.fold,
    filters.envRange?.variable,
    filters.envRange?.min,
    filters.envRange?.max,
  ]);

  function handleFilterChange(patch: Partial<FilterState>) {
    setFilters((prev) => ({ ...prev, ...patch }));
  }

  if (status === "error") {
    return (
      <div className="gs-app gs-app--center">
        <div className="gs-status-card gs-status-card--error">
          <h2>Couldn't load the GeoSea dataset</h2>
          <p>{errorMessage}</p>
          <p className="gs-status-card__hint">
            Check that <code>public/data/GeoSea_8_Variables.csv</code> exists and that its columns match
            the expected schema, then reload.
          </p>
        </div>
      </div>
    );
  }

  if (status === "loading" || !fullSummary) {
    return (
      <div className="gs-app gs-app--center">
        <div className="gs-status-card">
          <div className="gs-status-card__spinner" aria-hidden="true" />
          <h2>Loading GeoSea dataset…</h2>
          <p>Parsing 336,924 records — this runs once and stays in memory for the session.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="gs-app">
      <Header />
      <StatsBanner summary={fullSummary} />
      <div className="gs-app__body">
        <Sidebar
          summary={fullSummary}
          filters={filters}
          onFilterChange={handleFilterChange}
          mapMode={mapMode}
          onChangeMapMode={setMapMode}
          envStatsGlobal={envStatsGlobal}
          baseMapStyle={baseMapStyle}
          onChangeBaseMapStyle={setBaseMapStyle}
          onResetMap={() => setFitSignal((n) => n + 1)}
          interactionMode={interactionMode}
          onChangeInteractionMode={setInteractionMode}
        />
        <MapView
          filteredIndices={filteredIndices}
          mapMode={mapMode}
          envDomain={
            envStatsFiltered && envStatsFiltered.count > 0
              ? [envStatsFiltered.min, envStatsFiltered.max]
              : null
          }
          dataBounds={dataBounds}
          baseMapStyle={baseMapStyle}
          fitSignal={fitSignal}
          interactionMode={interactionMode}
        />
      </div>
    </div>
  );
}

export default App;
