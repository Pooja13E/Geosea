import { useEffect, useMemo, useState } from "react";
import type {
  BaseMapStyle,
  DatasetSummary,
  EnvironmentalStats,
  SpeciesProfile,
  FilterState,
  InteractionMode,
  MapMode,
  PresenceFilter,
} from "@/types/environmental";
import { ENV_VARIABLE_KEYS } from "@/types/environmental";
import { ENV_VARIABLE_META, isEnvironmentalMode } from "@/data/envVariableConfig";
import { formatCount } from "@/utils/formatters";
import { getSpeciesProfile } from "@/services/dataService";
import { SpeciesSelect } from "./SpeciesSelect";
import "./Sidebar.css";

interface SidebarProps {
  summary: DatasetSummary;
  filters: FilterState;
  onFilterChange: (patch: Partial<FilterState>) => void;
  mapMode: MapMode;
  onChangeMapMode: (mode: MapMode) => void;
  envStatsGlobal: EnvironmentalStats | null;
  baseMapStyle: BaseMapStyle;
  onChangeBaseMapStyle: (style: BaseMapStyle) => void;
  onResetMap: () => void;
  interactionMode: InteractionMode;
  onChangeInteractionMode: (mode: InteractionMode) => void;
}

const PRESENCE_OPTIONS: Array<{ value: PresenceFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "presence", label: "Presence" },
  { value: "absence", label: "Absence" },
];

const hasActiveFilters = (f: FilterState) =>
  f.species !== "all" || f.presence !== "all" || f.fold !== "all" || f.envRange !== null;

/**
 * The three top-level Map Layer choices shown in the sidebar. This is a
 * presentation-only grouping over the existing (mapMode, interactionMode)
 * state — "environmental" covers all 8 environmental variables (picked via
 * a nested sub-list) and "prediction" corresponds to interactionMode ===
 * "predict". No new global state is introduced.
 */
type MapLayerChoice = "occurrence" | "environmental" | "prediction" | "density";

function layerChoiceFor(mapMode: MapMode, interactionMode: InteractionMode): MapLayerChoice {
  if (interactionMode === "predict") return "prediction";
  if (interactionMode === "density") return "density";
  if (isEnvironmentalMode(mapMode)) return "environmental";
  return "occurrence";
}

export function Sidebar({
  summary,
  filters,
  onFilterChange,
  mapMode,
  onChangeMapMode,
  envStatsGlobal,
  baseMapStyle,
  onChangeBaseMapStyle,
  onResetMap,
  interactionMode,
  onChangeInteractionMode,
}: SidebarProps) {
  const activeLayer = layerChoiceFor(mapMode, interactionMode);

  function selectLayer(choice: MapLayerChoice) {
    if (choice === "prediction") {
      onChangeInteractionMode("predict");
      return;
    }
    if (choice === "density") {
      onChangeInteractionMode("density");
      onChangeMapMode("occurrences");
      return;
    }
    onChangeInteractionMode("browse");
    if (choice === "occurrence") {
      onChangeMapMode("occurrences");
    } else if (!isEnvironmentalMode(mapMode)) {
      // Coming from "occurrence" or "prediction" with no variable previously
      // selected — default to the first environmental variable.
      onChangeMapMode(ENV_VARIABLE_KEYS[0]);
    }
  }

  return (
    <aside className="gs-sidebar" aria-label="Map controls">
      <section className="gs-sidebar__section">
        <h2 className="gs-sidebar__heading">Study Region</h2>
        <div className="gs-sidebar__region-card">
          <p className="gs-sidebar__region-name">GeoSea global occurrence dataset</p>
          <p className="gs-sidebar__region-sub">
            {summary.latitudeRange[0].toFixed(1)}° to {summary.latitudeRange[1].toFixed(1)}° lat,{" "}
            {summary.longitudeRange[0].toFixed(1)}° to {summary.longitudeRange[1].toFixed(1)}° lon
          </p>
        </div>
      </section>

      <section className="gs-sidebar__section">
        <div className="gs-sidebar__heading-row">
          <h2 className="gs-sidebar__heading">Filters</h2>
          {hasActiveFilters(filters) && (
            <button
              type="button"
              className="gs-sidebar__clear"
              onClick={() =>
                onFilterChange({ species: "all", presence: "all", fold: "all", envRange: null })
              }
            >
              Clear
            </button>
          )}
        </div>

        <p className="gs-sidebar__label">Species Explorer</p>
        <SpeciesSelect value={filters.species} onChange={(species) => onFilterChange({ species })} />

        <p className="gs-sidebar__label">Observation type</p>
        <div className="gs-sidebar__toggle-group" role="group" aria-label="Presence or absence">
          {PRESENCE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              className={
                "gs-sidebar__toggle" + (filters.presence === opt.value ? " gs-sidebar__toggle--active" : "")
              }
              onClick={() => onFilterChange({ presence: opt.value })}
              aria-pressed={filters.presence === opt.value}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </section>

      <SpeciesExplorerProfile species={filters.species} />

      <section className="gs-sidebar__section">
        <h2 className="gs-sidebar__heading">Map Layer</h2>

        <ul className="gs-sidebar__layers" role="radiogroup" aria-label="Map layer">
          <li>
            <label className="gs-sidebar__layer">
              <input
                type="radio"
                name="gs-map-layer"
                checked={activeLayer === "occurrence"}
                onChange={() => selectLayer("occurrence")}
              />
              <span>Observed Occurrence Distribution</span>
            </label>
          </li>
          <li>
            <label className="gs-sidebar__layer">
              <input
                type="radio"
                name="gs-map-layer"
                checked={activeLayer === "environmental"}
                onChange={() => selectLayer("environmental")}
              />
              <span>Environmental Conditions</span>
            </label>
          </li>
          <li>
            <label className="gs-sidebar__layer">
              <input
                type="radio"
                name="gs-map-layer"
                checked={activeLayer === "prediction"}
                onChange={() => selectLayer("prediction")}
              />
              <span>Habitat Suitability Prediction</span>
            </label>
          </li>
          <li>
            <label className="gs-sidebar__layer">
              <input
                type="radio"
                name="gs-map-layer"
                checked={activeLayer === "density"}
                onChange={() => selectLayer("density")}
              />
              <span>Occurrence Density</span>
            </label>
          </li>
        </ul>

        {activeLayer === "environmental" && (
          <>
            <p className="gs-sidebar__group-label">Variable</p>
            <ul
              className="gs-sidebar__layers gs-sidebar__layers--compact"
              role="radiogroup"
              aria-label="Environmental variable"
            >
              {ENV_VARIABLE_KEYS.map((key) => (
                <li key={key}>
                  <label className="gs-sidebar__layer">
                    <input
                      type="radio"
                      name="gs-env-variable"
                      checked={mapMode === key}
                      onChange={() => onChangeMapMode(key)}
                    />
                    <span>{ENV_VARIABLE_META[key].label}</span>
                  </label>
                </li>
              ))}
            </ul>
          </>
        )}

        {activeLayer === "prediction" && (
          <p className="gs-sidebar__predict-hint">Click a location on the map to evaluate habitat suitability.</p>
        )}
        {activeLayer === "density" && (
          <p className="gs-sidebar__predict-hint">Shows where the loaded occurrence records are most concentrated. Click a density area for local counts.</p>
        )}
      </section>

      {activeLayer === "environmental" && isEnvironmentalMode(mapMode) && (
        <EnvironmentalControls
          key={mapMode}
          variable={mapMode}
          filters={filters}
          onFilterChange={onFilterChange}
          statsGlobal={envStatsGlobal}
        />
      )}

      <section className="gs-sidebar__section">
        <h2 className="gs-sidebar__heading">Map Style</h2>
        <div className="gs-sidebar__toggle-group" role="group" aria-label="Base map style">
          <button
            type="button"
            className={
              "gs-sidebar__toggle" + (baseMapStyle === "standard" ? " gs-sidebar__toggle--active" : "")
            }
            onClick={() => onChangeBaseMapStyle("standard")}
            aria-pressed={baseMapStyle === "standard"}
          >
            Base Map
          </button>
          <button
            type="button"
            className={
              "gs-sidebar__toggle" + (baseMapStyle === "satellite" ? " gs-sidebar__toggle--active" : "")
            }
            onClick={() => onChangeBaseMapStyle("satellite")}
            aria-pressed={baseMapStyle === "satellite"}
          >
            Satellite
          </button>
        </div>
      </section>

      <button type="button" className="gs-sidebar__reset" onClick={onResetMap}>
        Reset map view
      </button>

      <p className="gs-sidebar__footnote">
        {formatCount(summary.recordCount)} records loaded · {formatCount(summary.speciesCount)} species ·
        8 environmental variables available as map layers.
      </p>
    </aside>
  );
}

function SpeciesExplorerProfile({ species }: { species: string | "all" }) {
  const profile: SpeciesProfile | null = useMemo(
    () => (species === "all" ? null : getSpeciesProfile(species)),
    [species]
  );

  return (
    <section className="gs-sidebar__section gs-species-profile">
      <div className="gs-sidebar__heading-row">
        <h2 className="gs-sidebar__heading">Species Profile</h2>
        {profile && <span className="gs-species-profile__badge">Selected</span>}
      </div>
      {!profile ? (
        <p className="gs-species-profile__empty">Select a species above to explore its occurrence count, geographic coverage, and environmental range.</p>
      ) : (
        <>
          <h3 className="gs-species-profile__name">{profile.scientificName}</h3>
          <div className="gs-species-profile__count">
            <strong>{formatCount(profile.occurrenceCount)}</strong>
            <span>observed occurrences</span>
          </div>
          <div className="gs-species-profile__coverage">
            <div><span>Latitude</span><strong>{profile.latitudeRange[0].toFixed(1)}° to {profile.latitudeRange[1].toFixed(1)}°</strong></div>
            <div><span>Longitude</span><strong>{profile.longitudeRange[0].toFixed(1)}° to {profile.longitudeRange[1].toFixed(1)}°</strong></div>
          </div>
          <p className="gs-species-profile__subheading">Environmental range</p>
          <div className="gs-species-profile__env">
            {ENV_VARIABLE_KEYS.map((key) => {
              const meta = ENV_VARIABLE_META[key];
              const [min, max] = profile.environmentalRanges[key];
              return (
                <div key={key}>
                  <span>{meta.label}</span>
                  <strong>{min.toFixed(meta.decimals)} – {max.toFixed(meta.decimals)}{meta.unit ? ` ${meta.unit}` : ""}</strong>
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}

interface EnvironmentalControlsProps {
  variable: Exclude<MapMode, "occurrences">;
  filters: FilterState;
  onFilterChange: (patch: Partial<FilterState>) => void;
  statsGlobal: EnvironmentalStats | null;
}

/** Range filter + compact statistics panel for the currently-selected environmental variable. */
function EnvironmentalControls({
  variable,
  filters,
  onFilterChange,
  statsGlobal,
}: EnvironmentalControlsProps) {
  const meta = ENV_VARIABLE_META[variable];
  const datasetMin = statsGlobal?.min ?? 0;
  const datasetMax = statsGlobal?.max ?? 0;
  const active = filters.envRange && filters.envRange.variable === variable ? filters.envRange : null;

  const [minInput, setMinInput] = useState(active ? String(active.min) : "");
  const [maxInput, setMaxInput] = useState(active ? String(active.max) : "");

  // Keep local text inputs in sync if the variable's active range is cleared
  // elsewhere (e.g. "Clear filters").
  useEffect(() => {
    setMinInput(active ? String(active.min) : "");
    setMaxInput(active ? String(active.max) : "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active?.min, active?.max]);

  function commitRange(nextMinText: string, nextMaxText: string) {
    const min = nextMinText.trim() === "" ? datasetMin : Number.parseFloat(nextMinText);
    const max = nextMaxText.trim() === "" ? datasetMax : Number.parseFloat(nextMaxText);
    if (Number.isNaN(min) || Number.isNaN(max)) return;
    onFilterChange({ envRange: { variable, min: Math.min(min, max), max: Math.max(min, max) } });
  }

  return (
    <section className="gs-sidebar__section">
      <div className="gs-sidebar__heading-row">
        <h2 className="gs-sidebar__heading">{meta.label} Range</h2>
        {active && (
          <button
            type="button"
            className="gs-sidebar__clear"
            onClick={() => {
              onFilterChange({ envRange: null });
              setMinInput("");
              setMaxInput("");
            }}
          >
            Reset range
          </button>
        )}
      </div>

      <div className="gs-envrange">
        <label className="gs-envrange__field">
          <span>Min</span>
          <input
            type="number"
            inputMode="decimal"
            step="any"
            placeholder={datasetMin.toFixed(meta.decimals)}
            value={minInput}
            onChange={(e) => setMinInput(e.target.value)}
            onBlur={() => commitRange(minInput, maxInput)}
          />
        </label>
        <label className="gs-envrange__field">
          <span>Max</span>
          <input
            type="number"
            inputMode="decimal"
            step="any"
            placeholder={datasetMax.toFixed(meta.decimals)}
            value={maxInput}
            onChange={(e) => setMaxInput(e.target.value)}
            onBlur={() => commitRange(minInput, maxInput)}
          />
        </label>
      </div>
      <p className="gs-envrange__hint">
        Dataset range: {datasetMin.toFixed(meta.decimals)} – {datasetMax.toFixed(meta.decimals)}
        {meta.unit ? ` ${meta.unit}` : ""}
      </p>

    </section>
  );
}
