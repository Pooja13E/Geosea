import type { MapMode } from "@/types/environmental";
import { ENV_VARIABLE_META, isEnvironmentalMode } from "@/data/envVariableConfig";
import { formatCount } from "@/utils/formatters";
import "./MapLegend.css";

interface MapLegendProps {
  mode: MapMode;
  aggregated: boolean;
  visibleCount: number;
  /** [min, max] of the color ramp currently in use; only set in environmental mode. */
  envDomain: [number, number] | null;
}

export function MapLegend({ mode, aggregated, visibleCount, envDomain }: MapLegendProps) {
  if (isEnvironmentalMode(mode)) {
    const meta = ENV_VARIABLE_META[mode];
    const [low, high] = envDomain ?? [0, 0];
    return (
      <div className="gs-legend">
        <p className="gs-legend__title">{meta.label}</p>
        <div
          className="gs-legend__gradient"
          style={{ background: `linear-gradient(to right, ${meta.colorLow}, ${meta.colorHigh})` }}
        />
        <div className="gs-legend__gradient-labels">
          <span>{low.toFixed(meta.decimals)}</span>
          <span>{high.toFixed(meta.decimals)}</span>
        </div>
        <p className="gs-legend__note">
          {meta.unit ? `Units: ${meta.unit}. ` : ""}
          Observed values, spatially aggregated by grid cell (mean). Not an interpolated surface.
        </p>
        <div className="gs-legend__divider" />
        <p className="gs-legend__note">
          {formatCount(visibleCount)} observations in view — click a cell to zoom in.
        </p>
      </div>
    );
  }

  return (
    <div className="gs-legend">
      <p className="gs-legend__title">Observed Occurrence Distribution</p>

      <div className="gs-legend__row">
        <span className="gs-legend__dot gs-legend__dot--presence" />
        <span>Presence</span>
      </div>
      <div className="gs-legend__row">
        <span className="gs-legend__dot gs-legend__dot--absence" />
        <span>Absence</span>
      </div>

      <p className="gs-legend__note">
        {aggregated
          ? "Point density represents observed occurrence distribution. Hover a point for its record count; click to zoom in."
          : `Point density represents observed occurrence distribution. ${formatCount(visibleCount)} individual records in view.`}
      </p>


    </div>
  );
}
