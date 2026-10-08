import type { DatasetSummary } from "@/types/environmental";
import { ENV_VARIABLE_KEYS } from "@/types/environmental";
import { formatCount } from "@/utils/formatters";
import "./StatsBanner.css";

export function StatsBanner({ summary }: { summary: DatasetSummary }) {
  const stats = [
    ["Total occurrences", summary.recordCount],
    ["Presence", summary.presenceCount],
    ["Pseudo-absence", summary.absenceCount],
    ["Species", summary.speciesCount],
  ] as const;

  return (
    <section className="gs-stats-banner" aria-label="GeoSea dataset summary">
      {stats.map(([label, value]) => (
        <div className="gs-stats-banner__item" key={label}>
          <span className="gs-stats-banner__value">{formatCount(value)}</span>
          <span className="gs-stats-banner__label">{label}</span>
        </div>
      ))}
      <div className="gs-stats-banner__item gs-stats-banner__item--variables">
        <span className="gs-stats-banner__value">{ENV_VARIABLE_KEYS.length}</span>
        <span className="gs-stats-banner__label">Environmental variables</span>
      </div>
    </section>
  );
}
