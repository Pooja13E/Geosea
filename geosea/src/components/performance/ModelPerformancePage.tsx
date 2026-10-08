import { MODEL_PERFORMANCE, SPATIAL_CV_DESCRIPTION } from "@/data/modelPerformance";
import { AVAILABLE_MODELS } from "@/services/predictionService";
import type { ModelPerformanceMetrics } from "@/types/environmental";
import "./ModelPerformancePage.css";

/** Single-hue bar chart comparing one metric across all six models. No charting library — kept as a small inline SVG so it follows the design tokens exactly. */
function MetricBarChart({
  title,
  rows,
  getValue,
  formatValue,
}: {
  title: string;
  rows: ModelPerformanceMetrics[];
  getValue: (row: ModelPerformanceMetrics) => number;
  formatValue: (value: number) => string;
}) {
  const width = 560;
  const rowHeight = 34;
  const gap = 10;
  const labelWidth = 108;
  const valueWidth = 56;
  const barAreaWidth = width - labelWidth - valueWidth;
  const height = rows.length * (rowHeight + gap) - gap;
  const maxValue = Math.max(...rows.map(getValue), 0.0001);

  return (
    <div className="gs-perf__chart">
      <h4>{title}</h4>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        role="img"
        aria-label={title}
        className="gs-perf__chart-svg"
      >
        {rows.map((row, i) => {
          const value = getValue(row);
          const barWidth = Math.max((value / maxValue) * barAreaWidth, 2);
          const y = i * (rowHeight + gap);
          const isPerfect = false;
          return (
            <g key={row.modelId} transform={`translate(0, ${y})`}>
              <text
                x={labelWidth - 10}
                y={rowHeight / 2}
                textAnchor="end"
                dominantBaseline="middle"
                className="gs-perf__chart-label"
              >
                {row.label}
              </text>
              <rect
                x={labelWidth}
                y={rowHeight * 0.18}
                width={barAreaWidth}
                height={rowHeight * 0.64}
                rx={4}
                className="gs-perf__chart-track"
              />
              <rect
                x={labelWidth}
                y={rowHeight * 0.18}
                width={barWidth}
                height={rowHeight * 0.64}
                rx={4}
                className={isPerfect ? "gs-perf__chart-bar gs-perf__chart-bar--flagged" : "gs-perf__chart-bar"}
              />
              <text
                x={labelWidth + barAreaWidth + valueWidth - 6}
                y={rowHeight / 2}
                textAnchor="end"
                dominantBaseline="middle"
                className="gs-perf__chart-value"
              >
                {formatValue(value)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/**
 * Model Performance / Information view. This communicates offline evaluation
 * results and distinguishes them from the deployed production model.
 */
export function ModelPerformancePage() {
  return (
    <div className="gs-perf">
      <div className="gs-perf__intro">
        <h2>Model Performance</h2>
        <p>
          Evaluation results for the model families tested for GeoSea's habitat
          suitability prediction. These are <strong>offline spatial-CV results</strong>; the deployed
          production model is Random Forest.
        </p>
      </div>

      <div className="gs-perf__notice">
        <strong>Offline evaluation vs deployment.</strong> These metrics come from 5-fold spatial
        cross-validation. <strong>Random Forest is the deployed model</strong> used by the live map
        prediction workflow. The remaining rows are comparison results and are not loaded by FastAPI.
      </div>

      <div className="gs-perf__table-wrap">
        <table className="gs-perf__table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Spatial ROC-AUC</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
              <th>Prediction Status</th>
            </tr>
          </thead>
          <tbody>
            {MODEL_PERFORMANCE.map((row) => {
              const meta = AVAILABLE_MODELS.find((m) => m.id === row.modelId);
              const isPerfect = false;
              return (
                <tr key={row.modelId}>
                  <td className="gs-perf__model-name">
                    {row.label}
                    {isPerfect && <span className="gs-perf__flag-dot" title="Perfect score — needs investigation" />}
                  </td>
                  <td>{row.spatialRocAuc.toFixed(4)}</td>
                  <td>{(row.accuracy * 100).toFixed(2)}%</td>
                  <td>{(row.precision * 100).toFixed(2)}%</td>
                  <td>{(row.recall * 100).toFixed(2)}%</td>
                  <td>{row.f1.toFixed(4)}</td>
                  <td>
                    <span className="gs-perf__status-badge">
                      {meta?.status === "ready" ? "Available" : meta?.status === "training" ? "Training" : "Not connected"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="gs-perf__caution">
        <strong>Production selection:</strong> Random Forest was selected for deployment because it
        has the highest spatial ROC-AUC in the current comparison and stable fold-to-fold performance.
        XGBoost has stronger threshold-based classification metrics, but a slightly lower ROC-AUC.
      </div>

      <div className="gs-perf__charts">
        <MetricBarChart
          title="Spatial ROC-AUC by Model"
          rows={MODEL_PERFORMANCE}
          getValue={(r) => r.spatialRocAuc}
          formatValue={(v) => v.toFixed(4)}
        />
        <MetricBarChart
          title="F1 Score by Model"
          rows={MODEL_PERFORMANCE}
          getValue={(r) => r.f1}
          formatValue={(v) => v.toFixed(4)}
        />
      </div>

      <div className="gs-perf__interpretation">
        <h3>Interpretation</h3>
        <p>
          No single model dominates every metric, and different models suit different priorities.
          Among the non-perfect results, Random Forest has the highest spatial ROC-AUC and very high
          precision, but the lowest recall — it is conservative about predicting presence. XGBoost
          offers a strong balance across accuracy, precision, recall, and F1. LightGBM sits close to
          XGBoost with a similar profile. GAM has the highest F1 and recall among the non-perfect
          results, trading some precision for that balance. Spatial GNN has high precision but comparatively low recall, similar in shape to Random Forest.
          The current MLP result is substantially below the tree-based candidates in ROC-AUC.
        </p>
      </div>

      <div className="gs-perf__methodology">
        <h3>5-Fold Spatial Cross-Validation</h3>
        <p>{SPATIAL_CV_DESCRIPTION}</p>
      </div>
    </div>
  );
}
