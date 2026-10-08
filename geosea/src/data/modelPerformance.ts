/** Offline model-comparison results from the same 5-fold spatial CV scheme. */
import type { ModelPerformanceMetrics } from "@/types/environmental";

export const SPATIAL_CV_DESCRIPTION =
  "5-fold spatial cross-validation using the same dataset, features, and spatial folds for each model.";

export const MODEL_PERFORMANCE: ModelPerformanceMetrics[] = [
  { modelId: "random_forest", label: "Random Forest", spatialRocAuc: 0.9431388, accuracy: 0.8193257, precision: 0.9605677, recall: 0.6478750, f1: 0.7689746 },
  { modelId: "xgboost", label: "XGBoost", spatialRocAuc: 0.9391, accuracy: 0.8447, precision: 0.9540, recall: 0.7118, f1: 0.8147 },
  { modelId: "lightgbm", label: "LightGBM", spatialRocAuc: 0.9369, accuracy: 0.8314, precision: 0.9489, recall: 0.6865, f1: 0.7954 },
  { modelId: "gam_maxent", label: "GAM", spatialRocAuc: 0.9213, accuracy: 0.8506, precision: 0.8781, recall: 0.7983, f1: 0.8350 },
  { modelId: "tabular_mlp", label: "Tabular MLP", spatialRocAuc: 0.9074, accuracy: 0.8299, precision: 0.9210, recall: 0.7064, f1: 0.7943 },
  { modelId: "spatial_gnn", label: "Spatial GNN", spatialRocAuc: 0.9307, accuracy: 0.7873, precision: 0.9521, recall: 0.5967, f1: 0.7336 },
];
