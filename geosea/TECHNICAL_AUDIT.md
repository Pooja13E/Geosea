# GeoSea Technical Audit — final cleanup

## Completed in this release

- Seaweed Identification remains a single three-step workflow: image → location → environmental evidence.
- Removed the old WoRMS/GBIF distribution workflow from the image-analysis backend.
- Removed unused literature-range and species-profile backend routes/files.
- Removed geographic-distance verification because the current workflow ends at environmental evidence.
- Standardized **High / Medium / Low** environmental-consistency thresholds to 80 / 50.
- Standardized **Environmental consistency** terminology in the active workflow.
- Standardized the classifier name to **EfficientNet-B0 V2**.
- Corrected PAR wording to **Photosynthetically Active Radiation (PAR)**.
- Removed unused `AppView` type; active `InteractionMode` values remain because browse, prediction, density, and verification are used.
- Removed duplicate/generated Vite files: `vite.config.js` and `vite.config.d.ts`.
- Removed historical merge/update note files from the distributable release.
- Kept `backend/.env` intentionally because the current development workflow requires it.
- Removed generated build/cache artifacts from the ZIP.

## Species-count correction

The raw CSV has **10,269 distinct source name strings**. That figure is not a valid species count because the source contains author variants, varieties/subspecies, genus-level names, and higher-rank labels.

The dashboard headline now reports **5,615 canonical binomial-name entries**. This is a conservative name-normalization step. It does not claim that all 5,615 are currently accepted species because true synonym resolution requires an authoritative taxonomic backbone and reviewed mappings.

The underlying CSV rows and both trained ML models are unchanged. Therefore this correction changes the **reported/displayed taxonomy count**, not model predictions or training metrics.

## Remaining scientific improvement

For a publication-quality accepted-species count, create a versioned taxonomy mapping using WoRMS or GBIF: source name → accepted Aphia/GBIF taxon → accepted scientific name → rank. Review ambiguous cases before replacing the 5,615 conservative headline count. Do not silently merge taxa based only on string similarity.

## Recommended QA before final submission

- Test 1366×768, 1440×900, and 1920×1080 layouts.
- Test image upload, classification, map selection, Bio-ORACLE success, missing-variable response, and backend timeout/error states.
- Confirm `npm run build` after `npm install`.
- Run `python -m py_compile backend/*.py`.
- Keep `.env` local/private when the project is published.


## V5 follow-up fixes

- Restored the missing `POST /analyze-image` route in `backend/analysis.py`. The frontend already called this route, but it had been accidentally removed during cleanup; this caused the literal FastAPI `Not Found` error after clicking Identify Seaweed Species.
- Species Explorer and occurrence map labels now share the same conservative binomial normalization at CSV ingestion. Source CSV names are preserved and not overwritten. Author/rank suffix variants collapse to one canonical binomial label in the UI; genus-only and higher-rank records remain in the dataset/map but are not advertised as species options.
- This is not synonym resolution. Do not merge distinct accepted taxa until a reviewed WoRMS/GBIF taxonomic mapping is available.

## V6 — API startup resilience and consistent taxon names

- Made Google Earth Engine an optional import for the backend. Satellite endpoints return a clear setup message if `earthengine-api` is absent, while core GeoSea routes can still start.
- Deferred importing the heavyweight Torch/Torchvision classifier until `/analyze-image` is requested. A classifier dependency/model error now returns a descriptive HTTP 503 instead of preventing the entire API from importing.
- Added shared conservative scientific-name formatting in `src/utils/scientificNames.ts` and `backend/scientific_names.py`. Species names are normalized to `Genus species`; author citations and infraspecific suffixes are not displayed as separate species; genus-only/higher-rank names are not added to the species selector. This is formatting, not synonym resolution.
- The verification endpoint now filters to canonical binomial presence records and uses the same case-insensitive key as the frontend. The 12 image-classifier classes were checked against the CSV; each has matching presence records.
- Preserved the source CSV and model artifacts unchanged. Pseudo-absence keeps its dedicated internal label.
- Improved “Failed to fetch” messaging with backend startup, health-check, and CORS troubleshooting steps; `START_GEOSEA.bat` now starts the backend and frontend in separate terminals. Vite is pinned to port 5173 so its origin matches the default CORS configuration.
- Validation: Python compilation passed; FastAPI app imports with `/analyze-image` and `/species-verification` registered; `/health` responds; mocked `/analyze-image` returned HTTP 200 and normalized `Caulerpa racemosa`; all 12 classifier names match canonical CSV presence names. Frontend npm build could not be run in this environment because project npm dependencies are not installed here.
