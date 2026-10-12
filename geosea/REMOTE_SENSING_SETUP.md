# GeoSea Sentinel remote-sensing setup

When a user clicks a location in Habitat Suitability Prediction, GeoSea keeps the existing Bio-ORACLE + Random Forest workflow unchanged and separately requests a satellite image.

## Image selection

1. Sentinel-2 Surface Reflectance is searched first.
2. The local cloud screen combines the Sentinel-2 Scene Classification Layer and the matching cloud-probability product.
3. If Sentinel-2 has no usable local image, Sentinel-3 OLCI is requested as the broader-area fallback.
4. The satellite image is visual context only. No automatic seaweed detection is performed from it.

## Setup

1. Install backend dependencies:

```powershell
cd backend
pip install -r requirements.txt
```

2. Authenticate Earth Engine once:

```powershell
earthengine authenticate
```

3. Create `backend/.env`:

```dotenv
EARTHENGINE_PROJECT=your-real-google-cloud-project-id
```

4. Start FastAPI:

```powershell
uvicorn app:app --reload
```

## API endpoints

- `POST /remote-sensing/sentinel2` — preferred Sentinel-2 image.
- `POST /remote-sensing/sentinel3` — fallback Sentinel-3 image.
- `POST /predict` — existing Bio-ORACLE + Random Forest prediction.

NOAA marine indicators are not part of this dashboard version.
