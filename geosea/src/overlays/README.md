# GeoSea Overlays

This directory manages the optional, independent overlay layers for the GeoSea map (Satellite imagery, Bathymetry, LiDAR, and Processed Outputs).

## Enabling Overlays
By default, overlays are enabled in the frontend. You can explicitly disable the overlay feature by setting the `VITE_ENABLE_OVERLAYS=false` environment variable in `geosea/.env`.

All individual overlays default to OFF (disabled) when the application loads, regardless of the global feature flag.

## Configuring New Sources (e.g., Bathymetry)
If you wish to configure a free Bathymetry tile source or other tile overlays:
1. Open `overlayConfig.ts`.
2. Locate the layer configuration (e.g., `id: 'bathymetry'`).
3. Set the `url` property to your valid TileLayer URL.
4. Set `isConfigured: true`.
5. Remove or comment out `unavailableMessage`.

## Adding Local Assets (Processed Images / GeoJSON)
To add a locally supplied derived PNG or GeoJSON asset:
1. For **Processed Model Output**, simply use the file upload selector directly in the map's Overlay panel. You can provide the bounds (N, S, E, W) directly in the UI.
2. For **LiDAR Tracks**, live fetching requires an active NASA OpenAltimetry backend API integration which requires credentials/complex queries. Currently, you must manually supply a `icesat2_sample.geojson` in the `/data/overlays/` folder to view them.

*Important: Never commit large datasets, credentials, or raw imagery files to source control.*

## Troubleshooting Missing Tiles
For tile-based overlays (like NASA GIBS Satellite and Chlorophyll), if you see 404 console errors:
- The tiles might not be available for the selected date or at the current geographic extent.
- Missing tiles are handled gracefully (rendered as transparent) so the map UI remains unaffected.
- You can change the default dates within `overlayConfig.ts` if the active dataset rolls over.
