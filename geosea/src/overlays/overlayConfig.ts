export type OverlayType = 'tile' | 'image' | 'geojson' | 'wms';

export interface OverlayConfig {
  id: string;
  label: string;
  description: string;
  type: OverlayType;
  defaultOn: boolean;
  defaultOpacity: number;
  attribution: string;
  url?: string;
  defaultDate?: string;
  maxNativeZoom?: number;
  unavailableMessage?: string;
  bounds?: [[number, number], [number, number]];
  isConfigured?: boolean;
}

export const OVERLAYS: OverlayConfig[] = [
  {
    id: 'satellite',
    label: 'NASA True-Colour Satellite',
    description: 'VIIRS SNPP Corrected Reflectance (True Color)',
    type: 'tile',
    defaultOn: false,
    defaultOpacity: 1.0,
    attribution: 'NASA EOSDIS GIBS',
    url: 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/{time}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg',
    defaultDate: '2022-07-27',
    maxNativeZoom: 9,
    isConfigured: true
  },
  {
    id: 'chlorophyll',
    label: 'NASA Ocean Chlorophyll-a',
    description: 'VIIRS NOAA-21 Chlorophyll-a concentration',
    type: 'tile',
    defaultOn: false,
    defaultOpacity: 0.8,
    attribution: 'NASA EOSDIS GIBS',
    url: 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_NOAA21_Chlorophyll_a/default/{time}/GoogleMapsCompatible_Level7/{z}/{y}/{x}.png',
    defaultDate: '2025-10-23',
    maxNativeZoom: 7,
    isConfigured: true
  },
  {
    id: 'sst',
    label: 'NASA Sea Surface Temperature',
    description: 'GHRSST L4 MUR Sea Surface Temperature',
    type: 'tile',
    defaultOn: false,
    defaultOpacity: 0.8,
    attribution: 'NASA EOSDIS GIBS',
    url: 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/GHRSST_L4_MUR_Sea_Surface_Temperature/default/{time}/GoogleMapsCompatible_Level7/{z}/{y}/{x}.png',
    defaultDate: '2021-02-19',
    maxNativeZoom: 7,
    isConfigured: true
  },
  {
    id: 'bathymetry',
    label: 'Bathymetry (GEBCO)',
    description: 'Detailed seafloor depth mapping.',
    type: 'wms',
    defaultOn: false,
    defaultOpacity: 0.8,
    attribution: 'GEBCO',
    url: 'https://wms.gebco.net/mapserv',
    isConfigured: true
  },
  {
    id: 'icesat2',
    label: 'LiDAR Tracks (ICESat-2)',
    description: 'Local GeoJSON LiDAR tracks showing ocean surface features.',
    type: 'geojson',
    defaultOn: false,
    defaultOpacity: 0.9,
    attribution: 'NASA ICESat-2',
    url: '/data/overlays/icesat2_sample.geojson',
    unavailableMessage: 'Live ICESat-2 track fetching requires a configured API backend/credentials. Provide a local GeoJSON file at public/data/overlays/icesat2_sample.geojson to view LiDAR tracks manually.',
    isConfigured: false
  },
  {
    id: 'processed_image',
    label: 'Processed Model Output',
    description: 'Preview local PNG output from a custom CV model.',
    type: 'image',
    defaultOn: false,
    defaultOpacity: 0.7,
    attribution: 'Local Model',
    isConfigured: true
  }
];
