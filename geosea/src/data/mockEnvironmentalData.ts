/**
 * MOCK DATA — for UI development only.
 *
 * This file exists so the interface has something realistic to render
 * before the real GeoSea_model_ready_with_folds.csv is connected.
 *
 * Replace usage of this file by swapping the implementation of
 * src/services/dataService.ts — no component should import this file
 * directly. See dataService.ts for the exact swap point.
 */
import type { OccurrenceRecord } from "@/types/environmental";

export const MOCK_STUDY_REGION_LABEL = "GeoSea Study Region";

type MockRow = Omit<OccurrenceRecord, "index" | "isPseudoAbsence">;

const rawMockOccurrences: MockRow[] = [
  {
    scientific_name: "Sargassum muticum",
    latitude: 50.372,
    longitude: -4.142,
    presence: 1,
    temperature: 14.2,
    salinity: 34.8,
    chlorophyll: 2.1,
    nitrate: 3.4,
    phosphate: 0.42,
    pH: 8.06,
    PAR: 32.5,
    kdpar: 0.14,
    spatial_fold: 0,
  },
  {
    scientific_name: "Laminaria digitata",
    latitude: 56.132,
    longitude: -5.845,
    presence: 1,
    temperature: 11.6,
    salinity: 33.9,
    chlorophyll: 3.4,
    nitrate: 5.1,
    phosphate: 0.55,
    pH: 8.02,
    PAR: 24.8,
    kdpar: 0.19,
    spatial_fold: 1,
  },
  {
    scientific_name: "Fucus vesiculosus",
    latitude: 53.404,
    longitude: -3.049,
    presence: 1,
    temperature: 13.1,
    salinity: 32.6,
    chlorophyll: 4.0,
    nitrate: 4.6,
    phosphate: 0.48,
    pH: 8.01,
    PAR: 27.9,
    kdpar: 0.21,
    spatial_fold: 2,
  },
  {
    scientific_name: "Undaria pinnatifida",
    latitude: 50.821,
    longitude: -1.088,
    presence: 0,
    temperature: 15.0,
    salinity: 34.2,
    chlorophyll: 1.6,
    nitrate: 2.2,
    phosphate: 0.31,
    pH: 8.09,
    PAR: 35.1,
    kdpar: 0.11,
    spatial_fold: 3,
  },
  {
    scientific_name: "Palmaria palmata",
    latitude: 58.209,
    longitude: -6.388,
    presence: 1,
    temperature: 10.4,
    salinity: 34.5,
    chlorophyll: 2.9,
    nitrate: 6.0,
    phosphate: 0.61,
    pH: 7.98,
    PAR: 21.3,
    kdpar: 0.23,
    spatial_fold: 4,
  },
  {
    scientific_name: "Ulva lactuca",
    latitude: 51.507,
    longitude: -3.176,
    presence: 1,
    temperature: 13.8,
    salinity: 31.9,
    chlorophyll: 5.2,
    nitrate: 7.3,
    phosphate: 0.7,
    pH: 7.95,
    PAR: 26.4,
    kdpar: 0.26,
    spatial_fold: 0,
  },
  {
    scientific_name: "Sargassum muticum",
    latitude: 49.184,
    longitude: -2.107,
    presence: 0,
    temperature: 15.4,
    salinity: 34.9,
    chlorophyll: 1.4,
    nitrate: 1.9,
    phosphate: 0.28,
    pH: 8.11,
    PAR: 36.7,
    kdpar: 0.1,
    spatial_fold: 1,
  },
];

export const mockOccurrences: OccurrenceRecord[] = rawMockOccurrences.map((row, index) => ({
  ...row,
  index,
  isPseudoAbsence: row.presence === 0,
}));

/** Center used for the initial map view — mean of the mock points. */
export const MOCK_MAP_CENTER: [number, number] = [52.9, -3.7];
export const MOCK_MAP_ZOOM = 6;
