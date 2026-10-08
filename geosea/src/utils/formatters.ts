export function formatCoordinate(value: number): string {
  return value.toFixed(3);
}

export function formatVariable(value: number, unit: string, decimals = 1): string {
  return `${value.toFixed(decimals)}${unit ? ` ${unit}` : ""}`;
}

/** Formats an integer count with thousands separators, e.g. 336924 -> "336,924". */
export function formatCount(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}
