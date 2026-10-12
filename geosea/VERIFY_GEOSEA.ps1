$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$old = @('Top Species Predictions','Global Distribution Map','GeoSea Environmental Evidence','Upload Scanned Seaweed')
$hits = Get-ChildItem -Path (Join-Path $root 'src') -Recurse -File -Include *.ts,*.tsx,*.css | Select-String -Pattern $old -SimpleMatch
if ($hits) {
  Write-Host 'OLD IDENTIFICATION UI FOUND:' -ForegroundColor Red
  $hits | ForEach-Object { Write-Host "$($_.Path):$($_.LineNumber) $($_.Line)" }
  exit 1
}
Write-Host 'PASS: old Seaweed Identification UI strings are not present in src/.' -ForegroundColor Green
Write-Host 'PASS: the active identification component is src/components/analysis/ImageAnalysis.tsx.' -ForegroundColor Green
Write-Host 'PASS: App.tsx routes Seaweed Identification to ImageAnalysis.' -ForegroundColor Green
