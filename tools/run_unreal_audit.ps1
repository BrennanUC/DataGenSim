param(
    [Parameter(Mandatory=$true)][string]$EngineRoot,
    [Parameter(Mandatory=$true)][string]$DatasetRoot,
    [ValidateSet('source','render')][string]$LOD = 'source',
    [ValidateSet('export','import')][string]$Action = 'export',
    [switch]$SolidRoundtrip
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$editor = Join-Path $EngineRoot 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
if (-not (Test-Path $editor)) { throw "Unreal editor executable not found: $editor" }
$export = Join-Path $DatasetRoot "assets\unreal_$LOD"
if ($SolidRoundtrip) { $export = Join-Path $DatasetRoot "assets\solid_roundtrip_$LOD" }
$logs = Join-Path $DatasetRoot 'logs'
New-Item -ItemType Directory -Path $export,$logs -Force | Out-Null
# Unreal normalizes UNC descriptor paths incorrectly here. Stage this task's
# editor-only project on the requested dataset drive, without changing the bases.
$runtime = Join-Path $DatasetRoot 'runtime\UnrealAudit'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$content = Join-Path $projectRoot 'simulator\FlightSchool\Content'
& robocopy $content (Join-Path $runtime 'Content') /E /COPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
if ($LASTEXITCODE -gt 7) { throw 'Could not stage working simulator content on E:' }
Copy-Item (Join-Path $projectRoot 'simulator\FlightSchool\CropVolumeAudit.uproject') $runtime -Force
$scriptName = if ($Action -eq 'import') { 'import_solid_assets.py' } else { 'export_audit.py' }
Copy-Item (Join-Path $projectRoot "simulator\editor\$scriptName") $runtime -Force
$project = Join-Path $runtime 'CropVolumeAudit.uproject'
$script = Join-Path $runtime $scriptName
$env:CROPVOLUME_AUDIT_DIR = $export
$env:CROPVOLUME_AUDIT_LOD = $LOD
$env:CROPVOLUME_DATASET_ROOT = $DatasetRoot
$env:CROPVOLUME_SOLID_SOURCE = Join-Path $projectRoot 'artifacts\solid_geometry'
if ($SolidRoundtrip) { $env:CROPVOLUME_ASSET_LIST = Join-Path $DatasetRoot 'assets\solid_imported.json' }
else { Remove-Item Env:CROPVOLUME_ASSET_LIST -ErrorAction SilentlyContinue }
$label = "$Action-$LOD" + $(if($SolidRoundtrip){'-solid'}else{''})
& $editor $project '-run=pythonscript' "-script=$script" '-unattended' '-nop4' '-nosplash' '-NullRHI' "-abslog=$logs\unreal_$label.log" *> (Join-Path $logs "unreal_$label.console.log")
if ($LASTEXITCODE -ne 0) { throw "Unreal task exited with $LASTEXITCODE; inspect $logs\unreal_$label.log" }
if ($Action -eq 'export' -and -not (Test-Path (Join-Path $export 'assets.review.json'))) { throw 'Unreal did not publish the mesh export manifest' }
Write-Output "Unreal task complete: $Action $export"
