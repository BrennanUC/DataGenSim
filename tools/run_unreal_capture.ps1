param(
    [Parameter(Mandatory=$true)][string]$Sample,
    [Parameter(Mandatory=$true)][string]$Catalog,
    [Parameter(Mandatory=$true)][string]$EngineRoot,
    [Parameter(Mandatory=$true)][string]$DatasetRoot,
    [ValidateRange(1,7200)][int]$TimeoutSeconds = 900
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$scene = Get-Content (Join-Path $Sample 'scene.json') -Raw | ConvertFrom-Json
$useGame = $scene.field.backend -eq 'unreal_game'
$useBaseSim = $scene.field.backend -eq 'unreal_basesim_editor'
$runtime = Join-Path $DatasetRoot $(if ($useGame) { 'runtime\FlightSchool' } elseif ($useBaseSim) { 'runtime\BaseSimRandomField' } else { 'runtime\UnrealAudit' })
$project = Join-Path $runtime $(if ($useGame) { 'FlightSchool.uproject' } elseif ($useBaseSim) { 'BaseSimRandomField.uproject' } else { 'CropVolumeAudit.uproject' })
if ($useBaseSim) {
    & (Join-Path $PSScriptRoot 'stage_random_basesim.ps1') -Runtime $runtime -ProjectRoot $projectRoot -DatasetRoot $DatasetRoot | Out-Null
}
if (-not (Test-Path $project)) { throw 'Run the Unreal audit/import setup before capture' }
$editor = Join-Path $EngineRoot 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
$catalogData = Get-Content $Catalog -Raw | ConvertFrom-Json
if ($catalogData.catalog_id -ne $scene.catalog_id -or $scene.test_fixture) { throw 'Capture catalog mismatch or fixture' }
foreach ($assetId in @($scene.instances.asset_id | Sort-Object -Unique)) {
    $asset = $catalogData.assets | Where-Object { $_.id -eq $assetId }
    if (-not $asset.approved) { throw 'Unapproved plant asset' }
    $package = ($asset.unreal_asset -split '\.')[0] -replace '^/Game/', ''
    $package = Join-Path $runtime ('Content\' + $package + '.uasset')
    if ((Get-FileHash $package -Algorithm SHA256).Hash.ToLower() -ne $asset.unreal_uasset_sha256) { throw 'Plant asset changed since geometry audit' }
}
if (-not $useGame) {
    Copy-Item (Join-Path $projectRoot 'simulator\editor\render_sample.py') $runtime -Force
    Copy-Item (Join-Path $projectRoot 'src\cropvolume\profiling.py') (Join-Path $runtime 'cropvolume_profiling.py') -Force
}
$script = Join-Path $runtime 'render_sample.py'
$env:CROPVOLUME_CAPTURE_SCRIPT = $script
$env:CROPVOLUME_SAMPLE = $Sample
$env:CROPVOLUME_CATALOG = $Catalog
$config = Join-Path $runtime 'Config'
if (-not $useGame -and -not $useBaseSim) {
New-Item -ItemType Directory -Path $config -Force | Out-Null
@'
[/Script/Engine.RendererSettings]
r.DefaultFeature.AutoExposure.ExtendDefaultLuminanceRange=True
r.GenerateMeshDistanceFields=True
r.DynamicGlobalIlluminationMethod=1
r.ReflectionMethod=1
r.AntiAliasingMethod=2
'@ | Set-Content (Join-Path $config 'DefaultEngine.ini') -Encoding ASCII
}
$log = Join-Path $Sample 'unreal_capture.log'
$arguments = @($project, "-ExecutePythonScript=$script", '-unattended', '-nop4', '-nosplash', '-RenderOffscreen', '-NoSound', '-NoLoadingScreen', "-abslog=$log")
if ($useGame) {
    $binary = Join-Path $runtime 'Binaries\Win64\UnrealEditor-FlightSchool.dll'
    if (-not (Test-Path $binary)) { throw 'Build the FlightSchool editor module before game capture' }
    $env:CROPVOLUME_CAPTURE_BINARY_SHA256 = (Get-FileHash $binary -Algorithm SHA256).Hash.ToLower()
    $arguments = @($project, ($scene.field.source_map + '?game=/Script/FlightSchool.FSCropVolumeGameMode'),
        '-game', '-CropVolumeDataset', '-unattended', '-nop4', '-nosplash', '-RenderOffscreen', '-NoSound',
        '-windowed', '-ResX=1024', '-ResY=768', "-abslog=$log")
}
# Quote each complete Windows argument, including paths containing spaces.
$argumentLine = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $editor
$startInfo.Arguments = $argumentLine
$startInfo.WorkingDirectory = $runtime
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
# Own the process handle directly: Windows PowerShell Start-Process can return
# a detached object whose ExitCode stays null after WaitForExit().
$process = [System.Diagnostics.Process]::Start($startInfo)
$processRecordPath = Join-Path $Sample 'unreal_process.json'
$processRecord = @{pid=$process.Id;started_utc_ticks=$process.StartTime.ToUniversalTime().Ticks.ToString();state='running';log=$log}
[System.IO.File]::WriteAllText(($processRecordPath+'.tmp'),($processRecord|ConvertTo-Json -Compress),(New-Object System.Text.UTF8Encoding($false)))
Move-Item ($processRecordPath+'.tmp') $processRecordPath -Force
$stdout = $process.StandardOutput.ReadToEndAsync()
$stderr = $process.StandardError.ReadToEndAsync()
$finished = $process.WaitForExit($TimeoutSeconds * 1000)
if (-not $finished) {
    # Terminate only this task's editor and its child shader workers.
    & "$env:SystemRoot\System32\taskkill.exe" /PID $process.Id /T /F | Out-Null
}
$process.WaitForExit()
$processRecord.state='exited'
$processRecord.exit_code=$process.ExitCode
[System.IO.File]::WriteAllText(($processRecordPath+'.tmp'),($processRecord|ConvertTo-Json -Compress),(New-Object System.Text.UTF8Encoding($false)))
Move-Item ($processRecordPath+'.tmp') $processRecordPath -Force
[System.IO.File]::WriteAllText((Join-Path $Sample 'unreal_capture.console.log'), $stdout.Result)
[System.IO.File]::WriteAllText((Join-Path $Sample 'unreal_capture.stderr.log'), $stderr.Result)
if (-not $finished) { throw "Unreal capture timed out after $TimeoutSeconds seconds. Inspect $log" }
if ($process.ExitCode -ne 0) { throw "Unreal capture failed: $($process.ExitCode). Inspect $log" }
if (-not (Test-Path (Join-Path $Sample 'RENDER_FINISHED.json'))) { throw "Capture did not complete; inspect $log" }
Write-Output "Capture complete: $Sample"
