param(
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('capture','export','solid')][string]$Action = 'capture',
    [Parameter(Mandatory=$true)][string]$Runtime,
    [Parameter(Mandatory=$true)][string]$EngineRoot,
    [ValidateRange(1,7200)][int]$TimeoutSeconds = 900
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $projectRoot 'simulator\FlightSchool'
$editor = Join-Path $EngineRoot 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
if (-not (Test-Path $editor)) { throw "Unreal editor not found: $editor" }
$requestFile = if ($Action -eq 'export') { 'export_request.json' } elseif ($Action -eq 'solid') { 'solid_scene.json' } else { 'reference.json' }
$completionFile = if ($Action -eq 'export') { 'EXPORT_FINISHED.json' } elseif ($Action -eq 'solid') { 'SOLID_RENDER_FINISHED.json' } else { 'REFERENCE_RENDER_FINISHED.json' }
if (-not (Test-Path (Join-Path $Output $requestFile))) { throw "Create $requestFile before launching Unreal" }
foreach ($marker in @('REFERENCE_COMPLETE.json', 'REFERENCE_RENDER_FINISHED.json', 'EXPORT_FINISHED.json', 'INVENTORY_COMPLETE.json', 'SOLID_RENDER_FINISHED.json', 'VOLUME_COMPLETE.json')) {
    if (Test-Path (Join-Path $Output $marker)) { throw 'Use a fresh output directory for another reference capture' }
}
if ($Action -eq 'solid' -and (Split-Path -Leaf $Runtime) -eq 'BaseSimReference') {
    throw 'Solid captures require a separate runtime, for example runtime\BaseSimSolidRegion'
}
if ((Test-Path $Runtime) -and -not (Test-Path (Join-Path $Runtime 'BaseSimReference.uproject'))) {
    throw 'Use an empty runtime directory dedicated to BaseSimReference'
}
function ContentHashes([string]$Root) {
    $result = [ordered]@{}
    Get-ChildItem -LiteralPath $Root -Recurse -File | Sort-Object FullName | ForEach-Object {
        $relative = $_.FullName.Substring($Root.TrimEnd('\').Length + 1).Replace('\', '/')
        $result[$relative] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()
    }
    return $result
}
$sourceContent = Join-Path $source 'Content'
$original = ContentHashes $sourceContent
New-Item -ItemType Directory -Path $Runtime -Force | Out-Null
foreach ($folder in @('Content', 'Config')) {
    & robocopy (Join-Path $source $folder) (Join-Path $Runtime $folder) /E /COPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -gt 7) { throw "Cannot stage original $folder" }
}
# Editor-only descriptor: no game module, menus, growth, harvest or rebuilt beds.
@'
{"FileVersion":3,"EngineAssociation":"5.8","Description":"BaseSim authored field capture working copy","Plugins":[{"Name":"PythonScriptPlugin","Enabled":true},{"Name":"EditorScriptingUtilities","Enabled":true},{"Name":"GeometryScripting","Enabled":true}]}
'@ | Set-Content (Join-Path $Runtime 'BaseSimReference.uproject') -Encoding ASCII
@'
[/Script/UnrealEd.EditorLoadingSavingSettings]
bAutoSaveEnable=False
'@ | Set-Content (Join-Path $Runtime 'Config\DefaultEditor.ini') -Encoding ASCII
$scriptName = if ($Action -eq 'export') { 'export_existing_field.py' } elseif ($Action -eq 'solid') { 'capture_solid_region.py' } else { 'capture_field_reference.py' }
$script = Join-Path $Runtime $scriptName
Copy-Item (Join-Path $projectRoot "simulator\editor\$scriptName") $script -Force
Copy-Item (Join-Path $projectRoot 'src\cropvolume\profiling.py') (Join-Path $Runtime 'cropvolume_profiling.py') -Force
if ($Action -eq 'solid') {
    $solidSource = Join-Path $projectRoot 'simulator\FlightSchool\Content\CropVolume'
    if (-not (Test-Path $solidSource)) { throw 'Audited game solid assets are not available in runtime\FlightSchool' }
    & robocopy $solidSource (Join-Path $Runtime 'Content\CropVolume') /E /COPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -gt 7) { throw 'Cannot stage audited solid assets' }
    foreach ($helper in @('capture_field_reference.py', 'export_audit.py')) {
        Copy-Item (Join-Path $projectRoot "simulator\editor\$helper") $Runtime -Force
    }
}
$runtimeContent = Join-Path $Runtime 'Content'
$allStaged = ContentHashes $runtimeContent
$staged = [ordered]@{}
foreach ($key in $original.Keys) {
    $staged[$key] = $allStaged[$key]
    if ($staged[$key] -ne $original[$key]) { throw "Staged asset differs from supplied BaseSim: $key" }
}
if ($Action -in @('export', 'solid')) {
    $request = Get-Content (Join-Path $Output $requestFile) -Raw | ConvertFrom-Json
    foreach ($property in $request.expected_content.PSObject.Properties) {
        if ($original[$property.Name] -ne $property.Value) { throw "Original content changed since the accepted reference: $($property.Name)" }
    }
    if (@($request.expected_content.PSObject.Properties).Count -ne $original.Count) { throw 'Original content file count changed' }
}
$env:CROPVOLUME_REFERENCE_OUTPUT = $Output
$env:CROPVOLUME_REFERENCE_SCRIPT = $script
$log = Join-Path $Output 'unreal_reference.log'
$arguments = @((Join-Path $Runtime 'BaseSimReference.uproject'), "-ExecutePythonScript=$script",
    '-unattended', '-nop4', '-nosplash', '-RenderOffscreen', '-NoSound', '-NoLoadingScreen', "-abslog=$log")
if ($Action -eq 'export') {
    $arguments = @((Join-Path $Runtime 'BaseSimReference.uproject'), '-run=pythonscript', "-script=$script",
        '-unattended', '-nop4', '-nosplash', '-NullRHI', '-NoSound', "-abslog=$log")
}
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $editor
$startInfo.Arguments = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
$startInfo.WorkingDirectory = $Runtime
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$process = [System.Diagnostics.Process]::Start($startInfo)
$stdout = $process.StandardOutput.ReadToEndAsync()
$stderr = $process.StandardError.ReadToEndAsync()
$finished = $process.WaitForExit($TimeoutSeconds * 1000)
if (-not $finished) { & "$env:SystemRoot\System32\taskkill.exe" /PID $process.Id /T /F | Out-Null }
$process.WaitForExit()
[System.IO.File]::WriteAllText((Join-Path $Output 'unreal_reference.console.log'), $stdout.Result)
[System.IO.File]::WriteAllText((Join-Path $Output 'unreal_reference.stderr.log'), $stderr.Result)
$evidence = @{source=$source; files=$original; staged_files=$staged;
    files_after=(ContentHashes $sourceContent)}
$allAfter = ContentHashes $runtimeContent
$stagedAfter = [ordered]@{}
foreach ($key in $original.Keys) { $stagedAfter[$key] = $allAfter[$key] }
$evidence.staged_files_after = $stagedAfter
if ($Action -eq 'solid') {
    $extrasBefore = [ordered]@{}; $extrasAfter = [ordered]@{}
    foreach ($key in $allStaged.Keys) {
        if (-not $original.Contains($key)) { $extrasBefore[$key] = $allStaged[$key]; $extrasAfter[$key] = $allAfter[$key] }
    }
    $evidence.solid_files = $extrasBefore
    $evidence.solid_files_after = $extrasAfter
}
$evidence | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Output 'source_content.json') -Encoding ASCII
if (-not $finished) { throw "Reference capture timed out after $TimeoutSeconds seconds. Inspect $log" }
if ($process.ExitCode -ne 0) { throw "Unreal reference capture failed: $($process.ExitCode). Inspect $log" }
if (-not (Test-Path (Join-Path $Output $completionFile))) { throw "Original-field $Action did not complete; inspect $log" }
Write-Output "Original-field $Action finished: $Output"
