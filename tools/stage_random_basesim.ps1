param([Parameter(Mandatory=$true)][string]$Runtime,[Parameter(Mandatory=$true)][string]$ProjectRoot,[Parameter(Mandatory=$true)][string]$DatasetRoot)
$ErrorActionPreference='Stop'
$source=Join-Path $ProjectRoot 'simulator\FlightSchool'
$project=Join-Path $Runtime 'BaseSimRandomField.uproject'
if ((Test-Path $Runtime) -and -not (Test-Path $project)) { throw 'Use a dedicated BaseSimRandomField runtime directory' }
New-Item -ItemType Directory -Path $Runtime -Force | Out-Null
foreach ($folder in @('Content','Config')) {
    & robocopy (Join-Path $source $folder) (Join-Path $Runtime $folder) /E /COPY:DAT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -gt 7) { throw "Cannot stage original BaseSim $folder" }
}
# Audited plant packages are bundled with the field content.
@'
{"FileVersion":3,"EngineAssociation":"5.8","Description":"Random targets in the original BaseSim field environment","Plugins":[{"Name":"PythonScriptPlugin","Enabled":true},{"Name":"EditorScriptingUtilities","Enabled":true}]}
'@ | Set-Content $project -Encoding ASCII
@'
[/Script/UnrealEd.EditorLoadingSavingSettings]
bAutoSaveEnable=False
'@ | Set-Content (Join-Path $Runtime 'Config\DefaultEditor.ini') -Encoding ASCII
Copy-Item (Join-Path $ProjectRoot 'simulator\editor\capture_field_reference.py') $Runtime -Force
Write-Output $project
