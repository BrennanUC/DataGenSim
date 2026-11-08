param([Parameter(Mandatory=$true)][string]$Sample)
$ErrorActionPreference='Stop'
$log=Join-Path $Sample 'unreal_capture.log'
$pattern=[regex]::Escape("-abslog=$log")+'(?=["\s]|$)'
$recordPath=Join-Path $Sample 'unreal_process.json'
$record=if(Test-Path $recordPath){Get-Content $recordPath -Raw|ConvertFrom-Json}else{$null}
$stopped=@()
if($record -and $record.state -eq 'running') {
    $existing=Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
    if($existing -and $existing.StartTime.ToUniversalTime().Ticks.ToString() -eq $record.started_utc_ticks) {
        $details=Get-CimInstance Win32_Process -Filter ("ProcessId = "+[int]$record.pid)
        if(-not $details.CommandLine){throw 'Cannot verify the recorded renderer command line; files were preserved'}
    }
}
foreach($candidate in @(Get-CimInstance Win32_Process -Filter "Name = 'UnrealEditor-Cmd.exe'")) {
    if(-not $candidate.CommandLine -or $candidate.CommandLine -notmatch $pattern){continue}
    $process=Get-Process -Id $candidate.ProcessId -ErrorAction SilentlyContinue
    if(-not $process){continue}
    if($record -and [int]$record.pid -eq $candidate.ProcessId -and $record.started_utc_ticks -and
        $process.StartTime.ToUniversalTime().Ticks.ToString() -ne $record.started_utc_ticks){continue}
    & "$env:SystemRoot\System32\taskkill.exe" /PID $candidate.ProcessId /T /F | Out-Null
    if($LASTEXITCODE -ne 0 -and (Get-Process -Id $candidate.ProcessId -ErrorAction SilentlyContinue)){throw 'Renderer could not be stopped'}
    if(-not $process.WaitForExit(10000)){throw 'Renderer did not exit'}
    $stopped+=[int]$candidate.ProcessId
}
@{stopped_pids=@($stopped)}|ConvertTo-Json -Compress
