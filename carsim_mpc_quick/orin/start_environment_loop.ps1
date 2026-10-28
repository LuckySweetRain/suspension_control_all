param(
    [ValidateRange(0.05, 120)][double]$Seconds = 10,
    [ValidateRange(0, 43.2)][double]$CruiseKph = 0,
    [int]$SpawnIndex = 159,
    [string]$OrinHost = "192.168.50.2",
    [string]$Town = "",
    [ValidateSet("stop", "record")][string]$UnknownContact = "stop",
    [ValidateSet("town", "wave", "bump", "pothole", "combined")][string]$RoadProfile = "town",
    [ValidateSet("", "qwen", "tcp", "transfuser")][string]$Planner = "",
    [ValidateRange(-0.01, 0.01)][double]$RoadCurvature = 0,
    [ValidateSet("straight", "left", "right")][string]$NavCommand = "straight",
    [switch]$NoVisualize,
    [switch]$Record,
    [switch]$Compare,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
if ($Record -and ($Planner -ne "tcp" -or $Compare)) { throw "Recording requires TCP single-car mode." }
if ($Planner -and $Compare) { throw "Driving model comparison is not implemented in the two-car viewer yet." }
if ($RoadCurvature -ne 0 -and (-not $Planner -or $Compare)) { throw "Curved road requires a driving planner in single-car mode." }
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$carlaExe = "E:/Desktop/workspace/CARLA_0.9.11/WindowsNoEditor/CarlaUE4.exe"
$condaCommand = Get-Command conda -ErrorAction Stop

# Validate native DLLs before creating CARLA actors or opening the Orin stream.
# In particular, legacy Pillow/libtiff needs libdeflate.dll (libdeflate 1.14).
$dependencyCheck = "import numpy; print('NumPy import OK')"
if (-not $NoVisualize) {
    $dependencyCheck += "; from PIL import Image,ImageTk; import tkinter; root=tkinter.Tk(); root.withdraw(); image=ImageTk.PhotoImage(Image.new('RGB',(8,8))); root.update(); root.destroy(); print('Pillow and Tk rendering OK')"
}
if (($Compare -or $Planner) -and $NoVisualize) { $dependencyCheck += "; from PIL import Image; print('Pillow import OK')" }
& $condaCommand run --no-capture-output -n carla0911 python -c $dependencyCheck
if ($LASTEXITCODE -ne 0) {
    throw "carla0911 dependency check failed. See orin/ENVIRONMENT_LOOP.md for the pinned environment and Pillow DLL repair."
}
if ($CheckOnly) { return }

function Test-TcpEndpoint([string]$Address, [int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect($Address, $Port, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne(1500)) { return $false }
        $client.EndConnect($pending)
        return $true
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

if (-not (Test-TcpEndpoint $OrinHost 55002)) {
    throw "Orin controller is unavailable at ${OrinHost}:55002. Check direct Ethernet and suspension-environment.service."
}
if ($Planner -and -not (Test-TcpEndpoint $OrinHost 55003)) {
    throw "Orin driving planner is unavailable at ${OrinHost}:55003. Run deploy_driving_orin.py --planner $Planner first."
}
if (-not (Test-TcpEndpoint "127.0.0.1" 2000)) {
    if (-not (Test-Path -LiteralPath $carlaExe)) { throw "CARLA executable not found: $carlaExe" }
    Start-Process -FilePath $carlaExe -WorkingDirectory (Split-Path $carlaExe) -WindowStyle Hidden -ArgumentList @(
        "-quality-level=Low", "-ResX=800", "-ResY=600", "-windowed", "-nosound", "-RenderOffScreen"
    )
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if (Test-TcpEndpoint "127.0.0.1" 2000) { $ready = $true; break }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) { throw "CARLA did not start. Inspect the CARLA application log." }
}

$runArguments = @(
    "run", "--no-capture-output", "-n", "carla0911", "python",
    "carsim_mpc_quick/orin/run_environment_loop.py", "--seconds", $Seconds.ToString([cultureinfo]::InvariantCulture),
    "--spawn", $SpawnIndex, "--orin-host", $OrinHost, "--road-profile", $RoadProfile
)
if (-not $NoVisualize) { $runArguments += "--visualize" }
if ($Record) { $runArguments += "--record" }
if ($Planner) { $runArguments += @("--planner", $Planner, "--nav-command", $NavCommand) }
if ($CruiseKph -gt 0) { $runArguments += @("--cruise-kph", $CruiseKph.ToString([cultureinfo]::InvariantCulture)) }
if ($Town) { $runArguments += @("--town", $Town) }
if (-not $Compare) { $runArguments += @("--unknown-contact", $UnknownContact) }
if ($RoadCurvature -ne 0) { $runArguments += @("--road-curvature", $RoadCurvature.ToString([cultureinfo]::InvariantCulture)) }
if ($Compare) {
    if ($RoadProfile -eq "town") { $RoadProfile = "combined" }
    $runArguments = @(
        "run", "--no-capture-output", "-n", "carla0911", "python",
        "carsim_mpc_quick/orin/run_comparison.py", "--seconds", $Seconds.ToString([cultureinfo]::InvariantCulture),
        "--orin-host", $OrinHost, "--road-profile", $RoadProfile
    )
    if ($NoVisualize) { $runArguments += "--no-visualize" }
}
Push-Location -LiteralPath $projectRoot
try {
    & $condaCommand @runArguments
    if ($LASTEXITCODE -ne 0) { throw "Simulation failed. Reports remain under results/carla_environment/ or results/carla_comparison/." }
} finally {
    Pop-Location
}
