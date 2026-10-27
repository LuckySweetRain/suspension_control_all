param(
    [ValidateRange(0.05, 30)][double]$Seconds = 10,
    [int]$SpawnIndex = 159,
    [string]$OrinHost = "192.168.50.2",
    [ValidateSet("town", "wave", "bump", "pothole", "combined")][string]$RoadProfile = "town",
    [switch]$NoVisualize,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$carlaExe = "E:/Desktop/workspace/CARLA_0.9.11/WindowsNoEditor/CarlaUE4.exe"
$condaCommand = Get-Command conda -ErrorAction Stop

# Validate native DLLs before creating CARLA actors or opening the Orin stream.
# In particular, legacy Pillow/libtiff needs libdeflate.dll (libdeflate 1.14).
$dependencyCheck = "import numpy; print('NumPy import OK')"
if (-not $NoVisualize) {
    $dependencyCheck += "; from PIL import Image,ImageTk; import tkinter; root=tkinter.Tk(); root.withdraw(); image=ImageTk.PhotoImage(Image.new('RGB',(8,8))); root.update(); root.destroy(); print('Pillow and Tk rendering OK')"
}
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
Push-Location -LiteralPath $projectRoot
try {
    & $condaCommand @runArguments
    if ($LASTEXITCODE -ne 0) { throw "Simulation failed. The report and completed trace remain under results/carla_environment/." }
} finally {
    Pop-Location
}
