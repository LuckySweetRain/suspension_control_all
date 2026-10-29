param(
    [double]$Seconds = 10,
    [int]$SpawnIndex = 159,
    [string]$OrinHost = "",
    [string]$Town = "",
    [string]$UnknownContact = "stop",
    [string]$RoadProfile = "town",
    [string]$Planner = "",
    [double]$RoadCurvature = 0,
    [string]$NavCommand = "straight",
    [string]$Config = "",
    [switch]$NoVisualize,
    [switch]$Record,
    [switch]$Compare,
    [switch]$CheckOnly,
    [switch]$CloseOnComplete
)

# Compatibility wrapper: all validation/startup logic lives in scripts/run_cosim.py.
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
if ($Config) { $env:COSIM_CONFIG = (Resolve-Path -LiteralPath $Config).Path }
$configPath = if ($env:COSIM_CONFIG) { $env:COSIM_CONFIG } else { Join-Path $projectRoot "configs/cosim.local.json" }
$settings = if (Test-Path -LiteralPath $configPath) { Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json } else { $null }
$condaExe = if ($settings.windows_conda) { $settings.windows_conda } else { "conda" }
$environmentName = if ($settings.windows_environment) { $settings.windows_environment } else { "carla0911" }
$runArguments = @("run", "--no-capture-output", "-n", $environmentName, "python", (Join-Path $projectRoot "scripts/run_cosim.py"),
    "--seconds", $Seconds.ToString([cultureinfo]::InvariantCulture), "--spawn", $SpawnIndex,
    "--road-profile", $RoadProfile, "--road-curvature", $RoadCurvature.ToString([cultureinfo]::InvariantCulture),
    "--nav-command", $NavCommand, "--unknown-contact", $UnknownContact)
if ($OrinHost) { $runArguments += @("--orin-host", $OrinHost) }
if ($Town) { $runArguments += @("--town", $Town) }
if ($Planner) { $runArguments += @("--planner", $Planner) }
if ($NoVisualize) { $runArguments += "--no-visualize" }
if ($Record) { $runArguments += "--record" }
if ($Compare) { $runArguments += "--compare" }
if ($CheckOnly) { $runArguments += "--check-only" }
if ($CloseOnComplete) { $runArguments += "--close-on-complete" }
Push-Location -LiteralPath $projectRoot
try {
    & $condaExe @runArguments
    if ($LASTEXITCODE -ne 0) { throw "Co-simulation failed. See the error and results directory." }
} finally {
    Pop-Location
}
