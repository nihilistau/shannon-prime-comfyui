# launch_with_control.ps1
#
# Brings up ComfyUI on port 8189 and the Shannon-Prime control panel on
# port 8500 in one shot. Stops any previous instances cleanly first.
#
# Layout:
#   ComfyUI (the model server)         http://127.0.0.1:8189
#   sp-control-panel (the dashboard)   http://127.0.0.1:8500
#
# Logs:
#   C:\Projects\the_system_itself\comfyui-bench\comfyui_launch.log
#   C:\Projects\the_system_itself\comfyui-bench\sp_control_panel.log
#
# Usage:
#   .\scripts\launch_with_control.ps1
#   .\scripts\launch_with_control.ps1 -OpenBrowser

param(
    [switch]$OpenBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$BenchRoot     = "C:\Projects\the_system_itself\comfyui-bench"
$VenvPython    = "$BenchRoot\.venv\Scripts\python.exe"
$ComfyMain     = "$BenchRoot\main.py"
$ComfyLog      = "$BenchRoot\comfyui_launch.log"
$CtrlLog       = "$BenchRoot\sp_control_panel.log"
$CtrlScript    = "D:\F\shannon-prime-repos\scripts\sp_control_panel.py"
$ComfyPort     = 8189
$CtrlPort      = 8500

# ---------------------------------------------------------------------------
# Validate paths
# ---------------------------------------------------------------------------
foreach ($p in @($VenvPython, $ComfyMain, $CtrlScript)) {
    if (-not (Test-Path $p)) {
        Write-Error "Required path not found: $p"
        exit 1
    }
}

function Stop-PortListener {
    param([int]$Port, [string]$Name)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $procIds = @($conn.OwningProcess) | Where-Object { $_ -ne $null -and $_ -ne 0 } | Select-Object -Unique
        foreach ($procId in $procIds) {
            try {
                $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
                if ($proc) {
                    Write-Host "  Stopping $Name PID $procId on port $Port..."
                    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                }
            } catch {
                Write-Host "  PID $procId already gone."
            }
        }
        Start-Sleep -Seconds 2
    }
}

# ---------------------------------------------------------------------------
# Stop existing instances
# ---------------------------------------------------------------------------
Write-Host "Stopping previous instances..."
Stop-PortListener -Port $ComfyPort -Name "ComfyUI"
Stop-PortListener -Port $CtrlPort  -Name "sp-control-panel"

# ---------------------------------------------------------------------------
# Rotate logs
# ---------------------------------------------------------------------------
foreach ($logFile in @($ComfyLog, $CtrlLog)) {
    if (Test-Path $logFile) {
        Move-Item -Force $logFile "$logFile.prev" -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
# Start ComfyUI
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Starting ComfyUI on port $ComfyPort..."
$comfyProc = Start-Process `
    -FilePath $VenvPython `
    -ArgumentList @($ComfyMain, "--normalvram", "--disable-async-offload", "--port", $ComfyPort) `
    -WorkingDirectory $BenchRoot `
    -RedirectStandardOutput $ComfyLog `
    -RedirectStandardError "$ComfyLog.err" `
    -PassThru `
    -WindowStyle Hidden
Write-Host "  ComfyUI PID: $($comfyProc.Id)"

# ---------------------------------------------------------------------------
# Wait for ComfyUI to be ready (up to 90s)
# ---------------------------------------------------------------------------
Write-Host "  Waiting for ComfyUI to come online..."
$deadline = (Get-Date).AddSeconds(90)
$comfyReady = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    if ($comfyProc.HasExited) {
        Write-Host "FAIL - ComfyUI exited prematurely (code $($comfyProc.ExitCode))"
        Get-Content $ComfyLog -Tail 20 -ErrorAction SilentlyContinue
        exit 1
    }
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$ComfyPort/system_stats" `
                               -TimeoutSec 3 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $comfyReady = $true; break }
    } catch { }
}
if (-not $comfyReady) {
    Write-Host "WARNING - ComfyUI didn't respond in 90s. Check $ComfyLog. Continuing anyway."
} else {
    Write-Host "  ComfyUI online."
}

# ---------------------------------------------------------------------------
# Start the Shannon-Prime control panel
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Starting sp-control-panel on port $CtrlPort..."
$ctrlProc = Start-Process `
    -FilePath $VenvPython `
    -ArgumentList @($CtrlScript, "--port", $CtrlPort, "--comfy-host", "127.0.0.1:$ComfyPort") `
    -WorkingDirectory $BenchRoot `
    -RedirectStandardOutput $CtrlLog `
    -RedirectStandardError "$CtrlLog.err" `
    -PassThru `
    -WindowStyle Hidden
Write-Host "  sp-control-panel PID: $($ctrlProc.Id)"

# Wait briefly for the control panel
Start-Sleep -Seconds 2
$ctrlReady = $false
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$CtrlPort/" -TimeoutSec 3 -UseBasicParsing
    if ($r.StatusCode -eq 200) { $ctrlReady = $true }
} catch { }
if ($ctrlReady) {
    Write-Host "  sp-control-panel online."
} else {
    Write-Host "WARNING - control panel didn't respond. Check $CtrlLog"
}

# ---------------------------------------------------------------------------
# Summary + optional browser
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "==========================================================="
Write-Host " ComfyUI:           http://127.0.0.1:$ComfyPort  (PID $($comfyProc.Id))"
Write-Host " Control panel:     http://127.0.0.1:$CtrlPort  (PID $($ctrlProc.Id))"
Write-Host ""
Write-Host " Logs:"
Write-Host "   ComfyUI:         $ComfyLog"
Write-Host "   Control panel:   $CtrlLog"
Write-Host ""
Write-Host " To stop:           Stop-Process -Id $($comfyProc.Id), $($ctrlProc.Id) -Force"
Write-Host "==========================================================="

if ($OpenBrowser) {
    Start-Process "http://127.0.0.1:$CtrlPort"
}
