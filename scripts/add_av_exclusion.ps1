#requires -Version 5.1
<#
.SYNOPSIS
    Configure (or request) an antivirus exclusion for the OMRChecker scratch cache.

    All non-ASCII characters are intentionally avoided in this file so it
    works under PowerShell's default ANSI loader without a UTF-8 BOM.

.DESCRIPTION
    OMRChecker stages its transient runtime files (per-worker dirs, rotated
    image cache, worker outputs) under %LOCALAPPDATA%\OMRChecker\cache by
    default. On-access antivirus scanning of every transient file drops
    throughput for 1000+ page batches by 2-10x.

    This script tries to add a path exclusion using whatever antivirus the
    machine is actually running:

      * Microsoft Defender                    -> Add-MpPreference (admin)
      * Symantec Endpoint Protection (SEP)    -> centrally managed; emits
                                                 an IT request template
      * CrowdStrike / SentinelOne / Carbon
        Black / Sophos / McAfee / Trend Micro -> centrally managed; emits
                                                 an IT request template
      * No active AV detected                 -> no-op, prints info

    For centrally-managed AV the script CANNOT add the exclusion itself.
    Instead, it prints a copy-pasteable request describing exactly which
    path needs to be excluded and why, ready to send to your IT team.

.PARAMETER CachePath
    Override the cache path. Defaults to %LOCALAPPDATA%\OMRChecker\cache.

.EXAMPLE
    .\scripts\add_av_exclusion.ps1

.EXAMPLE
    .\scripts\add_av_exclusion.ps1 -CachePath "D:\omr-cache"

.NOTES
    Safe to run repeatedly. The Defender path is idempotent. The
    "ask IT" output is informational and never modifies the AV config.
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$CachePath = (Join-Path $env:LOCALAPPDATA "OMRChecker\cache")
)

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-RegisteredAntivirusProducts {
    try {
        $products = Get-CimInstance -Namespace 'root\SecurityCenter2' `
            -ClassName 'AntiVirusProduct' -ErrorAction Stop
        return $products
    } catch {
        return @()
    }
}

function Test-DefenderRealtimeActive {
    try {
        $svc = Get-Service -Name WinDefend -ErrorAction Stop
        if ($svc.Status -ne 'Running') { return $false }
        $pref = Get-MpPreference -ErrorAction Stop
        return (-not $pref.DisableRealtimeMonitoring)
    } catch {
        return $false
    }
}

# Always ensure the cache dir exists, even when the rest of the script
# can only print informational output. The Python code creates this
# directory lazily, but seeding it now makes the IT request immediately
# valid.
if (-not (Test-Path $CachePath)) {
    New-Item -ItemType Directory -Path $CachePath -Force | Out-Null
    Write-Host "Created cache directory: $CachePath" -ForegroundColor Green
}

$normalised = [System.IO.Path]::GetFullPath($CachePath).TrimEnd('\')

Write-Host ""
Write-Host "Detecting active antivirus product..." -ForegroundColor Cyan
$products = Get-RegisteredAntivirusProducts
$defenderActive = Test-DefenderRealtimeActive

if ($products) {
    foreach ($p in $products) {
        Write-Host ("  found: {0}" -f $p.displayName)
    }
} else {
    Write-Host "  (no products returned by SecurityCenter2)"
}

$primary = $null
if ($products) {
    # Prefer non-Defender entries when more than one is registered, because
    # Defender stops itself when a third-party AV takes over.
    $nonDefender = $products | Where-Object { $_.displayName -notmatch 'Windows Defender' }
    if ($nonDefender) {
        $primary = $nonDefender | Select-Object -First 1
    } else {
        $primary = $products | Select-Object -First 1
    }
}

$primaryName = if ($primary) { $primary.displayName } else { '(unknown)' }
Write-Host ("primary AV: {0}" -f $primaryName) -ForegroundColor Cyan
Write-Host ("Defender real-time: {0}" -f ($defenderActive)) -ForegroundColor Cyan
Write-Host ""

# --- Defender path ---------------------------------------------------------
if ($defenderActive) {
    if (-not (Test-IsAdmin)) {
        Write-Host "Re-launching with administrator privileges..." -ForegroundColor Yellow
        $argsList = @("-NoProfile", "-ExecutionPolicy", "Bypass",
                      "-File", $MyInvocation.MyCommand.Path,
                      "-CachePath", $CachePath)
        Start-Process powershell.exe -ArgumentList $argsList -Verb RunAs
        exit
    }

    $pref = Get-MpPreference
    $existing = @($pref.ExclusionPath)
    $already = $existing | Where-Object {
        $_ -and ([System.IO.Path]::GetFullPath($_).TrimEnd('\') -ieq $normalised)
    }

    if ($already) {
        Write-Host "Defender exclusion already configured for:" -ForegroundColor Green
        Write-Host "  $normalised"
        exit 0
    }

    if ($PSCmdlet.ShouldProcess($normalised, "Add Defender exclusion path")) {
        Add-MpPreference -ExclusionPath $normalised
        Write-Host "[OK] Added Defender exclusion for:" -ForegroundColor Green
        Write-Host "  $normalised"
    }

    Write-Host ""
    Write-Host "Current OMRChecker-related Defender exclusions:" -ForegroundColor Cyan
    (Get-MpPreference).ExclusionPath |
        Where-Object { $_ -like "*OMRChecker*" -or $_ -ieq $normalised }
    exit 0
}

# --- Centrally-managed AV path --------------------------------------------
# Defender is off (or stopped because another AV took over). We cannot
# touch the third-party AV from a user-level script. Emit an IT request
# template tailored to the detected product where possible.

$avHint = ''
switch -Regex ($primaryName) {
    'Symantec'      { $avHint = @"
Symantec Endpoint Protection (SEP) - exclusions are centrally managed via
the Symantec Endpoint Protection Manager (SEPM) console:

  1. In SEPM, open the Exceptions policy that applies to this client.
  2. Add an "Exception" of type "Folder".
  3. Path: $normalised
  4. Scope: "Auto-Protect, Scheduled and On-Demand Scans" (and also
     SONAR / Behavior Monitoring if a separate exception type is offered).
  5. Apply / push policy update.
"@ }
    'CrowdStrike'   { $avHint = "CrowdStrike Falcon - ask your IT admin to add an ML/IOA/IOC exclusion for the path in the Falcon console under Configuration > Prevention Policies > Exclusions." }
    'SentinelOne'   { $avHint = "SentinelOne - ask your IT admin to add a Path exclusion (mode: Suppress) in the SentinelOne management console under Sentinels > Exclusions." }
    'Carbon Black'  { $avHint = "VMware Carbon Black - ask your IT admin to add a path exclusion in the EDR/Cloud console under Enforce > Policies > Exclusions." }
    'Sophos'        { $avHint = "Sophos - ask your IT admin to add a scanning exclusion under Sophos Central > Endpoint Settings > Global Exclusions." }
    'McAfee'        { $avHint = "McAfee / Trellix - ask your IT admin to add an On-Access Scan exclusion in ePO under Policy Catalog > Endpoint Security > On-Access Scan > Exclusions." }
    'Trend'         { $avHint = "Trend Micro - ask your IT admin to add a real-time scan exclusion in Apex One / OfficeScan under Policies > Real-Time Scan Settings > Scan Exclusion List." }
    default         { $avHint = "Your AV is managed by IT and cannot be modified from this user account. Forward the request below to your IT/security team." }
}

Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
Write-Host " IT REQUEST TEMPLATE - copy and send to your IT/security team" -ForegroundColor Yellow
Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
Write-Host ""

$user = "$env:USERDOMAIN\$env:USERNAME"
$host_ = [System.Environment]::MachineName
$body = @"
Subject: OMRChecker scratch-folder exclusion request ($host_)

Hi IT team,

I'm running an internal OMR (Optical Mark Recognition) batch processor that
needs to render PDF pages into images and run them through the OMR engine.
For batches of 1000+ pages the on-access antivirus scan of transient
working files dominates the runtime (we see 2-10x slowdowns).

Could you please add a real-time scan exclusion for the following path?

    Machine : $host_
    User    : $user
    Path    : $normalised
    Type    : Folder (recursive)
    AV      : $primaryName

Notes:
  * This directory only contains transient files generated and consumed by
    the OMR pipeline (per-worker runtime dirs, rotated image cache, per-image
    worker output dirs). No user documents are stored here.
  * The directory is auto-cleared between runs.
  * It lives under %LOCALAPPDATA% (per-user), not under %ProgramData% or any
    system-wide path.
  * All other paths can keep their normal real-time protection.

$avHint

Thanks!
"@

Write-Host $body
Write-Host ""
Write-Host "Cache path that needs to be excluded:" -ForegroundColor Cyan
Write-Host "  $normalised"
Write-Host ""
Write-Host "Note: the OMR code already runs WITHOUT this exclusion - it just" -ForegroundColor Gray
Write-Host "      uses the in-memory pipeline and JPEG outputs to minimise"        -ForegroundColor Gray
Write-Host "      the impact. Adding the exclusion is purely a performance win."   -ForegroundColor Gray
