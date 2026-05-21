#requires -Version 5.1
<#
.SYNOPSIS
    Add a Microsoft Defender exclusion for the OMRChecker scratch cache.

.DESCRIPTION
    OMRChecker stages its transient runtime files (per-worker dirs, rotated
    image cache, worker outputs) under %LOCALAPPDATA%\OMRChecker\cache by
    default. When Microsoft Defender's Real-Time Protection scans every one
    of those files on close, throughput for batches of 1000+ pages drops
    by 2-10x.

    Adding ONE path exclusion for this cache directory eliminates almost
    all of that overhead while keeping Defender enabled for the rest of
    the system.

    This script requires an elevated PowerShell session (Run as
    Administrator) because Set-MpPreference and Add-MpPreference touch
    Defender configuration. If launched without admin rights it will
    re-launch itself via Start-Process -Verb RunAs.

.PARAMETER CachePath
    Override the cache path. Defaults to %LOCALAPPDATA%\OMRChecker\cache.

.PARAMETER WhatIf
    Show what would change without actually modifying Defender settings.

.EXAMPLE
    .\scripts\add_defender_exclusion.ps1

.EXAMPLE
    .\scripts\add_defender_exclusion.ps1 -CachePath "D:\omr-cache"

.NOTES
    Safe to run repeatedly: Add-MpPreference -ExclusionPath is idempotent.
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

if (-not (Test-IsAdmin)) {
    Write-Host "Re-launching with administrator privileges..." -ForegroundColor Yellow
    $argsList = @("-NoProfile", "-ExecutionPolicy", "Bypass",
                  "-File", $MyInvocation.MyCommand.Path,
                  "-CachePath", $CachePath)
    Start-Process powershell.exe -ArgumentList $argsList -Verb RunAs
    exit
}

if (-not (Test-Path $CachePath)) {
    New-Item -ItemType Directory -Path $CachePath -Force | Out-Null
    Write-Host "Created cache directory: $CachePath" -ForegroundColor Green
}

try {
    $pref = Get-MpPreference -ErrorAction Stop
} catch {
    Write-Error "Could not read Defender preferences. Are you on Windows with Defender installed?"
    exit 1
}

$existing = @($pref.ExclusionPath)
$normalised = [System.IO.Path]::GetFullPath($CachePath).TrimEnd('\')
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
    Write-Host ""
    Write-Host "OMRChecker will now route all transient runtime files through"
    Write-Host "this path, avoiding on-access scans for 5000+ page batches."
}

Write-Host ""
Write-Host "Current OMRChecker-related exclusions:" -ForegroundColor Cyan
(Get-MpPreference).ExclusionPath | Where-Object { $_ -like "*OMRChecker*" -or $_ -ieq $normalised }
