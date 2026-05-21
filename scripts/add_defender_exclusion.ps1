#requires -Version 5.1
<#
.SYNOPSIS
    Deprecated shim — calls add_av_exclusion.ps1.

.DESCRIPTION
    The previous Defender-only helper has been superseded by
    add_av_exclusion.ps1, which detects whichever antivirus product is
    actually active (Defender, Symantec Endpoint Protection, CrowdStrike,
    SentinelOne, Carbon Black, etc.) and either adds the exclusion
    directly (Defender) or emits a copy-pasteable IT request for the
    centrally-managed products.

    This shim is kept so existing docs / muscle memory still work.
#>

param(
    [string]$CachePath = (Join-Path $env:LOCALAPPDATA "OMRChecker\cache")
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $here 'add_av_exclusion.ps1') -CachePath $CachePath
