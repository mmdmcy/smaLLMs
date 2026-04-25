$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptDir '.portui-runtime\portui.ps1') -ManifestDir (Join-Path $scriptDir 'portui') @args
exit $LASTEXITCODE