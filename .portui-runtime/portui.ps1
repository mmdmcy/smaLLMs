param(
    [string]$ManifestDir,
    [string]$WorkspaceDir,
    [string]$Project,
    [string]$InstallProject,
    [string]$InitProject,
    [switch]$ListProjects,
    [switch]$List,
    [string]$Run,
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:Variables = @{}
$script:CurrentManifestDir = ''
$script:CurrentWorkspaceDir = ''
$script:CurrentProjectDir = ''
$script:CurrentProjectId = ''
$script:CurrentManifestName = 'PortUI'
$script:CurrentManifestDescription = ''
$script:CurrentActions = @()

function Show-Usage {
    @'
Usage: .\portui.ps1 [-ManifestDir DIR | -WorkspaceDir DIR] [-Project ID] [-ListProjects] [-List] [-Run ACTION_ID]

PortUI opens project-local terminal menus from portui\ or .portui\ manifests.
It does not build executables; the launcher scripts are the portable entrypoints.

Options:
  -ManifestDir DIR   Path to one PortUI manifest directory.
  -WorkspaceDir DIR  Path to a workspace containing project manifests in repo\portui or repo\.portui.
  -Project ID        Project id inside workspace mode.
  -InstallProject DIR
                    Install or update project-local PortUI runtime files in a repo that already has portui\ or .portui\.
  -InitProject DIR   Create a starter PortUI app in a repo, then install the project-local runtime.
  -ListProjects      Print discovered workspace projects and exit.
  -List              Print actions for the selected manifest or project and exit.
  -Run ACTION_ID     Run a specific action non-interactively.
  -Help              Show this help.
'@
}

function Set-PortUIVariable {
    param(
        [string]$Name,
        [string]$Value
    )

    if ($Name -notmatch '^[A-Za-z0-9_]+$') {
        throw "Invalid variable name: $Name"
    }

    $script:Variables[$Name] = $Value
}

function Expand-PortUIText {
    param(
        [string]$Text
    )

    if ([string]::IsNullOrEmpty($Text)) {
        return $Text
    }

    $expanded = $Text
    for ($pass = 0; $pass -lt 8; $pass++) {
        $changed = $false
        foreach ($entry in $script:Variables.GetEnumerator()) {
            $token = '{{' + $entry.Key + '}}'
            $updated = $expanded.Replace($token, [string]$entry.Value)
            if ($updated -ne $expanded) {
                $expanded = $updated
                $changed = $true
            }
        }

        if (-not $changed) {
            break
        }
    }

    return $expanded
}

function Get-KeyValueData {
    param(
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing file: $Path"
    }

    $map = @{}
    foreach ($rawLine in [System.IO.File]::ReadAllLines($Path)) {
        $line = $rawLine.TrimEnd("`r")
        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith('#')) {
            continue
        }

        $splitIndex = $line.IndexOf('=')
        if ($splitIndex -lt 0) {
            continue
        }

        $key = $line.Substring(0, $splitIndex)
        $value = $line.Substring($splitIndex + 1)
        $map[$key] = $value
    }

    return $map
}

function Get-HostOSName {
    if ($env:OS -eq 'Windows_NT') { return 'windows' }

    try {
        $runtime = [System.Runtime.InteropServices.RuntimeInformation]
        if ($runtime::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::OSX)) { return 'macos' }
        if ($runtime::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)) { return 'linux' }
    } catch {
    }

    return 'unknown'
}

function Test-PortUITruthy {
    param(
        [string]$Value
    )

    return $Value -match '^(?i:1|true|yes|on)$'
}

function Get-ResolvedDirectory {
    param(
        [string]$Path
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw 'A directory path is required.'
    }

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Missing directory: $Path"
    }

    return (Resolve-Path -LiteralPath $Path).Path
}

function Get-LocalManifestDirectory {
    foreach ($candidate in @(
        (Join-Path $PSScriptRoot '.portui'),
        (Join-Path $PSScriptRoot 'portui')
    )) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'manifest.env')) {
            return $candidate
        }
    }

    return $null
}

function Get-ProjectManifestDirectoryInRepo {
    param(
        [string]$ResolvedProjectDir
    )

    $portuiDir = Join-Path $ResolvedProjectDir 'portui'
    $hiddenDir = Join-Path $ResolvedProjectDir '.portui'
    if (Test-Path -LiteralPath (Join-Path $portuiDir 'manifest.env')) {
        return $portuiDir
    }
    if (Test-Path -LiteralPath (Join-Path $hiddenDir 'manifest.env')) {
        return $hiddenDir
    }

    throw "Project does not contain portui\manifest.env or .portui\manifest.env: $ResolvedProjectDir"
}

function Write-TextFileNoBom {
    param(
        [string]$Path,
        [string]$Content
    )

    $directory = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($directory)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Install-PortUIProject {
    param(
        [string]$ProjectDir
    )

    $resolvedProjectDir = Get-ResolvedDirectory -Path $ProjectDir
    $manifestDir = Get-ProjectManifestDirectoryInRepo -ResolvedProjectDir $resolvedProjectDir
    $manifestLeaf = Split-Path -Leaf $manifestDir
    $runtimeDir = Join-Path $resolvedProjectDir '.portui-runtime'

    New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null

    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'portui.sh') -Destination (Join-Path $runtimeDir 'portui.sh') -Force
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'portui.ps1') -Destination (Join-Path $runtimeDir 'portui.ps1') -Force
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'portui.cmd') -Destination (Join-Path $runtimeDir 'portui.cmd') -Force
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'VERSION')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'VERSION') -Destination (Join-Path $runtimeDir 'VERSION') -Force
    }

    $shimSh = (@'
#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$SCRIPT_DIR/.portui-runtime/portui.sh" --manifest-dir "$SCRIPT_DIR/{0}" "$@"
'@) -f $manifestLeaf
    $shimPs1 = (@'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $scriptDir '.portui-runtime\portui.ps1') -ManifestDir (Join-Path $scriptDir '{0}') @args
exit $LASTEXITCODE
'@) -f $manifestLeaf
    $shimCmd = (@'
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0\.portui-runtime\portui.ps1" -ManifestDir "%~dp0{0}" %*
'@) -f $manifestLeaf

    Write-TextFileNoBom -Path (Join-Path $resolvedProjectDir 'portui.sh') -Content $shimSh
    Write-TextFileNoBom -Path (Join-Path $resolvedProjectDir 'portui.ps1') -Content $shimPs1
    Write-TextFileNoBom -Path (Join-Path $resolvedProjectDir 'portui.cmd') -Content $shimCmd

    Write-Host "Installed PortUI runtime into $resolvedProjectDir"
    Write-Host "Manifest: $manifestDir"
    Write-Host 'Run from the project root with ./portui.sh, .\portui.ps1, or portui.cmd'
}

function Initialize-PortUIProjectFiles {
    param(
        [string]$ResolvedProjectDir
    )

    $projectName = Split-Path -Leaf $ResolvedProjectDir
    $manifestDir = Join-Path $ResolvedProjectDir 'portui'
    $actionsDir = Join-Path $manifestDir 'actions'

    if (
        (Test-Path -LiteralPath (Join-Path $ResolvedProjectDir 'portui\manifest.env')) -or
        (Test-Path -LiteralPath (Join-Path $ResolvedProjectDir '.portui\manifest.env'))
    ) {
        throw "Project already has a PortUI app definition: $ResolvedProjectDir"
    }

    New-Item -ItemType Directory -Path $actionsDir -Force | Out-Null

    Write-TextFileNoBom -Path (Join-Path $manifestDir 'manifest.env') -Content @"
NAME=$projectName PortUI
DESCRIPTION=Starter PortUI app for $projectName.
VARIABLE_repo={{projectDir}}
"@

    Write-TextFileNoBom -Path (Join-Path $actionsDir '01-doctor.env') -Content @"
ID=doctor
TITLE=Doctor
DESCRIPTION=Print the current project, workspace, and OS values.
TIMEOUT_SECONDS=20
CWD={{projectDir}}
POSIX_PROGRAM=sh
POSIX_ARGS=-c|printf '%s\n' 'project={{projectId}}' 'workspace={{workspaceDir}}' 'os={{os}}'
WINDOWS_PROGRAM=powershell
WINDOWS_ARGS=-NoProfile|-Command|Write-Output 'project={{projectId}}'; Write-Output 'workspace={{workspaceDir}}'; Write-Output 'os={{os}}'
"@

    Write-TextFileNoBom -Path (Join-Path $actionsDir '02-list-files.env') -Content @"
ID=list-files
TITLE=List Files
DESCRIPTION=List the files in the project root.
TIMEOUT_SECONDS=20
CWD={{projectDir}}
POSIX_PROGRAM=ls
POSIX_ARGS=-la|.
WINDOWS_PROGRAM=powershell
WINDOWS_ARGS=-NoProfile|-Command|Get-ChildItem -Force .
"@

    Write-TextFileNoBom -Path (Join-Path $actionsDir '03-git-status.env') -Content @"
ID=git-status
TITLE=Git Status
DESCRIPTION=Show a compact git status when the project is a git repository.
TIMEOUT_SECONDS=30
CWD={{projectDir}}
PROGRAM=git
ARGS=status|--short|--branch
"@
}

function Initialize-AndInstallPortUIProject {
    param(
        [string]$ProjectDir
    )

    $resolvedProjectDir = Get-ResolvedDirectory -Path $ProjectDir
    Initialize-PortUIProjectFiles -ResolvedProjectDir $resolvedProjectDir
    Install-PortUIProject -ProjectDir $resolvedProjectDir
    Write-Host "Created starter PortUI app in $resolvedProjectDir\portui"
}

function Get-ProjectDirectoryFromManifestDir {
    param(
        [string]$ResolvedManifestDir
    )

    $leaf = Split-Path -Leaf $ResolvedManifestDir
    if ($leaf -in @('portui', '.portui')) {
        return Split-Path -Parent $ResolvedManifestDir
    }

    return $ResolvedManifestDir
}

function Get-ProjectIdFromManifestDir {
    param(
        [string]$ResolvedManifestDir
    )

    return Split-Path -Leaf (Get-ProjectDirectoryFromManifestDir -ResolvedManifestDir $ResolvedManifestDir)
}

function Get-ManifestSummary {
    param(
        [string]$ResolvedManifestDir
    )

    $summary = @{
        ProjectId = Get-ProjectIdFromManifestDir -ResolvedManifestDir $ResolvedManifestDir
        Name = Get-ProjectIdFromManifestDir -ResolvedManifestDir $ResolvedManifestDir
        Description = ''
        ManifestDir = $ResolvedManifestDir
        ProjectDir = Get-ProjectDirectoryFromManifestDir -ResolvedManifestDir $ResolvedManifestDir
    }

    $manifestPath = Join-Path $ResolvedManifestDir 'manifest.env'
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        return $summary
    }

    $data = Get-KeyValueData -Path $manifestPath
    if ($data.ContainsKey('NAME') -and -not [string]::IsNullOrWhiteSpace($data.NAME)) {
        $summary.Name = $data.NAME
    }
    if ($data.ContainsKey('DESCRIPTION')) {
        $summary.Description = $data.DESCRIPTION
    }

    return $summary
}

function Reset-PortUIVariables {
    $script:Variables = @{}
}

function Initialize-Builtins {
    param(
        [string]$ResolvedManifestDir,
        [string]$ResolvedWorkspaceDir
    )

    $homeDir = [Environment]::GetFolderPath('UserProfile')
    $cwd = (Get-Location).Path
    $osName = Get-HostOSName
    $projectDir = Get-ProjectDirectoryFromManifestDir -ResolvedManifestDir $ResolvedManifestDir
    $projectId = Get-ProjectIdFromManifestDir -ResolvedManifestDir $ResolvedManifestDir

    if ([string]::IsNullOrWhiteSpace($ResolvedWorkspaceDir)) {
        $ResolvedWorkspaceDir = Split-Path -Parent $projectDir
    }

    $script:CurrentManifestDir = $ResolvedManifestDir
    $script:CurrentWorkspaceDir = $ResolvedWorkspaceDir
    $script:CurrentProjectDir = $projectDir
    $script:CurrentProjectId = $projectId

    Set-PortUIVariable -Name 'home' -Value $homeDir
    Set-PortUIVariable -Name 'cwd' -Value $cwd
    Set-PortUIVariable -Name 'os' -Value $osName
    Set-PortUIVariable -Name 'manifestDir' -Value $ResolvedManifestDir
    Set-PortUIVariable -Name 'projectDir' -Value $projectDir
    Set-PortUIVariable -Name 'projectId' -Value $projectId
    Set-PortUIVariable -Name 'workspaceDir' -Value $ResolvedWorkspaceDir

    if ($osName -eq 'windows') {
        Set-PortUIVariable -Name 'pathSep' -Value '\'
        Set-PortUIVariable -Name 'listSep' -Value ';'
        Set-PortUIVariable -Name 'exeSuffix' -Value '.exe'
    } else {
        Set-PortUIVariable -Name 'pathSep' -Value '/'
        Set-PortUIVariable -Name 'listSep' -Value ':'
        Set-PortUIVariable -Name 'exeSuffix' -Value ''
    }
}

function Load-Manifest {
    param(
        [string]$ResolvedManifestDir
    )

    $manifestPath = Join-Path $ResolvedManifestDir 'manifest.env'
    $data = Get-KeyValueData -Path $manifestPath
    $manifest = @{
        Name = $script:CurrentProjectId
        Description = ''
    }

    foreach ($entry in $data.GetEnumerator()) {
        switch -Wildcard ($entry.Key) {
            'NAME' { $manifest.Name = $entry.Value }
            'DESCRIPTION' { $manifest.Description = $entry.Value }
            'VARIABLE_*' {
                $variableName = $entry.Key.Substring('VARIABLE_'.Length)
                Set-PortUIVariable -Name $variableName -Value $entry.Value
            }
        }
    }

    $keys = @($script:Variables.Keys)
    foreach ($key in $keys) {
        Set-PortUIVariable -Name $key -Value (Expand-PortUIText $script:Variables[$key])
    }

    $script:CurrentManifestName = $manifest.Name
    $script:CurrentManifestDescription = $manifest.Description
    return $manifest
}

function New-Variant {
    return @{
        Program = ''
        Args = ''
        Cwd = ''
        Env = @{}
    }
}

function Load-Action {
    param(
        [string]$Path
    )

    $action = @{
        ID = ''
        Title = ''
        Description = ''
        TimeoutSeconds = 30
        Interactive = $false
        Base = New-Variant
        Posix = New-Variant
        Linux = New-Variant
        MacOS = New-Variant
        Windows = New-Variant
        Path = $Path
    }

    $data = Get-KeyValueData -Path $Path
    foreach ($entry in $data.GetEnumerator()) {
        $key = $entry.Key
        $value = $entry.Value

        switch -Wildcard ($key) {
            'ID' { $action.ID = $value }
            'TITLE' { $action.Title = $value }
            'DESCRIPTION' { $action.Description = $value }
            'TIMEOUT_SECONDS' {
                $parsed = 0
                if ([int]::TryParse($value, [ref]$parsed) -and $parsed -ge 0) {
                    $action.TimeoutSeconds = $parsed
                }
            }
            'INTERACTIVE' { $action.Interactive = Test-PortUITruthy -Value $value }
            'PROGRAM' { $action.Base.Program = $value }
            'ARGS' { $action.Base.Args = $value }
            'CWD' { $action.Base.Cwd = $value }
            'ENV_*' { $action.Base.Env[$key.Substring(4)] = $value }
            'POSIX_PROGRAM' { $action.Posix.Program = $value }
            'POSIX_ARGS' { $action.Posix.Args = $value }
            'POSIX_CWD' { $action.Posix.Cwd = $value }
            'POSIX_ENV_*' { $action.Posix.Env[$key.Substring(10)] = $value }
            'LINUX_PROGRAM' { $action.Linux.Program = $value }
            'LINUX_ARGS' { $action.Linux.Args = $value }
            'LINUX_CWD' { $action.Linux.Cwd = $value }
            'LINUX_ENV_*' { $action.Linux.Env[$key.Substring(10)] = $value }
            'MACOS_PROGRAM' { $action.MacOS.Program = $value }
            'MACOS_ARGS' { $action.MacOS.Args = $value }
            'MACOS_CWD' { $action.MacOS.Cwd = $value }
            'MACOS_ENV_*' { $action.MacOS.Env[$key.Substring(10)] = $value }
            'WINDOWS_PROGRAM' { $action.Windows.Program = $value }
            'WINDOWS_ARGS' { $action.Windows.Args = $value }
            'WINDOWS_CWD' { $action.Windows.Cwd = $value }
            'WINDOWS_ENV_*' { $action.Windows.Env[$key.Substring(12)] = $value }
        }
    }

    if ([string]::IsNullOrWhiteSpace($action.ID)) {
        throw "Action file is missing ID: $Path"
    }
    if ([string]::IsNullOrWhiteSpace($action.Title)) {
        $action.Title = $action.ID
    }

    return $action
}

function Get-ActionFiles {
    param(
        [string]$ResolvedManifestDir
    )

    $actionsDir = Join-Path $ResolvedManifestDir 'actions'
    if (-not (Test-Path -LiteralPath $actionsDir)) {
        throw "Missing actions directory: $actionsDir"
    }

    return Get-ChildItem -LiteralPath $actionsDir -Filter '*.env' -File | Sort-Object Name
}

function Get-ProjectManifestDirs {
    param(
        [string]$ResolvedWorkspaceDir
    )

    $results = @()
    foreach ($candidate in Get-ChildItem -LiteralPath $ResolvedWorkspaceDir -Directory -Force) {
        foreach ($manifestName in @('portui', '.portui')) {
            $manifestDir = Join-Path $candidate.FullName $manifestName
            $manifestFile = Join-Path $manifestDir 'manifest.env'
            if (Test-Path -LiteralPath $manifestFile) {
                $results += (Resolve-Path -LiteralPath $manifestDir).Path
            }
        }
    }

    return @($results | Sort-Object -Unique)
}

function Merge-Variant {
    param(
        [hashtable]$Resolved,
        [hashtable]$Variant,
        [string]$Label
    )

    $hasChanges = $false

    if (-not [string]::IsNullOrWhiteSpace($Variant.Program)) {
        $Resolved.Program = $Variant.Program
        $hasChanges = $true
    }
    if (-not [string]::IsNullOrWhiteSpace($Variant.Args)) {
        $Resolved.Args = $Variant.Args
        $hasChanges = $true
    }
    if (-not [string]::IsNullOrWhiteSpace($Variant.Cwd)) {
        $Resolved.Cwd = $Variant.Cwd
        $hasChanges = $true
    }
    foreach ($key in $Variant.Env.Keys) {
        $Resolved.Env[$key] = $Variant.Env[$key]
        $hasChanges = $true
    }

    if ($hasChanges) {
        $Resolved.Source += " -> $Label"
    }
}

function Resolve-Action {
    param(
        [hashtable]$Action
    )

    $osName = Get-HostOSName
    $resolved = @{
        Program = $Action.Base.Program
        Args = $Action.Base.Args
        Cwd = $Action.Base.Cwd
        Env = @{}
        Source = 'base'
    }

    foreach ($key in $Action.Base.Env.Keys) {
        $resolved.Env[$key] = $Action.Base.Env[$key]
    }

    if ($osName -ne 'windows') {
        Merge-Variant -Resolved $resolved -Variant $Action.Posix -Label 'posix'
    }

    switch ($osName) {
        'linux' { Merge-Variant -Resolved $resolved -Variant $Action.Linux -Label 'linux' }
        'macos' { Merge-Variant -Resolved $resolved -Variant $Action.MacOS -Label 'macos' }
        'windows' { Merge-Variant -Resolved $resolved -Variant $Action.Windows -Label 'windows' }
    }

    if ([string]::IsNullOrWhiteSpace($resolved.Program)) {
        throw "Action $($Action.ID) does not resolve to a runnable program on $osName"
    }

    $resolved.Program = Expand-PortUIText $resolved.Program
    $resolved.Args = Expand-PortUIText $resolved.Args
    if ([string]::IsNullOrWhiteSpace($resolved.Cwd)) {
        $resolved.Cwd = (Get-Location).Path
    } else {
        $resolved.Cwd = Expand-PortUIText $resolved.Cwd
    }

    $expandedEnv = @{}
    foreach ($key in $resolved.Env.Keys) {
        $expandedEnv[$key] = Expand-PortUIText $resolved.Env[$key]
    }
    $resolved.Env = $expandedEnv

    return $resolved
}

function Split-Args {
    param(
        [string]$ArgsString
    )

    if ([string]::IsNullOrWhiteSpace($ArgsString)) {
        return @()
    }

    return @($ArgsString.Split('|'))
}

function Quote-CommandPart {
    param(
        [string]$Value
    )

    if ([string]::IsNullOrEmpty($Value)) {
        return '""'
    }

    if ($Value -match '[^A-Za-z0-9_\./:=+\-]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }

    return $Value
}

function Format-DisplayCommand {
    param(
        [hashtable]$Resolved
    )

    $parts = @((Quote-CommandPart $Resolved.Program))
    foreach ($arg in (Split-Args $Resolved.Args)) {
        $parts += Quote-CommandPart $arg
    }
    return ($parts -join ' ')
}

function Build-ArgumentString {
    param(
        [string[]]$ArgList
    )

    $parts = @()
    foreach ($arg in $ArgList) {
        if ([string]::IsNullOrEmpty($arg)) {
            $parts += '""'
        } elseif ($arg -match '[\s"]') {
            $parts += '"' + ($arg -replace '"', '\"') + '"'
        } else {
            $parts += $arg
        }
    }

    return ($parts -join ' ')
}

function Invoke-ResolvedAction {
    param(
        [hashtable]$Action,
        [hashtable]$Resolved
    )

    if ($Action.Interactive) {
        $start = Get-Date
        $previousLocation = (Get-Location).Path
        $previousEnv = @{}
        $Resolved.Env['PORTUI_INTERACTIVE'] = '1'

        foreach ($key in $Resolved.Env.Keys) {
            $envPath = "Env:$key"
            $existing = Get-Item -Path $envPath -ErrorAction SilentlyContinue
            if ($existing) {
                $previousEnv[$key] = @{ Exists = $true; Value = $existing.Value }
            } else {
                $previousEnv[$key] = @{ Exists = $false; Value = '' }
            }
            Set-Item -Path $envPath -Value ([string]$Resolved.Env[$key])
        }

        $exitCode = 0
        try {
            Set-Location -LiteralPath $Resolved.Cwd
            $resolvedArgs = @(Split-Args $Resolved.Args)
            & $Resolved.Program @resolvedArgs
            if ($null -ne $LASTEXITCODE) {
                $exitCode = $LASTEXITCODE
            }
        } catch {
            Write-Host $_
            $exitCode = 1
        } finally {
            Set-Location -LiteralPath $previousLocation
            foreach ($key in $previousEnv.Keys) {
                $envPath = "Env:$key"
                if ($previousEnv[$key].Exists) {
                    Set-Item -Path $envPath -Value $previousEnv[$key].Value
                } else {
                    Remove-Item -Path $envPath -ErrorAction SilentlyContinue
                }
            }
        }

        $duration = [int]((Get-Date) - $start).TotalSeconds
        Write-Host ""
        Write-Host "Status: exit code $exitCode"
        Write-Host "Duration: ${duration}s"
        Write-Host ""
        return $exitCode
    }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Resolved.Program
    $psi.Arguments = Build-ArgumentString -ArgList (Split-Args $Resolved.Args)
    $psi.WorkingDirectory = $Resolved.Cwd
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    foreach ($key in $Resolved.Env.Keys) {
        $psi.EnvironmentVariables[$key] = [string]$Resolved.Env[$key]
    }

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi

    $start = Get-Date
    try {
        [void]$process.Start()
    } catch {
        $duration = [int]((Get-Date) - $start).TotalSeconds
        Write-Host ""
        Write-Host "Status: failed to start"
        Write-Host "Duration: ${duration}s"
        Write-Host ""
        Write-Host $_.Exception.Message
        return 1
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()

    $timedOut = $false
    if ($Action.TimeoutSeconds -gt 0) {
        if (-not $process.WaitForExit($Action.TimeoutSeconds * 1000)) {
            $timedOut = $true
            try { $process.Kill() } catch {}
            $process.WaitForExit()
        } else {
            $process.WaitForExit()
        }
    } else {
        $process.WaitForExit()
    }

    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $duration = [int]((Get-Date) - $start).TotalSeconds

    Write-Host ""
    if ($timedOut) {
        Write-Host "Status: timed out after $($Action.TimeoutSeconds)s"
    } else {
        Write-Host "Status: exit code $($process.ExitCode)"
    }
    Write-Host "Duration: ${duration}s"
    Write-Host ""

    if (-not [string]::IsNullOrWhiteSpace($stdout)) {
        Write-Host $stdout.TrimEnd()
    }
    if (-not [string]::IsNullOrWhiteSpace($stderr)) {
        if (-not [string]::IsNullOrWhiteSpace($stdout)) {
            Write-Host ""
        }
        Write-Host $stderr.TrimEnd()
    }

    if ($timedOut) { return 124 }
    return $process.ExitCode
}

function Show-ActionPreview {
    param(
        [hashtable]$Action,
        [hashtable]$Resolved
    )

    Write-Host ""
    Write-Host $Action.Title
    if ($Action.Description) {
        Write-Host $Action.Description
    }
    Write-Host "Project: $($script:CurrentProjectId)"
    Write-Host "Working directory: $($Resolved.Cwd)"
    Write-Host "Resolution: $($Resolved.Source)"
    Write-Host "Command: $(Format-DisplayCommand -Resolved $Resolved)"
    if ($Action.Interactive) {
        Write-Host 'I/O: interactive terminal'
    }

    if ($Resolved.Env.Count -gt 0) {
        Write-Host "Environment overrides:"
        foreach ($key in ($Resolved.Env.Keys | Sort-Object)) {
            Write-Host "  ${key}=$($Resolved.Env[$key])"
        }
    }
}

function Load-ManifestContext {
    param(
        [string]$ResolvedManifestDir,
        [string]$ResolvedWorkspaceDir
    )

    Reset-PortUIVariables
    Initialize-Builtins -ResolvedManifestDir $ResolvedManifestDir -ResolvedWorkspaceDir $ResolvedWorkspaceDir
    $null = Load-Manifest -ResolvedManifestDir $ResolvedManifestDir
    $script:CurrentActions = @(Get-ActionFiles -ResolvedManifestDir $ResolvedManifestDir | ForEach-Object { Load-Action -Path $_.FullName })
}

function Show-ProjectList {
    param(
        [string[]]$Projects
    )

    Write-Host 'PortUI Workspace'
    Write-Host $WorkspaceDir
    Write-Host ''

    if ($Projects.Count -eq 0) {
        throw "No PortUI projects found in workspace: $WorkspaceDir"
    }

    for ($i = 0; $i -lt $Projects.Count; $i++) {
        $summary = Get-ManifestSummary -ResolvedManifestDir $Projects[$i]
        Write-Host ('{0,2}. {1} [{2}]' -f ($i + 1), $summary.Name, $summary.ProjectId)
        if ($summary.Description) {
            Write-Host "    $($summary.Description)"
        }
    }
}

function Get-ProjectManifestDir {
    param(
        [string[]]$Projects,
        [string]$TargetId
    )

    foreach ($manifestDir in $Projects) {
        if ((Get-ProjectIdFromManifestDir -ResolvedManifestDir $manifestDir) -eq $TargetId) {
            return $manifestDir
        }
    }

    return $null
}

function Show-ActionList {
    Write-Host $script:CurrentManifestName
    if ($script:CurrentManifestDescription) {
        Write-Host $script:CurrentManifestDescription
    }
    Write-Host "Project: $($script:CurrentProjectId)"
    if ($script:CurrentWorkspaceDir) {
        Write-Host "Workspace: $($script:CurrentWorkspaceDir)"
    }
    Write-Host ''

    for ($i = 0; $i -lt $script:CurrentActions.Count; $i++) {
        $action = $script:CurrentActions[$i]
        Write-Host ('{0,2}. {1} [{2}]' -f ($i + 1), $action.Title, $action.ID)
        if ($action.Description) {
            Write-Host "    $($action.Description)"
        }
    }
}

function Invoke-ActionById {
    param(
        [string]$ActionId
    )

    $action = $script:CurrentActions | Where-Object { $_.ID -eq $ActionId } | Select-Object -First 1
    if (-not $action) {
        throw "No action with id: $ActionId"
    }

    $resolved = Resolve-Action -Action $action
    Show-ActionPreview -Action $action -Resolved $resolved
    $exitCode = Invoke-ResolvedAction -Action $action -Resolved $resolved
    exit $exitCode
}

function Show-ActionMenu {
    param(
        [bool]$AllowBack
    )

    while ($true) {
        Clear-Host
        Write-Host $script:CurrentManifestName
        if ($script:CurrentManifestDescription) {
            Write-Host $script:CurrentManifestDescription
        }
        Write-Host "Project: $($script:CurrentProjectId)"
        Write-Host "Project directory: $($script:CurrentProjectDir)"
        if ($script:CurrentWorkspaceDir) {
            Write-Host "Workspace: $($script:CurrentWorkspaceDir)"
        }
        Write-Host "OS: $(Get-HostOSName)"
        Write-Host ''

        if ($script:CurrentActions.Count -eq 0) {
            throw "No actions found."
        }

        for ($i = 0; $i -lt $script:CurrentActions.Count; $i++) {
            $action = $script:CurrentActions[$i]
            Write-Host ('{0,2}. {1} [{2}]' -f ($i + 1), $action.Title, $action.ID)
            if ($action.Description) {
                Write-Host "    $($action.Description)"
            }
        }

        Write-Host ''
        if ($AllowBack) {
            $selection = Read-Host 'Select an action number, b to go back, or q to quit'
        } else {
            $selection = Read-Host 'Select an action number, or q to quit'
        }

        if ($selection -match '^(q|quit|exit)$') {
            exit 0
        }
        if ($AllowBack -and $selection -match '^(b|back)$') {
            return
        }

        $index = 0
        if (-not [int]::TryParse($selection, [ref]$index)) {
            continue
        }
        if ($index -lt 1 -or $index -gt $script:CurrentActions.Count) {
            continue
        }

        $action = $script:CurrentActions[$index - 1]
        $resolved = Resolve-Action -Action $action

        Clear-Host
        Show-ActionPreview -Action $action -Resolved $resolved
        $confirm = Read-Host 'Run this action? [Y/n]'
        if ($confirm -match '^(n|no)$') {
            continue
        }

        $null = Invoke-ResolvedAction -Action $action -Resolved $resolved
        Read-Host 'Press Enter to return to the menu' | Out-Null
    }
}

function Show-WorkspaceMenu {
    param(
        [string[]]$Projects
    )

    while ($true) {
        Clear-Host
        Write-Host 'PortUI Workspace'
        Write-Host $WorkspaceDir
        Write-Host "OS: $(Get-HostOSName)"
        Write-Host ''

        if ($Projects.Count -eq 0) {
            throw "No PortUI projects found in workspace: $WorkspaceDir"
        }

        for ($i = 0; $i -lt $Projects.Count; $i++) {
            $summary = Get-ManifestSummary -ResolvedManifestDir $Projects[$i]
            Write-Host ('{0,2}. {1} [{2}]' -f ($i + 1), $summary.Name, $summary.ProjectId)
            if ($summary.Description) {
                Write-Host "    $($summary.Description)"
            }
        }

        Write-Host ''
        $selection = Read-Host 'Select a project number, or q to quit'
        if ($selection -match '^(q|quit|exit)$') {
            exit 0
        }

        $index = 0
        if (-not [int]::TryParse($selection, [ref]$index)) {
            continue
        }
        if ($index -lt 1 -or $index -gt $Projects.Count) {
            continue
        }

        Load-ManifestContext -ResolvedManifestDir $Projects[$index - 1] -ResolvedWorkspaceDir $WorkspaceDir
        Show-ActionMenu -AllowBack $true
    }
}

if ($Help) {
    Show-Usage
    exit 0
}

if (-not [string]::IsNullOrWhiteSpace($InitProject)) {
    if (
        -not [string]::IsNullOrWhiteSpace($InstallProject) -or
        -not [string]::IsNullOrWhiteSpace($ManifestDir) -or
        -not [string]::IsNullOrWhiteSpace($WorkspaceDir) -or
        -not [string]::IsNullOrWhiteSpace($Project) -or
        $ListProjects -or
        $List -or
        -not [string]::IsNullOrWhiteSpace($Run)
    ) {
        throw '-InitProject cannot be combined with other runtime selection or action flags.'
    }

    Initialize-AndInstallPortUIProject -ProjectDir $InitProject
    exit 0
}

if (-not [string]::IsNullOrWhiteSpace($InstallProject)) {
    if (
        -not [string]::IsNullOrWhiteSpace($ManifestDir) -or
        -not [string]::IsNullOrWhiteSpace($WorkspaceDir) -or
        -not [string]::IsNullOrWhiteSpace($Project) -or
        $ListProjects -or
        $List -or
        -not [string]::IsNullOrWhiteSpace($Run)
    ) {
        throw '-InstallProject cannot be combined with runtime selection or action flags.'
    }

    Install-PortUIProject -ProjectDir $InstallProject
    exit 0
}

if (-not [string]::IsNullOrWhiteSpace($ManifestDir) -and (
    -not [string]::IsNullOrWhiteSpace($WorkspaceDir) -or
    -not [string]::IsNullOrWhiteSpace($Project) -or
    $ListProjects
)) {
    throw '-ManifestDir cannot be combined with workspace options.'
}

$defaultManifestDir = Join-Path $PSScriptRoot 'examples/demo'
$defaultWorkspaceDir = Get-ResolvedDirectory -Path (Join-Path $PSScriptRoot '..')
$mode = 'auto'
$projects = @()

if (-not [string]::IsNullOrWhiteSpace($ManifestDir)) {
    $ManifestDir = Get-ResolvedDirectory -Path $ManifestDir
    $mode = 'manifest'
} elseif (-not [string]::IsNullOrWhiteSpace($WorkspaceDir)) {
    $WorkspaceDir = Get-ResolvedDirectory -Path $WorkspaceDir
    $projects = @(Get-ProjectManifestDirs -ResolvedWorkspaceDir $WorkspaceDir)
    $mode = 'workspace'
} else {
    if ($localManifestDir = Get-LocalManifestDirectory) {
        $ManifestDir = Get-ResolvedDirectory -Path $localManifestDir
        $mode = 'manifest'
    } else {
    $WorkspaceDir = $defaultWorkspaceDir
    $projects = @(Get-ProjectManifestDirs -ResolvedWorkspaceDir $WorkspaceDir)
    if ($projects.Count -gt 0) {
        $mode = 'workspace'
    } elseif (-not [string]::IsNullOrWhiteSpace($Project) -or $ListProjects) {
        throw "No PortUI workspace projects were discovered under $WorkspaceDir"
    } else {
        $ManifestDir = Get-ResolvedDirectory -Path $defaultManifestDir
        $mode = 'manifest'
    }
    }
}

if ($mode -eq 'manifest') {
    if (-not [string]::IsNullOrWhiteSpace($Project) -or $ListProjects) {
        throw 'Project selection is only available in workspace mode.'
    }

    Load-ManifestContext -ResolvedManifestDir $ManifestDir -ResolvedWorkspaceDir ''

    if ($List) {
        Show-ActionList
        exit 0
    }

    if (-not [string]::IsNullOrWhiteSpace($Run)) {
        Invoke-ActionById -ActionId $Run
    }

    Show-ActionMenu -AllowBack $false
    exit 0
}

if ($projects.Count -eq 0) {
    throw "No PortUI projects found in workspace: $WorkspaceDir"
}

if ($ListProjects) {
    Show-ProjectList -Projects $projects
    exit 0
}

if (-not [string]::IsNullOrWhiteSpace($Project)) {
    $selectedManifest = Get-ProjectManifestDir -Projects $projects -TargetId $Project
    if (-not $selectedManifest) {
        throw "No project with id: $Project"
    }

    Load-ManifestContext -ResolvedManifestDir $selectedManifest -ResolvedWorkspaceDir $WorkspaceDir

    if ($List) {
        Show-ActionList
        exit 0
    }

    if (-not [string]::IsNullOrWhiteSpace($Run)) {
        Invoke-ActionById -ActionId $Run
    }

    Show-ActionMenu -AllowBack $true
    exit 0
}

if ($List -or -not [string]::IsNullOrWhiteSpace($Run)) {
    throw 'Workspace mode requires -Project when using -List or -Run.'
}

Show-WorkspaceMenu -Projects $projects
