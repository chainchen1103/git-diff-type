# CI check for install.ps1: serve a locally built gca as a release, then
# install, reinstall and uninstall it. Runs under PowerShell 7 and Windows
# PowerShell 5.1, from the repository root after `cargo build --bin gca`.
$ErrorActionPreference = 'Stop'
$port = if ($PSVersionTable.PSEdition -eq 'Core') { 8766 } else { 8767 }
$release = Join-Path $env:RUNNER_TEMP "release-$port"
$asset = 'gca-x86_64-pc-windows-msvc.exe'
New-Item -ItemType Directory -Force -Path $release | Out-Null
Copy-Item -LiteralPath 'gca-rs\target\debug\gca.exe' -Destination (Join-Path $release $asset)
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $release $asset)).Hash.ToLowerInvariant()
Set-Content -Path (Join-Path $release 'SHA256SUMS') -Value "$hash  $asset" -Encoding ascii
$server = Start-Process python -ArgumentList '-m', 'http.server', $port, '--bind', '127.0.0.1', '--directory', $release -PassThru -WindowStyle Hidden
# Python can take more than a few seconds to start on a fresh runner: wait
# until the server takes connections rather than for a fixed time.
$deadline = (Get-Date).AddSeconds(60)
while ($true) {
    if ($server.HasExited) { throw "the test server stopped (exit code $($server.ExitCode))" }
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect('127.0.0.1', $port)
        break
    } catch {
        if ((Get-Date) -gt $deadline) { throw "the test server did not start on port $port" }
        Start-Sleep -Milliseconds 250
    } finally {
        $client.Close()
    }
}

function Get-UserPath {
    (Get-Item 'HKCU:\Environment').GetValue('Path', '', 'DoNotExpandEnvironmentNames')
}

try {
    $env:GCA_DOWNLOAD_URL = "http://127.0.0.1:$port"
    $env:GCA_INSTALL_DIR = Join-Path $env:RUNNER_TEMP "gca-$port"
    $dir = $env:GCA_INSTALL_DIR

    Invoke-Expression (Get-Content -Raw install.ps1)
    $version = gca --version  # found through the PATH the script set for this session
    if ($version -notmatch '^gca \d') { throw "unexpected output from gca --version: $version" }
    if ((Get-Item 'HKCU:\Environment').GetValueKind('Path') -ne 'ExpandString') { throw 'the user PATH is no longer REG_EXPAND_SZ' }
    if (@((Get-UserPath) -split ';' | Where-Object { $_ -eq $dir }).Count -ne 1) { throw "the user PATH lacks $dir" }

    Invoke-Expression (Get-Content -Raw install.ps1)
    if (@((Get-UserPath) -split ';' | Where-Object { $_ -eq $dir }).Count -ne 1) { throw 'reinstalling added another PATH entry' }

    $env:GCA_UNINSTALL = '1'
    Invoke-Expression (Get-Content -Raw install.ps1)
    if (Test-Path -LiteralPath (Join-Path $dir 'gca.exe')) { throw 'gca.exe is still installed' }
    if (((Get-UserPath) -split ';') -contains $dir) { throw 'the PATH entry was left behind' }
    Write-Host "install.ps1 works under PowerShell $($PSVersionTable.PSVersion)"
} finally {
    Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
}
