# Install gca on Windows from the latest GitHub release. In PowerShell:
#
#   irm https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.ps1 | iex
#
# Run it again to upgrade. To remove gca:
#
#   $env:GCA_UNINSTALL = 1; irm https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.ps1 | iex
#
# In Command Prompt (cmd), hand either one to PowerShell:
#
#   powershell -c "irm https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.ps1 | iex"
#   powershell -c "$env:GCA_UNINSTALL = 1; irm https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.ps1 | iex"
#
# Settings, as environment variables:
#   GCA_VERSION = v0.4.0     install this release instead of the latest
#   GCA_INSTALL_DIR = DIR    install into DIR instead of %LOCALAPPDATA%\gca
#   GCA_NO_MODIFY_PATH = 1   leave your PATH alone
#   GCA_DOWNLOAD_URL = URL   download from URL instead of GitHub (a mirror, or a test)
#
# The download is checked against the release's SHA256SUMS before anything is
# installed. The x64 build also runs on Windows on Arm. gca-installer.exe uses
# the same folder and PATH entry, so either one can remove what the other
# installed. Works in Windows PowerShell 5.1 and PowerShell 7.
#
# Everything runs in one script block: a truncated download runs nothing, and
# no variables are left in your session apart from PATH.

& {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'  # the progress bar slows Invoke-WebRequest down a lot
    $Repo = 'chainchen1103/git-diff-type'
    $Asset = 'gca-x86_64-pc-windows-msvc.exe'

    function Get-InstallDir {
        if ($env:GCA_INSTALL_DIR) { return $env:GCA_INSTALL_DIR }
        $base = $env:LOCALAPPDATA
        if (-not $base) { $base = Join-Path $env:USERPROFILE 'AppData\Local' }
        return (Join-Path $base 'gca')
    }

    function Get-DownloadBase {
        if ($env:GCA_DOWNLOAD_URL) { return $env:GCA_DOWNLOAD_URL.TrimEnd('/') }
        if ($env:GCA_VERSION) { return "https://github.com/$Repo/releases/download/$($env:GCA_VERSION)" }
        return "https://github.com/$Repo/releases/latest/download"
    }

    function Save-Url([string]$Url, [string]$Path) {
        try {
            Invoke-WebRequest -Uri $Url -OutFile $Path -UseBasicParsing
        } catch {
            throw "could not download $Url ($($_.Exception.Message)); is there a release yet? see https://github.com/$Repo/releases"
        }
    }

    # The hash SHA256SUMS lists for $Name, lowercase.
    function Get-ExpectedHash([string]$SumsPath, [string]$Name) {
        foreach ($line in Get-Content -LiteralPath $SumsPath) {
            $parts = $line.Trim() -split '\s+', 2
            if ($parts.Count -eq 2 -and ($parts[1] -eq $Name -or $parts[1] -eq "*$Name")) {
                return $parts[0].ToLowerInvariant()
            }
        }
        throw "SHA256SUMS lists no $Name; nothing was installed"
    }

    function Test-SameDir([string]$Entry, [string]$Dir) {
        $expanded = [Environment]::ExpandEnvironmentVariables($Entry).Trim().TrimEnd('\', '/')
        return $expanded -ieq $Dir.Trim().TrimEnd('\', '/')
    }

    # Tell Explorer and new terminals that PATH changed. Best effort.
    function Send-EnvironmentChange {
        try {
            if (-not ('GcaInstall.Native' -as [type])) {
                Add-Type -Namespace GcaInstall -Name Native -MemberDefinition @'
[DllImport("user32.dll", SetLastError = true, CharSet = CharSet.Auto)]
public static extern IntPtr SendMessageTimeout(IntPtr hWnd, uint Msg, UIntPtr wParam, string lParam, uint fuFlags, uint uTimeout, out UIntPtr lpdwResult);
'@
            }
            $result = [UIntPtr]::Zero
            [GcaInstall.Native]::SendMessageTimeout([IntPtr]0xffff, 0x1a, [UIntPtr]::Zero, 'Environment', 2, 5000, [ref]$result) | Out-Null
        } catch { }
    }

    # A PATH value with $Dir added or removed, or $null when it would not change.
    function Edit-PathValue([string]$Value, [string]$Dir, [bool]$Add) {
        $entries = @($Value -split ';' | Where-Object { $_ })
        $others = @($entries | Where-Object { -not (Test-SameDir $_ $Dir) })
        if ($Add) {
            if ($others.Count -ne $entries.Count) { return $null }
            return ((@($entries) + $Dir) -join ';')
        }
        if ($others.Count -eq $entries.Count) { return $null }
        return ($others -join ';')
    }

    # Change the user PATH in the registry. It stays REG_EXPAND_SZ, so entries
    # such as %USERPROFILE%\bin keep working. Returns whether anything changed.
    function Update-UserPath([string]$Dir, [bool]$Add) {
        $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
        if (-not $key) { $key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment') }
        try {
            $raw = [string]$key.GetValue('Path', '', 'DoNotExpandEnvironmentNames')
            $new = Edit-PathValue $raw $Dir $Add
            if ($null -eq $new) { return $false }
            $key.SetValue('Path', $new, 'ExpandString')
        } finally {
            $key.Close()
        }
        Send-EnvironmentChange
        return $true
    }

    function Install-Gca([string]$Dir) {
        $base = Get-DownloadBase
        $tmp = Join-Path ([IO.Path]::GetTempPath()) ('gca-' + [guid]::NewGuid())
        New-Item -ItemType Directory -Path $tmp | Out-Null
        try {
            Write-Host "downloading $Asset from $base"
            $exe = Join-Path $tmp $Asset
            $sums = Join-Path $tmp 'SHA256SUMS'
            Save-Url "$base/$Asset" $exe
            Save-Url "$base/SHA256SUMS" $sums
            $expected = Get-ExpectedHash $sums $Asset
            $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant()
            if ($actual -ne $expected) {
                throw "checksum mismatch for $Asset (expected $expected, got $actual); nothing was installed"
            }
            New-Item -ItemType Directory -Force -Path $Dir | Out-Null
            $target = Join-Path $Dir 'gca.exe'
            try {
                Copy-Item -LiteralPath $exe -Destination $target -Force
            } catch {
                throw "could not write $target; close any running gca and try again ($($_.Exception.Message))"
            }
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Host "installed $(& $target --version) to $target"

        if ($env:GCA_NO_MODIFY_PATH -eq '1') {
            Write-Host "PATH left alone; add $Dir to it to run gca from anywhere"
        } elseif (Update-UserPath $Dir $true) {
            Write-Host "added $Dir to your PATH"
        }
        if (-not (@($env:Path -split ';') | Where-Object { $_ -and (Test-SameDir $_ $Dir) })) {
            $env:Path = ([string]$env:Path).TrimEnd(';') + ";$Dir"
        }
        $found = Get-Command gca -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found -and -not (Test-SameDir (Split-Path $found.Source) $Dir)) {
            Write-Host "note: $($found.Source) comes first on your PATH and will run instead"
        }
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            Write-Host 'note: gca runs git, which is not installed yet:  winget install --id Git.Git -e --source winget'
        }
        Write-Host 'gca is ready in new terminals, and already in this PowerShell session; stage some changes and run gca (see gca --help)'
    }

    function Uninstall-Gca([string]$Dir) {
        $target = Join-Path $Dir 'gca.exe'
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Force
            Write-Host "removed $target"
        } else {
            Write-Host "gca is not installed in $Dir"
        }
        if ((Test-Path -LiteralPath $Dir) -and -not (Get-ChildItem -LiteralPath $Dir -Force | Select-Object -First 1)) {
            Remove-Item -LiteralPath $Dir -Force
        }
        if (Update-UserPath $Dir $false) {
            Write-Host "removed $Dir from your PATH"
        }
        $env:Path = (@($env:Path -split ';') | Where-Object { $_ -and -not (Test-SameDir $_ $Dir) }) -join ';'
        Remove-Item Env:GCA_UNINSTALL -ErrorAction SilentlyContinue
        Write-Host 'settings stay in your git config; remove them with:  git config --global --remove-section gca'
    }

    if ($PSVersionTable.PSEdition -eq 'Core' -and -not $IsWindows) {
        throw 'install.ps1 is for Windows; on macOS or Linux run:  curl -fsSL https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.sh | sh'
    }
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $dir = Get-InstallDir
    if ($env:GCA_UNINSTALL -eq '1') {
        Uninstall-Gca $dir
    } else {
        Install-Gca $dir
    }
}
