// Windows installer for gca.
//
//   gca-installer               install gca.exe to %LOCALAPPDATA%\gca and add it to PATH
//   gca-installer --uninstall   remove it again
//   gca-installer --yes         no questions and no "press Enter" at the end, for scripts
//
// Windows only. On macOS and Linux, use install.sh (see the README).

#[cfg(windows)]
fn main() {
    win::main();
}

#[cfg(not(windows))]
fn main() {
    eprintln!("gca-installer is Windows-only. On macOS or Linux, run:");
    eprintln!("  curl -fsSL https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.sh | sh");
    std::process::exit(1);
}

#[cfg(windows)]
mod win {
    use std::env;
    use std::fs;
    use std::io::{self, Write};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use winreg::enums::{HKEY_CURRENT_USER, KEY_READ, KEY_WRITE, REG_EXPAND_SZ};
    use winreg::{RegKey, RegValue};

    static GCA_EXE_BYTES: &[u8] = include_bytes!("../../target/release/gca.exe");

    struct Options {
        uninstall: bool,
        yes: bool,
    }

    pub fn main() {
        let mut opts = Options {
            uninstall: false,
            yes: false,
        };
        for arg in env::args().skip(1) {
            match arg.as_str() {
                "--uninstall" => opts.uninstall = true,
                "-y" | "--yes" => opts.yes = true,
                "-h" | "--help" => {
                    println!("usage: gca-installer [--uninstall] [--yes]");
                    return;
                }
                other => {
                    eprintln!("unknown option {other}; see --help");
                    std::process::exit(2);
                }
            }
        }

        let ok = if opts.uninstall {
            uninstall()
        } else {
            install(&opts)
        };
        if !opts.yes {
            pause();
        }
        if !ok {
            std::process::exit(1);
        }
    }

    fn install(opts: &Options) -> bool {
        println!("=== gca installer ===\n");
        let dir = match install_dir().and_then(|d| write_binary(&d).map(|_| d)) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("failed to install gca: {e}");
                return false;
            }
        };
        match add_to_path(&dir) {
            Ok(true) => println!("[ok] added {} to your PATH", dir.display()),
            Ok(false) => println!("[ok] {} is already in your PATH", dir.display()),
            Err(e) => eprintln!("[warn] could not update PATH: {e}"),
        }
        check_git(opts.yes);
        println!("\n=== done ===");
        println!("open a new terminal, stage some changes, and run `gca` (see `gca --help`).");
        true
    }

    fn uninstall() -> bool {
        println!("=== gca uninstaller ===\n");
        let dir = match install_dir() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{e}");
                return false;
            }
        };
        let exe = dir.join("gca.exe");
        let mut ok = true;
        match fs::remove_file(&exe) {
            Ok(()) => println!("[ok] removed {}", exe.display()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                println!("[ok] {} was not installed", exe.display())
            }
            Err(e) => {
                eprintln!("[error] could not remove {}: {e}", exe.display());
                ok = false;
            }
        }
        // Only removes the folder if nothing else is in it.
        let _ = fs::remove_dir(&dir);
        match remove_from_path(&dir) {
            Ok(true) => println!("[ok] removed {} from your PATH", dir.display()),
            Ok(false) => println!("[ok] {} was not in your PATH", dir.display()),
            Err(e) => {
                eprintln!("[error] could not update PATH: {e}");
                ok = false;
            }
        }
        // where `gca model install` puts the models
        let models = env::var_os("GCA_MODELS_DIR")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("LOCALAPPDATA")
                    .filter(|v| !v.is_empty())
                    .map(|d| PathBuf::from(d).join("gca").join("models"))
            });
        if let Some(models) = models.filter(|m| m.is_dir()) {
            println!(
                "\nthe models gca downloaded stay in {}; delete that folder to remove them",
                models.display()
            );
        }
        println!("\nsettings stay in your git config; remove them with");
        println!("  git config --global --remove-section gca");
        ok
    }

    fn install_dir() -> io::Result<PathBuf> {
        // An empty variable counts as unset.
        let base = env::var_os("LOCALAPPDATA")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("USERPROFILE")
                    .filter(|v| !v.is_empty())
                    .map(|home| PathBuf::from(home).join("AppData").join("Local"))
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "neither LOCALAPPDATA nor USERPROFILE is set",
                )
            })?;
        Ok(base.join("gca"))
    }

    fn write_binary(dir: &Path) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let exe_path = dir.join("gca.exe");
        fs::write(&exe_path, GCA_EXE_BYTES)?;
        let size_mb = GCA_EXE_BYTES.len() as f64 / (1024.0 * 1024.0);
        println!("[ok] wrote {} ({size_mb:.1} MB)", exe_path.display());
        Ok(())
    }

    fn user_path() -> io::Result<(RegKey, String)> {
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        let env_key = hkcu.open_subkey_with_flags("Environment", KEY_READ | KEY_WRITE)?;
        let current: String = match env_key.get_value("Path") {
            Ok(v) => v,
            Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e),
        };
        Ok((env_key, current))
    }

    fn same_dir(entry: &str, dir: &str) -> bool {
        entry
            .trim()
            .trim_end_matches('\\')
            .eq_ignore_ascii_case(dir.trim_end_matches('\\'))
    }

    fn add_to_path(dir: &Path) -> io::Result<bool> {
        let (env_key, current) = user_path()?;
        let dir_str = dir.to_string_lossy();
        if current.split(';').any(|p| same_dir(p, &dir_str)) {
            return Ok(false);
        }
        let new_path = if current.is_empty() {
            dir_str.to_string()
        } else {
            format!("{};{}", current.trim_end_matches(';'), dir_str)
        };
        // REG_EXPAND_SZ keeps entries such as %USERPROFILE%\... expanding.
        env_key.set_raw_value("Path", &expand_sz(&new_path))?;
        broadcast_env_change();
        Ok(true)
    }

    fn remove_from_path(dir: &Path) -> io::Result<bool> {
        let (env_key, current) = user_path()?;
        let dir_str = dir.to_string_lossy();
        let kept: Vec<&str> = current
            .split(';')
            .filter(|p| !same_dir(p, &dir_str))
            .collect();
        if kept.len() == current.split(';').count() {
            return Ok(false);
        }
        env_key.set_raw_value("Path", &expand_sz(&kept.join(";")))?;
        broadcast_env_change();
        Ok(true)
    }

    fn expand_sz(s: &str) -> RegValue {
        let bytes = s
            .encode_utf16()
            .chain([0])
            .flat_map(u16::to_le_bytes)
            .collect();
        RegValue {
            bytes,
            vtype: REG_EXPAND_SZ,
        }
    }

    // Best effort: broadcast WM_SETTINGCHANGE so Explorer and new terminals see
    // the updated PATH without signing out.
    fn broadcast_env_change() {
        let _ = Command::new("powershell")
            .args([
                "-NoProfile", "-Command",
                r#"Add-Type -Namespace Win32 -Name NativeMethods -MemberDefinition '[DllImport("user32.dll",SetLastError=true,CharSet=CharSet.Auto)]public static extern IntPtr SendMessageTimeout(IntPtr hWnd,uint Msg,UIntPtr wParam,string lParam,uint fuFlags,uint uTimeout,out UIntPtr lpdwResult);'; $HWND_BROADCAST=[IntPtr]0xffff; $WM_SETTINGCHANGE=0x1a; $result=[UIntPtr]::Zero; [Win32.NativeMethods]::SendMessageTimeout($HWND_BROADCAST,$WM_SETTINGCHANGE,[UIntPtr]::Zero,'Environment',2,5000,[ref]$result) | Out-Null"#,
            ])
            .output();
    }

    fn check_git(yes: bool) {
        print!("\nchecking for git... ");
        let _ = io::stdout().flush();
        if let Ok(out) = Command::new("git").arg("--version").output() {
            if out.status.success() {
                println!("{}", String::from_utf8_lossy(&out.stdout).trim());
                return;
            }
        }
        println!("not found");
        println!("gca runs git for every commit, so it needs Git for Windows.");
        if !yes && !confirm("install it now with winget?") {
            println!("skipped; get it from https://git-scm.com/download/win");
            return;
        }
        let status = Command::new("winget")
            .args(["install", "--id", "Git.Git", "-e", "--source", "winget"])
            .status();
        match status {
            Ok(s) if s.success() => {
                println!("\n[ok] git installed; open a new terminal so it is on your PATH.");
            }
            _ => {
                println!("\n[warn] winget could not install git.");
                println!("       get it from https://git-scm.com/download/win");
            }
        }
    }

    fn confirm(question: &str) -> bool {
        print!("{question} [Y/n] ");
        let _ = io::stdout().flush();
        let mut answer = String::new();
        if io::stdin().read_line(&mut answer).is_err() {
            return false;
        }
        matches!(
            answer.trim().to_ascii_lowercase().as_str(),
            "" | "y" | "yes"
        )
    }

    fn pause() {
        print!("\npress Enter to close...");
        let _ = io::stdout().flush();
        let _ = io::stdin().read_line(&mut String::new());
    }
}
