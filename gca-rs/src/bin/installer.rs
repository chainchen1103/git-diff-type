#[cfg(not(windows))]
fn main() {
    eprintln!("gca-installer is only available on Windows");
    std::process::exit(1);
}

#[cfg(windows)]
fn main() {
    windows::run();
}

#[cfg(windows)]
mod windows {

    use std::env;
    use std::fs;
    use std::io::{self, Write};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use winreg::enums::{HKEY_CURRENT_USER, KEY_READ, KEY_WRITE, REG_EXPAND_SZ};
    use winreg::{RegKey, RegValue};

    static GCA_EXE_BYTES: &[u8] = include_bytes!("../../target/release/gca.exe");

    pub fn run() {
        println!("gca installer\n");

        let install_dir = match install_binary() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("failed to install gca: {e}");
                pause();
                std::process::exit(1);
            }
        };

        match add_to_path(&install_dir) {
            Ok(true) => println!("[ok] added {} to user PATH", install_dir.display()),
            Ok(false) => println!("[ok] {} is already in PATH", install_dir.display()),
            Err(e) => eprintln!("[warn] could not update PATH: {e}"),
        }

        check_git();

        println!("\nDone.");
        println!("open a new terminal and run `gca` to get started.");
        pause();
    }

    fn install_binary() -> io::Result<PathBuf> {
        let local_app = env::var_os("LOCALAPPDATA")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("USERPROFILE")
                    .filter(|value| !value.is_empty())
                    .map(|profile| PathBuf::from(profile).join("AppData").join("Local"))
            })
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "cannot find LOCALAPPDATA or USERPROFILE",
                )
            })?;
        let dir = local_app.join("gca");
        fs::create_dir_all(&dir)?;

        let exe_path = dir.join("gca.exe");
        fs::write(&exe_path, GCA_EXE_BYTES)?;
        let size_mb = GCA_EXE_BYTES.len() as f64 / (1024.0 * 1024.0);
        println!("[ok] wrote {}, {:.1} MB", exe_path.display(), size_mb);
        Ok(dir)
    }

    fn add_to_path(dir: &Path) -> io::Result<bool> {
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        let env_key = hkcu.open_subkey_with_flags("Environment", KEY_READ | KEY_WRITE)?;

        let current: String = match env_key.get_value("Path") {
            Ok(v) => v,
            Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e),
        };
        let dir_str = dir.to_string_lossy();

        let already = current
            .split(';')
            .any(|p| p.trim().eq_ignore_ascii_case(&dir_str));
        if already {
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

    fn check_git() {
        print!("\nchecking for git... ");
        let _ = io::stdout().flush();

        match Command::new("git").arg("--version").output() {
            Ok(out) if out.status.success() => {
                let ver = String::from_utf8_lossy(&out.stdout);
                println!("{}", ver.trim());
            }
            _ => {
                println!("not found");
                println!("\ngit is required. attempting to install via winget...\n");
                let status = Command::new("winget")
                    .args(["install", "--id", "Git.Git", "-e", "--source", "winget"])
                    .status();
                match status {
                    Ok(s) if s.success() => {
                        println!(
                            "\n[ok] git installed; restart your terminal for it to appear in PATH."
                        );
                    }
                    _ => {
                        println!("\n[warn] winget install failed or winget not available.");
                        println!(
                            "       install git manually from https://git-scm.com/download/win"
                        );
                    }
                }
            }
        }
    }

    fn pause() {
        print!("\npress Enter to close...");
        let _ = io::stdout().flush();
        let _ = io::stdin().read_line(&mut String::new());
    }
}
