// gca installer — extracts the embedded gca.exe to %LOCALAPPDATA%\gca,
// adds the directory to the user PATH, and checks for git.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;

use winreg::enums::{HKEY_CURRENT_USER, KEY_READ, KEY_WRITE};
use winreg::RegKey;

static GCA_EXE_BYTES: &[u8] = include_bytes!("../../target/release/gca.exe");

fn main() {
    println!("=== gca installer ===\n");

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

    println!("\n=== done ===");
    println!("open a new terminal and run `gca` to get started.");
    pause();
}

fn install_binary() -> io::Result<PathBuf> {
    let local_app = env::var("LOCALAPPDATA")
        .unwrap_or_else(|_| {
            let home = env::var("USERPROFILE").expect("cannot find LOCALAPPDATA or USERPROFILE");
            format!("{home}\\AppData\\Local")
        });
    let dir = PathBuf::from(&local_app).join("gca");
    fs::create_dir_all(&dir)?;

    let exe_path = dir.join("gca.exe");
    fs::write(&exe_path, GCA_EXE_BYTES)?;
    let size_mb = GCA_EXE_BYTES.len() as f64 / (1024.0 * 1024.0);
    println!("[ok] wrote {} ({:.1} MB)", exe_path.display(), size_mb);
    Ok(dir)
}

fn add_to_path(dir: &PathBuf) -> io::Result<bool> {
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let env_key = hkcu.open_subkey_with_flags("Environment", KEY_READ | KEY_WRITE)?;

    let current: String = env_key.get_value("Path").unwrap_or_default();
    let dir_str = dir.to_string_lossy();

    // Check if already present (case-insensitive on Windows).
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
    env_key.set_value("Path", &new_path)?;

    // Broadcast WM_SETTINGCHANGE so some terminals pick up the change.
    broadcast_env_change();

    Ok(true)
}

fn broadcast_env_change() {
    // Best-effort: ask explorer to reload environment variables.
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
                    println!("\n[ok] git installed; restart your terminal for it to appear in PATH.");
                }
                _ => {
                    println!("\n[warn] winget install failed or winget not available.");
                    println!("       install git manually from https://git-scm.com/download/win");
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
