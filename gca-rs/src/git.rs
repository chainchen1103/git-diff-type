use anyhow::{anyhow, Result};
use std::process::{Command, Stdio};

pub struct Stats {
    pub files_changed: u32,
    pub additions: u32,
    pub deletions: u32,
}

fn run(args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| anyhow!("failed to run git: {e}"))?;
    if !out.status.success() {
        return Err(anyhow!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn diff(cached: bool) -> Result<String> {
    let args: Vec<&str> = if cached {
        vec!["diff", "--cached", "--no-color", "--no-ext-diff"]
    } else {
        vec!["diff", "--no-color", "--no-ext-diff"]
    };
    let out = run(&args)?;
    Ok(out.trim().to_string())
}

pub fn staged_files(cached: bool) -> Result<Vec<String>> {
    let args: Vec<&str> = if cached {
        vec!["diff", "--cached", "--name-only"]
    } else {
        vec!["diff", "--name-only"]
    };
    let out = run(&args)?;
    Ok(out.lines().filter(|l| !l.is_empty()).map(|s| s.to_string()).collect())
}

fn run_inherit(args: &[&str]) -> Result<()> {
    let status = Command::new("git")
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| anyhow!("failed to run git: {e}"))?;
    if !status.success() {
        return Err(anyhow!("git {} exited with {}", args.join(" "), status));
    }
    Ok(())
}

pub fn add_all() -> Result<()> {
    run_inherit(&["add", "-A"])
}

pub fn add_paths(paths: &[String]) -> Result<()> {
    let mut args: Vec<&str> = vec!["add"];
    for p in paths {
        args.push(p.as_str());
    }
    run_inherit(&args)
}

pub fn commit(message: &str) -> Result<()> {
    run_inherit(&["commit", "-m", message])
}

pub fn push(remote: Option<&str>) -> Result<()> {
    match remote {
        Some(r) => run_inherit(&["push", r]),
        None => run_inherit(&["push"]),
    }
}

pub fn get_config(key: &str) -> Option<String> {
    let out = Command::new("git").args(["config", "--get", key]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

pub fn set_config_global(key: &str, value: &str) -> Result<()> {
    run_inherit(&["config", "--global", key, value])
}

pub fn stats(cached: bool) -> Result<Stats> {
    let args: Vec<&str> = if cached {
        vec!["diff", "--cached", "--numstat"]
    } else {
        vec!["diff", "--numstat"]
    };
    let out = run(&args)?;
    let mut s = Stats { files_changed: 0, additions: 0, deletions: 0 };
    for line in out.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }
        let adds: u32 = if parts[0] == "-" { 0 } else { parts[0].parse().unwrap_or(0) };
        let dels: u32 = if parts[1] == "-" { 0 } else { parts[1].parse().unwrap_or(0) };
        s.files_changed += 1;
        s.additions += adds;
        s.deletions += dels;
    }
    Ok(s)
}
