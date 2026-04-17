mod features;
mod git;
mod heuristics;
mod model;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};
use std::path::PathBuf;

use model::Model;

const PUSH_CONFIG_KEY: &str = "gca.push";
const REMOTE_CONFIG_KEY: &str = "gca.remote";

#[derive(Parser, Debug)]
#[command(name = "gca", about = "Git commit type analyzer")]
struct Cli {
    /// Override the embedded model with an external JSON file.
    #[arg(long, global = true)]
    model: Option<PathBuf>,

    /// Number of top suggestions to show.
    #[arg(long, default_value_t = 3, global = true)]
    topk: usize,

    /// Print the suggestion but do not commit or push.
    #[arg(long, global = true)]
    dry_run: bool,

    /// One-off: commit but do not push (overrides `gca.push`).
    #[arg(long, global = true)]
    no_push: bool,

    /// One-off: ask before pushing (overrides `gca.push`).
    #[arg(long, global = true)]
    confirm_push: bool,

    /// One-off: push to this remote (overrides `gca.remote`).
    #[arg(long, global = true)]
    remote: Option<String>,

    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Persistent settings stored in git config.
    Config {
        #[command(subcommand)]
        what: ConfigCmd,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigCmd {
    /// Set or show the default push behavior: auto | ask | never.
    Push { mode: Option<String> },
    /// Set or show the default push remote (e.g. origin, upstream).
    Remote { name: Option<String> },
}

#[derive(Clone, Copy, PartialEq)]
enum PushMode { Auto, Ask, Never }

fn resolve_push_mode(no_push: bool, confirm_push: bool) -> PushMode {
    if no_push { return PushMode::Never; }
    if confirm_push { return PushMode::Ask; }
    match git::get_config(PUSH_CONFIG_KEY).as_deref() {
        Some("never") | Some("off") | Some("no") => PushMode::Never,
        Some("ask") | Some("confirm") => PushMode::Ask,
        Some("auto") | Some("yes") | None => PushMode::Auto,
        Some(other) => {
            eprintln!("warning: unknown {PUSH_CONFIG_KEY} value {other:?}; falling back to auto");
            PushMode::Auto
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(cmd) = &cli.command {
        return handle_subcommand(cmd);
    }

    run_commit_flow(&cli)
}

fn handle_subcommand(cmd: &Cmd) -> Result<()> {
    match cmd {
        Cmd::Config { what: ConfigCmd::Push { mode: None } } => {
            let cur = git::get_config(PUSH_CONFIG_KEY).unwrap_or_else(|| "auto".to_string());
            println!("{PUSH_CONFIG_KEY} = {cur}");
            Ok(())
        }
        Cmd::Config { what: ConfigCmd::Push { mode: Some(m) } } => {
            let normalized = m.to_lowercase();
            match normalized.as_str() {
                "auto" | "ask" | "never" => {
                    git::set_config_global(PUSH_CONFIG_KEY, &normalized)
                        .context("failed to update git config")?;
                    println!("set {PUSH_CONFIG_KEY} = {normalized}");
                    Ok(())
                }
                _ => anyhow::bail!("invalid mode {m:?}; expected auto | ask | never"),
            }
        }
        Cmd::Config { what: ConfigCmd::Remote { name: None } } => {
            let cur = git::get_config(REMOTE_CONFIG_KEY)
                .unwrap_or_else(|| "(default)".to_string());
            println!("{REMOTE_CONFIG_KEY} = {cur}");
            Ok(())
        }
        Cmd::Config { what: ConfigCmd::Remote { name: Some(n) } } => {
            git::set_config_global(REMOTE_CONFIG_KEY, n)
                .context("failed to update git config")?;
            println!("set {REMOTE_CONFIG_KEY} = {n}");
            Ok(())
        }
    }
}

fn resolve_remote(flag: &Option<String>) -> Option<String> {
    if flag.is_some() { return flag.clone(); }
    git::get_config(REMOTE_CONFIG_KEY)
}

fn run_commit_flow(cli: &Cli) -> Result<()> {
    let mut diff_text = git::diff(true).context("failed to read staged diff")?;
    if diff_text.is_empty() {
        println!("no staged changes; running `git add -A`");
        git::add_all()?;
        diff_text = git::diff(true).context("failed to read staged diff")?;
        if diff_text.is_empty() {
            println!("nothing to commit — working tree clean");
            return Ok(());
        }
    }

    let files = git::staged_files(true)?;
    let stats = git::stats(true)?;

    static EMBEDDED_MODEL: &[u8] = include_bytes!("../../out/model_v2.json");
    let model = match &cli.model {
        Some(p) => Model::load(p)
            .with_context(|| format!("failed to load model from {}", p.display()))?,
        None => Model::from_bytes(EMBEDDED_MODEL).context("failed to parse embedded model")?,
    };

    let diff_truncated: String = diff_text.chars().take(20_000).collect();
    let numeric = [
        stats.files_changed as f64,
        stats.additions as f64,
        stats.deletions as f64,
        stats.additions as f64 / (stats.deletions as f64 + 1.0),
    ];
    let feats = model.build_features(&diff_truncated, numeric);
    let probs = model.predict_proba(&feats);
    let top = model.topk(&probs, cli.topk.max(1));

    let default_idx = heuristics::classify(&files)
        .and_then(|hit| top.iter().position(|(l, _)| *l == hit.label))
        .unwrap_or(0);

    println!(
        "Stats: +{} / -{} lines in {} files",
        stats.additions, stats.deletions, stats.files_changed
    );

    let items: Vec<String> = top
        .iter()
        .map(|(l, s)| format!("{:<9} ({:5.1}%)", l, s * 100.0))
        .collect();

    let chosen = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Commit type")
        .items(&items)
        .default(default_idx)
        .interact()?;
    let label = top[chosen].0;

    let description: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt(format!("{label}:"))
        .validate_with(|s: &String| -> std::result::Result<(), &str> {
            if s.trim().is_empty() { Err("description cannot be empty") } else { Ok(()) }
        })
        .interact_text()?;

    let message = format!("{}: {}", label, description.trim());

    if cli.dry_run {
        println!("\n(dry-run) git commit -m \"{message}\"");
        return Ok(());
    }

    git::commit(&message).context("git commit failed")?;

    let remote = resolve_remote(&cli.remote);
    let remote_ref = remote.as_deref();

    match resolve_push_mode(cli.no_push, cli.confirm_push) {
        PushMode::Never => Ok(()),
        PushMode::Ask => {
            let prompt = match &remote {
                Some(r) => format!("Push to {r}?"),
                None => "Push to remote?".to_string(),
            };
            let yes = Confirm::with_theme(&ColorfulTheme::default())
                .with_prompt(prompt)
                .default(true)
                .interact()?;
            if yes { git::push(remote_ref).context("git push failed") } else { Ok(()) }
        }
        PushMode::Auto => git::push(remote_ref).context("git push failed"),
    }
}
