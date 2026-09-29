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

    /// Print suggestions without changing staged files, committing, or pushing.
    #[arg(long, global = true)]
    dry_run: bool,

    /// Commit without pushing. Overrides `gca.push` for this run.
    #[arg(long, global = true)]
    no_push: bool,

    /// Ask before pushing. Overrides `gca.push` for this run.
    #[arg(long, global = true, conflicts_with = "no_push")]
    confirm_push: bool,

    /// Push to this remote. Overrides `gca.remote` for this run.
    #[arg(long, global = true)]
    remote: Option<String>,

    /// Paths to stage with `git add`. If omitted and nothing is staged, runs `git add -A`.
    #[arg(trailing_var_arg = true)]
    paths: Vec<String>,

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
    /// Preview which files would be staged, without staging them.
    List {
        #[arg(trailing_var_arg = true)]
        paths: Vec<String>,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigCmd {
    /// Set or show the default push behavior: auto | ask | never.
    Push { mode: Option<String> },
    /// Set or show the default push remote, such as origin or upstream.
    Remote { name: Option<String> },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PushMode {
    Auto,
    Ask,
    Never,
}

fn resolve_push_mode(no_push: bool, confirm_push: bool) -> Result<PushMode> {
    if no_push {
        return Ok(PushMode::Never);
    }
    if confirm_push {
        return Ok(PushMode::Ask);
    }
    let config = git::get_config(PUSH_CONFIG_KEY).map(|value| value.to_ascii_lowercase());
    let mode = match config.as_deref() {
        Some("never") | Some("off") | Some("no") => PushMode::Never,
        Some("ask") | Some("confirm") => PushMode::Ask,
        Some("auto") | Some("yes") | None => PushMode::Auto,
        Some(other) => {
            anyhow::bail!(
                "invalid {PUSH_CONFIG_KEY} value {other:?}; expected auto, ask, or never"
            );
        }
    };
    Ok(mode)
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
        Cmd::Config {
            what: ConfigCmd::Push { mode: None },
        } => {
            let cur = git::get_config(PUSH_CONFIG_KEY).unwrap_or_else(|| "auto".to_string());
            println!("{PUSH_CONFIG_KEY} = {cur}");
            Ok(())
        }
        Cmd::Config {
            what: ConfigCmd::Push { mode: Some(m) },
        } => {
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
        Cmd::Config {
            what: ConfigCmd::Remote { name: None },
        } => {
            let cur = git::get_config(REMOTE_CONFIG_KEY).unwrap_or_else(|| "default".to_string());
            println!("{REMOTE_CONFIG_KEY} = {cur}");
            Ok(())
        }
        Cmd::Config {
            what: ConfigCmd::Remote { name: Some(n) },
        } => {
            git::set_config_global(REMOTE_CONFIG_KEY, n).context("failed to update git config")?;
            println!("set {REMOTE_CONFIG_KEY} = {n}");
            Ok(())
        }
        Cmd::List { paths } => {
            let out = git::add_dry_run(paths).context("git add --dry-run failed")?;
            if out.trim().is_empty() {
                println!("nothing to stage");
            } else {
                print!("{out}");
            }
            Ok(())
        }
    }
}

fn resolve_remote(flag: &Option<String>) -> Option<String> {
    flag.clone().or_else(|| git::get_config(REMOTE_CONFIG_KEY))
}

fn run_commit_flow(cli: &Cli) -> Result<()> {
    static EMBEDDED_MODEL: &[u8] = include_bytes!("../../out/model_v2.json");
    let model = match &cli.model {
        Some(p) => {
            Model::load(p).with_context(|| format!("failed to load model from {}", p.display()))?
        }
        None => Model::from_bytes(EMBEDDED_MODEL).context("failed to parse embedded model")?,
    };
    let push_mode = if cli.dry_run {
        PushMode::Never
    } else {
        resolve_push_mode(cli.no_push, cli.confirm_push)?
    };
    let (diff_text, files, stats) = if cli.dry_run {
        git::preview(&cli.paths).context("failed to preview changes")?
    } else {
        if !cli.paths.is_empty() {
            git::add_paths(&cli.paths).context("git add failed")?;
        }

        let mut diff_text = git::staged_diff(&cli.paths).context("failed to read staged diff")?;
        if diff_text.is_empty() {
            if cli.paths.is_empty() {
                println!("no staged changes; running `git add -A`");
                git::add_all()?;
                diff_text = git::staged_diff(&cli.paths).context("failed to read staged diff")?;
            }
            if diff_text.is_empty() {
                println!("nothing to commit");
                return Ok(());
            }
        }

        let (files, stats) = git::staged_summary(&cli.paths)?;
        (diff_text, files, stats)
    };
    if diff_text.is_empty() {
        println!("nothing to commit");
        return Ok(());
    }

    let diff_truncated: String = diff_text.chars().take(20_000).collect();
    let numeric = [
        stats.files_changed as f64,
        stats.additions as f64,
        stats.deletions as f64,
        stats.additions as f64 / (stats.deletions as f64 + 1.0),
    ];
    let feats = model.build_features(&diff_truncated, numeric);
    let probs = model
        .predict_proba(&feats)
        .context("model inference failed")?;
    let mut top = model.topk(&probs, cli.topk.max(1));

    // A heuristic hit is pre-selected even when the model ranks it below top-k.
    let heuristic = heuristics::classify(&files)
        .filter(|label| model.payload.classes.iter().any(|class| class == label));
    let default_idx = match heuristic {
        None => 0,
        Some(hit) => {
            let pos = top.iter().position(|(l, _)| *l == hit);
            pos.unwrap_or_else(|| {
                top.push((hit, model.prob(&probs, hit)));
                top.len() - 1
            })
        }
    };

    println!(
        "Stats: +{} / -{} lines in {} files",
        stats.additions, stats.deletions, stats.files_changed
    );

    let items: Vec<String> = top
        .iter()
        .map(|(l, s)| format!("{:<9} {:5.1}%", l, s * 100.0))
        .collect();

    if cli.dry_run {
        println!("Suggested type: {}", top[default_idx].0);
        for item in &items {
            println!("{item}");
        }
        return Ok(());
    }

    let chosen = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Commit type")
        .items(&items)
        .default(default_idx)
        .interact()?;
    let label = top[chosen].0;

    let description: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt(format!("{label}:"))
        .validate_with(|s: &String| -> std::result::Result<(), &str> {
            if s.trim().is_empty() {
                Err("description cannot be empty")
            } else {
                Ok(())
            }
        })
        .interact_text()?;

    let message = format!("{}: {}", label, description.trim());

    git::commit(&message, &cli.paths).context("git commit failed")?;

    let remote = resolve_remote(&cli.remote);
    let remote_ref = remote.as_deref();

    match push_mode {
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
            if yes {
                git::push(remote_ref).context("git push failed")
            } else {
                Ok(())
            }
        }
        PushMode::Auto => git::push(remote_ref).context("git push failed"),
    }
}
