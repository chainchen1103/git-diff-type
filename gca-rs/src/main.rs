mod features;
mod git;
mod heuristics;
mod model;

use anyhow::{Context, Result};
use clap::Parser;
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select};
use std::path::PathBuf;

use model::Model;

#[derive(Parser, Debug)]
#[command(name = "gca", about = "Git commit type analyzer")]
struct Args {
    /// Override the embedded model with an external JSON file.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Number of top suggestions to show.
    #[arg(long, default_value_t = 3)]
    topk: usize,

    /// Print the suggestion but do not commit or push.
    #[arg(long)]
    dry_run: bool,

    /// Commit but do not push.
    #[arg(long)]
    no_push: bool,

    /// Ask for confirmation before pushing.
    #[arg(long)]
    confirm_push: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

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
    let model = match &args.model {
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
    let top = model.topk(&probs, args.topk.max(1));

    // Path-based heuristic takes the default cursor position when it fires.
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

    if args.dry_run {
        println!("\n(dry-run) git commit -m \"{message}\"");
        return Ok(());
    }

    git::commit(&message).context("git commit failed")?;

    if args.no_push {
        return Ok(());
    }
    if args.confirm_push {
        let yes = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("Push to remote?")
            .default(true)
            .interact()?;
        if !yes {
            return Ok(());
        }
    }
    git::push().context("git push failed")?;
    Ok(())
}
