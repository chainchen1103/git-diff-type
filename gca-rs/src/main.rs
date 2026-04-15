mod features;
mod git;
mod heuristics;
mod model;

use anyhow::{Context, Result};
use clap::Parser;
use dialoguer::{theme::ColorfulTheme, Select};
use std::path::PathBuf;

use model::Model;

#[derive(Parser, Debug)]
#[command(name = "gca", about = "Git commit type analyzer")]
struct Args {
    /// Override the embedded model with an external JSON file.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Look at unstaged changes instead of staged.
    #[arg(long)]
    unstaged: bool,

    /// Skip the interactive picker and print the top suggestion only.
    #[arg(long)]
    dry_run: bool,

    /// Number of top suggestions to show.
    #[arg(long, default_value_t = 3)]
    topk: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let cached = !args.unstaged;
    let diff_text = git::diff(cached).context("failed to read git diff")?;
    if diff_text.is_empty() {
        let target = if cached { "staged" } else { "unstaged" };
        println!("no {target} changes found");
        if cached {
            println!("   (try 'git add <file>' first, or pass --unstaged)");
        }
        return Ok(());
    }

    let files = git::staged_files(cached)?;
    let stats = git::stats(cached)?;

    if let Some(hit) = heuristics::classify(&files) {
        println!("\n{}", "=".repeat(40));
        println!("Suggested type: \x1b[1;32m{}\x1b[0m  (heuristic)", hit.label);
        println!("Reason: {}", hit.reason);
        println!("{}", "=".repeat(40));
        print_commit_line(hit.label);
        return Ok(());
    }

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

    println!(
        "\nStats: +{} / -{} lines in {} files",
        stats.additions, stats.deletions, stats.files_changed
    );

    if args.dry_run {
        let (label, score) = top[0];
        println!("\n{}", "=".repeat(40));
        println!("Suggested type: \x1b[1;32m{label}\x1b[0m  ({:.1}%)", score * 100.0);
        println!("{}", "=".repeat(40));
        print_commit_line(label);
        return Ok(());
    }

    let items: Vec<String> = top
        .iter()
        .map(|(l, s)| format!("{:<9} ({:5.1}%)", l, s * 100.0))
        .collect();

    let chosen = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Pick commit type")
        .items(&items)
        .default(0)
        .interact()?;

    print_commit_line(top[chosen].0);
    Ok(())
}

fn print_commit_line(label: &str) {
    println!("\nReady to commit? Copy this:\n");
    println!("git commit -m \"{label}: <description>\"");
}
