//! gca: suggests a Conventional Commit type for the changes you are about to
//! commit, then commits them with `git commit`.

mod draft;
mod features;
mod git;
mod heuristics;
mod history;
mod hook;
mod message;
mod model;
mod subject;

use anyhow::{bail, Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use console::style;
use dialoguer::theme::{ColorfulTheme, SimpleTheme, Theme};
use dialoguer::{Completion, Confirm, Input, Select};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use draft::Draft;
use git::FileChange;
use history::ScopeHint;
use model::Model;
use subject::SubjectModel;

const PUSH_KEY: &str = "gca.push";
const REMOTE_KEY: &str = "gca.remote";
const ORDER_KEY: &str = "gca.order";
/// Diffs are cut here before feature extraction, as in training.
const MAX_DIFF_CHARS: usize = 20_000;
/// How many staged files to list before the prompts.
const LIST_FILES: usize = 8;

static EMBEDDED_MODEL: &[u8] = include_bytes!("../../out/model_v2.json");

/// Set once `git commit` starts; from then on git reports what happened.
static COMMITTING: AtomicBool = AtomicBool::new(false);
/// Set by the Ctrl-C handler.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

const EXAMPLES: &str = "\
Examples:
  gca                          commit what is staged
  gca -a                       commit every change to tracked files, like git commit -a
  gca src/auth                 commit only src/auth
  gca -t fix -m \"handle empty diff\"
                               commit without prompts
  gca --dry-run --json         print the suggestions for scripts and editors
  gca hook install             give plain `git commit` messages a type too";

#[derive(Parser, Debug)]
#[command(
    name = "gca",
    version,
    about = "Suggests a Conventional Commit type for your staged changes, then commits them",
    after_help = EXAMPLES,
    args_conflicts_with_subcommands = true
)]
struct Cli {
    /// Commit only these paths, like `git commit <path>...`. New files are included.
    #[arg(value_name = "PATH")]
    paths: Vec<String>,

    /// Stage every change to tracked files first, like `git commit -a`. New files are left out.
    #[arg(short = 'a', long = "all", conflicts_with = "paths")]
    all: bool,

    /// Commit type. Skips the type prompt.
    #[arg(short = 't', long = "type", value_name = "TYPE", value_parser = message::TYPES)]
    kind: Option<String>,

    /// Scope, as in `feat(scope): ...`. Skips the scope prompt; "" means no scope.
    #[arg(long, value_name = "SCOPE")]
    scope: Option<String>,

    /// Subject line. Repeat it to add body paragraphs, like `git commit -m`.
    #[arg(short = 'm', long = "message", value_name = "MSG")]
    message: Vec<String>,

    /// Mark the commit as a breaking change (`feat!: ...`).
    #[arg(short = 'b', long)]
    breaking: bool,

    /// Accept the suggested type and scope instead of asking.
    #[arg(short = 'y', long)]
    yes: bool,

    /// Open the message in your editor before committing (`git commit --edit`).
    #[arg(short = 'e', long)]
    edit: bool,

    /// Skip the pre-commit and commit-msg hooks (`git commit --no-verify`).
    #[arg(short = 'n', long)]
    no_verify: bool,

    /// Add a Signed-off-by trailer (`git commit --signoff`).
    #[arg(short = 's', long)]
    signoff: bool,

    /// Push after committing. Overrides gca.push for this run.
    #[arg(long, conflicts_with_all = ["confirm_push", "no_push"])]
    push: bool,

    /// Ask before pushing. Overrides gca.push for this run.
    #[arg(long, conflicts_with = "no_push")]
    confirm_push: bool,

    /// Do not push. Overrides gca.push for this run.
    #[arg(long)]
    no_push: bool,

    /// Remote to push to. Overrides gca.remote for this run.
    #[arg(long, value_name = "NAME")]
    remote: Option<String>,

    /// Show the files and the ranked suggestions, then exit without staging or committing.
    #[arg(long)]
    dry_run: bool,

    /// With --dry-run, print JSON for scripts and editor integrations.
    #[arg(long, requires = "dry_run")]
    json: bool,

    /// How many suggestions to list.
    #[arg(long, value_name = "N", default_value_t = 3, value_parser = clap::value_parser!(u8).range(1..=11))]
    topk: u8,

    /// Use this model file instead of the built-in one.
    #[arg(long, value_name = "FILE")]
    model: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Show or change settings. They live in git config, globally unless --local.
    Config {
        #[command(subcommand)]
        what: ConfigCmd,
    },
    /// Print a shell completion script, e.g. `gca completions zsh > _gca`.
    Completions {
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
    /// Put the suggested type into messages from plain `git commit`, editors
    /// and git GUIs, with a prepare-commit-msg hook in this repository.
    Hook {
        #[command(subcommand)]
        action: HookCmd,
    },
}

#[derive(Subcommand, Debug)]
enum HookCmd {
    /// Install the hook (in core.hooksPath, if that is set).
    Install,
    /// Remove the hook gca installed.
    Uninstall,
    /// What the hook runs: edit the message git is about to use.
    #[command(hide = true)]
    Run {
        file: PathBuf,
        source: Option<String>,
        commit: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigCmd {
    /// What to do after committing: never push (default), ask, or push (auto).
    Push {
        #[arg(value_parser = ["never", "ask", "auto"])]
        mode: Option<String>,
        /// Store the setting in this repository only.
        #[arg(long)]
        local: bool,
    },
    /// Remote to push to when the branch has no upstream yet.
    Remote {
        name: Option<String>,
        /// Store the setting in this repository only.
        #[arg(long)]
        local: bool,
    },
    /// Which prompt comes first: the type (default) or the subject. Asked
    /// first, the subject also ranks the types.
    Order {
        #[arg(value_parser = ["type-first", "subject-first"])]
        order: Option<String>,
        /// Store the setting in this repository only.
        #[arg(long)]
        local: bool,
    },
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// What is already staged.
    Staged,
    /// `-a`: every change to tracked files.
    All,
    /// Only the paths given on the command line.
    Paths,
}

/// Which prompt comes first (gca.order).
#[derive(Clone, Copy, PartialEq)]
enum Order {
    TypeFirst,
    SubjectFirst,
}

#[derive(Clone, Copy, PartialEq)]
enum PushMode {
    Never,
    Ask,
    Auto,
}

struct Change {
    diff: String,
    files: Vec<FileChange>,
    stats: git::Stats,
}

fn main() -> ExitCode {
    if no_color() {
        console::set_colors_enabled(false);
        console::set_colors_enabled_stderr(false);
    }
    // Ctrl-C: the prompts hide the cursor, so put it back, and let the flow
    // stop at its next step instead of dying halfway through.
    let _ = ctrlc::set_handler(|| {
        INTERRUPTED.store(true, Ordering::SeqCst);
        let _ = console::Term::stderr().show_cursor();
    });
    let cli = Cli::parse();
    let result = match &cli.command {
        Some(Cmd::Config { what }) => config(what),
        Some(Cmd::Completions { shell }) => {
            clap_complete::generate(*shell, &mut Cli::command(), "gca", &mut std::io::stdout());
            Ok(ExitCode::SUCCESS)
        }
        Some(Cmd::Hook { action }) => match action {
            HookCmd::Install => git::ensure_work_tree()
                .and_then(|_| hook::install())
                .map(|_| ExitCode::SUCCESS),
            HookCmd::Uninstall => git::ensure_work_tree()
                .and_then(|_| hook::uninstall())
                .map(|_| ExitCode::SUCCESS),
            HookCmd::Run { file, source, .. } => {
                // A hook that fails would stop the commit; say why and let it through.
                if let Err(e) = hook_run(file, source.as_deref()) {
                    eprintln!("gca hook: {e:#}");
                }
                Ok(ExitCode::SUCCESS)
            }
        },
        None => commit_flow(&cli),
    };
    match result {
        Ok(code) => code,
        Err(e) if INTERRUPTED.load(Ordering::SeqCst) || is_interrupt(&e) => interrupted(),
        Err(e) => {
            eprintln!("{} {e:#}", style("error:").red().bold());
            ExitCode::FAILURE
        }
    }
}

/// Ctrl-C inside a prompt arrives as an interrupted read.
fn is_interrupt(e: &anyhow::Error) -> bool {
    e.chain().any(|cause| {
        let io = cause.downcast_ref::<std::io::Error>().or_else(|| {
            cause
                .downcast_ref::<dialoguer::Error>()
                .map(|dialoguer::Error::IO(io)| io)
        });
        io.is_some_and(|io| io.kind() == std::io::ErrorKind::Interrupted)
    })
}

fn interrupted() -> ExitCode {
    let _ = console::Term::stderr().show_cursor();
    if !COMMITTING.load(Ordering::SeqCst) {
        eprintln!("\naborted; nothing was committed");
    }
    ExitCode::from(130)
}

fn commit_flow(cli: &Cli) -> Result<ExitCode> {
    git::ensure_work_tree()?;
    if let Some(op) = git::operation_in_progress()? {
        bail!("a {op} is in progress; conclude it with `git commit` (or abort it) first");
    }
    // Settle the settings first, so a bad value stops before anything is
    // staged or committed.
    let (push_mode, order) = if cli.dry_run {
        (PushMode::Never, Order::TypeFirst)
    } else {
        (push_mode(cli)?, prompt_order()?)
    };
    let mode = if cli.all {
        Mode::All
    } else if cli.paths.is_empty() {
        Mode::Staged
    } else {
        Mode::Paths
    };

    let Some(change) = read_change(mode, &cli.paths)? else {
        explain_nothing_to_commit(mode)?;
        return Ok(ExitCode::FAILURE);
    };

    let model = load_model(cli.model.as_deref())?;
    let drafted = draft::draft(&change.files, &change.diff);
    // The subject, when it is known before the type: -m, or else the draft.
    let subject = cli
        .message
        .first()
        .map(|m| m.trim())
        .filter(|m| !m.is_empty())
        .or(drafted.as_ref().map(|d| d.subject.as_str()));
    let Ranking {
        types: ranked,
        preselect,
        with_subject,
    } = rank(&model, &change, drafted.as_ref(), subject, cli.topk.into())?;
    let paths: Vec<String> = change.files.iter().map(|f| f.path.clone()).collect();
    let hint = history::scope_hint(&git::recent_log(history::DEPTH), &paths);

    if cli.dry_run {
        if cli.json {
            let used = subject.filter(|_| with_subject);
            print_json(&change, &ranked, preselect, &hint, drafted.as_ref(), used)?;
        } else {
            print!("{}", summary(&change));
            for (i, (label, p)) in ranked.iter().enumerate() {
                let mark = if i == preselect { ">" } else { " " };
                println!("{mark} {label:<9} ({:5.1}%)", p * 100.0);
            }
            if let Some(scope) = &hint.suggestion {
                println!("scope: {scope}");
            }
            if let Some(d) = &drafted {
                println!("subject: {}", d.subject);
            }
        }
        return Ok(ExitCode::SUCCESS);
    }

    let ask_type = cli.kind.is_none() && !cli.yes;
    let ask_scope = cli.scope.is_none() && !cli.yes && hint.repo_uses_scopes;
    let ask_subject = cli.message.is_empty();
    if (ask_type || ask_scope || ask_subject) && !interactive() {
        let mut needed = Vec::new();
        if ask_type {
            needed.push("--type");
        }
        if ask_scope {
            needed.push("--scope (\"\" for none)");
        }
        if ask_subject {
            needed.push("-m");
        }
        let yes = if ask_type || ask_scope {
            "; --yes takes the suggested type and scope"
        } else {
            ""
        };
        bail!(
            "no terminal to ask on; pass {} to commit without prompts{yes}",
            needed.join(", ")
        );
    }
    if INTERRUPTED.load(Ordering::SeqCst) {
        return Ok(interrupted());
    }
    if ask_type || ask_scope || ask_subject {
        eprint!("{}", summary(&change));
    }
    let theme = theme();
    let draft_subject = drafted.as_ref().map(|d| d.subject.as_str());

    // Subject first: ask for it, then rank the types with it.
    let mut typed_subject = None;
    let (ranked, preselect) = if order == Order::SubjectFirst && ask_subject {
        // The type and scope are not chosen yet; the shortest header they
        // could make must fit, and a longer one is caught below.
        let shortest = cli.kind.as_deref().unwrap_or("ci");
        let scope_given = cli
            .scope
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let subject = prompt_subject(&*theme, "Subject", draft_subject, None, |s| {
            message::header(shortest, scope_given, cli.breaking, s)
        })?;
        let ranking = rank(
            &model,
            &change,
            drafted.as_ref(),
            Some(&subject),
            cli.topk.into(),
        )?;
        typed_subject = Some(subject);
        (ranking.types, ranking.preselect)
    } else {
        (ranked, preselect)
    };

    let kind = match &cli.kind {
        Some(k) => k.clone(),
        None if cli.yes => ranked[preselect].0.to_string(),
        None => match choose_type(&*theme, &ranked, preselect)? {
            Some(k) => k,
            None => return Ok(aborted()),
        },
    };

    let scope = match &cli.scope {
        Some(s) => message::normalize_scope(s).map_err(anyhow::Error::msg)?,
        None if cli.yes => hint.suggestion.clone().filter(|_| hint.repo_uses_scopes),
        None if ask_scope => {
            let input: String = Input::with_theme(&*theme)
                .with_prompt("Scope (optional)")
                .with_initial_text(hint.suggestion.clone().unwrap_or_default())
                .allow_empty(true)
                .validate_with(|s: &String| message::normalize_scope(s).map(|_| ()))
                .interact_text()?;
            message::normalize_scope(&input).map_err(anyhow::Error::msg)?
        }
        None => None,
    };

    let (subject, body) = match cli.message.split_first() {
        Some((first, rest)) => (first.trim().to_string(), body_paragraphs(rest)),
        None => {
            let header_for = |s: &str| message::header(&kind, scope.as_deref(), cli.breaking, s);
            let prefix = header_for("");
            let subject = match typed_subject {
                Some(s) if message::check_header_len(&header_for(&s)).is_ok() => s,
                Some(s) => {
                    eprintln!(
                        "{}",
                        style(format!(
                            "with this type and scope the header is over {} characters; shorten the subject",
                            message::MAX_HEADER_LEN
                        ))
                        .yellow()
                    );
                    prompt_subject(&*theme, prompt_label(&prefix), None, Some(&s), header_for)?
                }
                None => prompt_subject(
                    &*theme,
                    prompt_label(&prefix),
                    draft_subject,
                    None,
                    header_for,
                )?,
            };
            (subject, Vec::new())
        }
    };
    message::check_subject(&subject).map_err(anyhow::Error::msg)?;
    let header = message::header(&kind, scope.as_deref(), cli.breaking, &subject);
    message::check_header_len(&header).map_err(anyhow::Error::msg)?;

    if INTERRUPTED.load(Ordering::SeqCst) {
        return Ok(interrupted());
    }
    COMMITTING.store(true, Ordering::SeqCst);
    if mode == Mode::Paths {
        git::add_paths(None, &cli.paths)?;
    }
    let request = git::CommitRequest {
        header: &header,
        body: &body,
        all: mode == Mode::All,
        paths: if mode == Mode::Paths { &cli.paths } else { &[] },
        edit: cli.edit,
        no_verify: cli.no_verify,
        signoff: cli.signoff,
    };
    if git::commit(&request).is_err() {
        // git has already printed the reason, e.g. a hook's output.
        let note = if mode == Mode::Paths {
            " (the paths you named are staged now)"
        } else {
            ""
        };
        eprintln!(
            "{} nothing was committed{note}",
            style("error:").red().bold()
        );
        return Ok(ExitCode::FAILURE);
    }

    let remote = cli.remote.clone().or_else(|| git::get_config(REMOTE_KEY));
    let push = match push_mode {
        PushMode::Never => false,
        PushMode::Auto => true,
        PushMode::Ask if !interactive() => {
            eprintln!("not pushing: gca.push is \"ask\" and there is no terminal to ask on");
            false
        }
        PushMode::Ask => {
            let target = remote.as_deref().unwrap_or("the remote");
            Confirm::with_theme(&*theme)
                .with_prompt(format!("Push to {target}?"))
                .default(true)
                .interact_opt()?
                == Some(true)
        }
    };
    if push {
        git::push(remote.as_deref()).context("committed, but the push failed")?;
    }
    Ok(ExitCode::SUCCESS)
}

/// The change about to be committed. `-a` and paths are previewed in a
/// scratch index, so nothing the user staged moves until the commit runs.
fn read_change(mode: Mode, paths: &[String]) -> Result<Option<Change>> {
    let scratch = match mode {
        Mode::Staged => None,
        Mode::All | Mode::Paths => Some(git::ScratchIndex::new()?),
    };
    let index = scratch.as_ref().map(git::ScratchIndex::path);
    match (mode, index) {
        (Mode::All, Some(index)) => git::add_tracked(index)?,
        (Mode::Paths, _) => git::add_paths(index, paths)?,
        _ => {}
    }
    let diff = git::staged_diff(index, paths)?;
    if diff.is_empty() {
        return Ok(None);
    }
    Ok(Some(Change {
        files: git::staged_files(index, paths)?,
        stats: git::staged_stats(index, paths)?,
        diff,
    }))
}

/// `gca hook run`: put the suggested `type(scope): ` in front of the
/// message git is about to use, as `gca -y` would choose it.
fn hook_run(file: &Path, source: Option<&str>) -> Result<()> {
    let Some(source) = hook::Source::from_git(source) else {
        return Ok(());
    };
    if std::env::var("GCA_HOOK").is_ok_and(|v| matches!(v.trim(), "0" | "off" | "false" | "no")) {
        return Ok(());
    }
    let text = std::fs::read_to_string(file)
        .with_context(|| format!("could not read {}", file.display()))?;
    let first = text.lines().next().unwrap_or("").trim();
    if hook::keeps(first)
        || (source == hook::Source::Message && first.is_empty())
        || git::replaying()?
    {
        return Ok(());
    }
    let Some(change) = read_change(Mode::Staged, &[])? else {
        return Ok(());
    };
    let model = load_model(None)?;
    let drafted = draft::draft(&change.files, &change.diff);
    let draft_subject = drafted.as_ref().map(|d| d.subject.as_str());
    let subject = match source {
        hook::Source::Message => Some(first),
        hook::Source::Editor => draft_subject,
    };
    let ranking = rank(&model, &change, drafted.as_ref(), subject, 3)?;
    let paths: Vec<String> = change.files.iter().map(|f| f.path.clone()).collect();
    let hint = history::scope_hint(&git::recent_log(history::DEPTH), &paths);
    let scope = hint.suggestion.filter(|_| hint.repo_uses_scopes);
    let kind = ranking.types[ranking.preselect].0;
    let prefix = message::header(kind, scope.as_deref(), false, "");
    let note = (source == hook::Source::Editor && hook::comments_are_stripped())
        .then(|| hook::note(&ranking.types));
    if let Some(edited) = hook::edit(&text, source, &prefix, draft_subject, note.as_deref()) {
        std::fs::write(file, edited)
            .with_context(|| format!("could not write {}", file.display()))?;
    }
    Ok(())
}

fn load_model(path: Option<&Path>) -> Result<Model> {
    match path {
        Some(p) => Model::load(p).with_context(|| format!("could not load {}", p.display())),
        None => Model::from_bytes(EMBEDDED_MODEL).context("the built-in model is damaged"),
    }
}

struct Ranking<'m> {
    /// The top suggestions and their probabilities.
    types: Vec<(&'m str, f64)>,
    /// Which one to pre-select.
    preselect: usize,
    /// Whether the subject was taken into account.
    with_subject: bool,
}

/// A known subject's model is combined with the diff model's. A path rule
/// (docs, test, ci) or the type a subject draft implies (a release is
/// `chore`) is pre-selected even when ranked lower, as long as the model
/// knows that type.
fn rank<'m>(
    model: &'m Model,
    change: &Change,
    drafted: Option<&Draft>,
    subject: Option<&str>,
    topk: usize,
) -> Result<Ranking<'m>> {
    let diff: String = change.diff.chars().take(MAX_DIFF_CHARS).collect();
    let s = &change.stats;
    let numeric = [
        s.files_changed as f64,
        s.additions as f64,
        s.deletions as f64,
        s.additions as f64 / (s.deletions as f64 + 1.0),
    ];
    let mut probs = model
        .predict_proba(&model.build_features(&diff, numeric))
        .context("model inference failed")?;
    let mut with_subject = false;
    if let Some(subject) = subject {
        let subjects = SubjectModel::embedded().context("the built-in subject model is invalid")?;
        if let Some(combined) = subjects.combine(&model.payload.classes, &probs, subject) {
            probs = combined;
            with_subject = true;
        }
    }
    let mut ranked = model.topk(&probs, topk);
    let paths: Vec<String> = change.files.iter().map(|f| f.path.clone()).collect();
    let rule = heuristics::classify(&paths)
        .or_else(|| drafted.and_then(|d| d.kind))
        .filter(|label| model.payload.classes.iter().any(|class| class == label));
    let preselect = match rule {
        None => 0,
        Some(hit) => ranked
            .iter()
            .position(|(l, _)| *l == hit)
            .unwrap_or_else(|| {
                ranked.push((hit, model.prob(&probs, hit)));
                ranked.len() - 1
            }),
    };
    Ok(Ranking {
        types: ranked,
        preselect,
        with_subject,
    })
}

/// The ranked types, plus "other type" for the rest. `None` if cancelled.
fn choose_type(
    theme: &dyn Theme,
    ranked: &[(&str, f64)],
    preselect: usize,
) -> Result<Option<String>> {
    let others: Vec<&str> = message::TYPES
        .iter()
        .copied()
        .filter(|t| !ranked.iter().any(|(l, _)| l == t))
        .collect();
    let mut items: Vec<String> = ranked
        .iter()
        .map(|(label, p)| format!("{label:<9} ({:5.1}%)", p * 100.0))
        .collect();
    if !others.is_empty() {
        items.push("other type…".into());
    }
    let Some(i) = Select::with_theme(theme)
        .with_prompt("Commit type")
        .items(&items)
        .default(preselect)
        .interact_opt()?
    else {
        return Ok(None);
    };
    if i < ranked.len() {
        return Ok(Some(ranked[i].0.to_string()));
    }
    Ok(Select::with_theme(theme)
        .with_prompt("Commit type")
        .items(&others)
        .default(0)
        .interact_opt()?
        .map(|j| others[j].to_string()))
}

fn body_paragraphs(extra: &[String]) -> Vec<String> {
    extra
        .iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// `3 files  +42 -7` and the first few paths with their status letters.
fn summary(change: &Change) -> String {
    let s = &change.stats;
    let noun = if s.files_changed == 1 {
        "file"
    } else {
        "files"
    };
    let mut out = format!(
        "{} {noun}  {} {}\n",
        s.files_changed,
        style(format!("+{}", s.additions)).green(),
        style(format!("-{}", s.deletions)).red()
    );
    for f in change.files.iter().take(LIST_FILES) {
        let letter = match f.status {
            'A' => style(f.status).green(),
            'D' => style(f.status).red(),
            'R' | 'C' => style(f.status).cyan(),
            _ => style(f.status).yellow(),
        };
        out.push_str(&format!("  {letter} {}\n", f.path));
    }
    if change.files.len() > LIST_FILES {
        out.push_str(&format!(
            "  … and {} more\n",
            change.files.len() - LIST_FILES
        ));
    }
    out
}

fn print_json(
    change: &Change,
    ranked: &[(&str, f64)],
    preselect: usize,
    hint: &ScopeHint,
    drafted: Option<&Draft>,
    ranked_with_subject: Option<&str>,
) -> Result<()> {
    let files: Vec<_> = change
        .files
        .iter()
        .map(|f| {
            let mut entry = serde_json::json!({ "status": f.status.to_string(), "path": f.path });
            if let Some(old) = &f.old_path {
                entry["from"] = old.as_str().into();
            }
            entry
        })
        .collect();
    let suggestions: Vec<_> = ranked
        .iter()
        .map(|(label, p)| serde_json::json!({ "type": label, "probability": p }))
        .collect();
    let out = serde_json::json!({
        "files": files,
        "additions": change.stats.additions,
        "deletions": change.stats.deletions,
        "suggestions": suggestions,
        "preselected": ranked[preselect].0,
        "scope": hint.suggestion,
        "repo_uses_scopes": hint.repo_uses_scopes,
        "subject_draft": drafted.map(|d| d.subject.as_str()),
        "ranked_with_subject": ranked_with_subject,
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn explain_nothing_to_commit(mode: Mode) -> Result<()> {
    let reason = match mode {
        Mode::Staged => "no changes are staged",
        Mode::All => "no tracked file has changed",
        Mode::Paths => "the paths you named have no changes",
    };
    eprintln!("nothing to commit: {reason}");
    let status = git::status_short()?;
    let lines: Vec<&str> = status.lines().collect();
    if lines.is_empty() {
        eprintln!("the working tree is clean");
        return Ok(());
    }
    eprintln!();
    for line in lines.iter().take(10) {
        eprintln!("  {line}");
    }
    if lines.len() > 10 {
        eprintln!("  … and {} more", lines.len() - 10);
    }
    eprintln!();
    match mode {
        Mode::Staged => {
            eprintln!("stage what belongs in this commit, then run gca again:");
            eprintln!("  git add <path>...    stage files (git add -p picks hunks)");
            eprintln!("  gca -a               or commit every change to tracked files");
            eprintln!("  gca <path>...        or commit only these paths");
        }
        Mode::All if lines.iter().any(|l| l.starts_with("??")) => {
            eprintln!("-a leaves new files out; add them with `git add <path>`, or name them: gca <path>...");
        }
        _ => {}
    }
    Ok(())
}

fn push_mode(cli: &Cli) -> Result<PushMode> {
    if cli.push {
        return Ok(PushMode::Auto);
    }
    if cli.confirm_push {
        return Ok(PushMode::Ask);
    }
    if cli.no_push {
        return Ok(PushMode::Never);
    }
    let value = git::get_config(PUSH_KEY).map(|v| v.to_ascii_lowercase());
    Ok(match value.as_deref() {
        None | Some("never" | "no" | "off" | "false") => PushMode::Never,
        Some("ask" | "confirm") => PushMode::Ask,
        Some("auto" | "yes" | "on" | "true") => PushMode::Auto,
        Some(other) => bail!("invalid {PUSH_KEY} value {other:?}; expected never, ask or auto"),
    })
}

fn prompt_order() -> Result<Order> {
    let value = git::get_config(ORDER_KEY).map(|v| v.to_ascii_lowercase());
    Ok(match value.as_deref() {
        None | Some("type-first" | "type") => Order::TypeFirst,
        Some("subject-first" | "subject") => Order::SubjectFirst,
        Some(other) => {
            bail!("invalid {ORDER_KEY} value {other:?}; expected type-first or subject-first")
        }
    })
}

fn config(what: &ConfigCmd) -> Result<ExitCode> {
    let (key, value, local, unset) = match what {
        ConfigCmd::Push { mode, local } => (PUSH_KEY, mode, *local, "never (default)"),
        ConfigCmd::Remote { name, local } => (
            REMOTE_KEY,
            name,
            *local,
            "(not set: the upstream, then origin)",
        ),
        ConfigCmd::Order { order, local } => (ORDER_KEY, order, *local, "type-first (default)"),
    };
    match value {
        None => {
            let current = git::get_config(key).unwrap_or_else(|| unset.to_string());
            println!("{key} = {current}");
        }
        Some(v) => {
            if local {
                git::ensure_work_tree()?;
            }
            git::set_config(key, v, local)?;
            let place = if local {
                "this repository"
            } else {
                "global config"
            };
            println!("{key} = {v}  ({place})");
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn aborted() -> ExitCode {
    eprintln!("aborted; nothing was committed");
    ExitCode::FAILURE
}

fn interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

fn no_color() -> bool {
    std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty())
}

fn theme() -> Box<dyn Theme> {
    if no_color() {
        Box::new(SimpleTheme)
    } else {
        Box::new(ColorfulTheme::default())
    }
}

/// The subject prompt's label: `type(scope):`, which the plain theme follows
/// with its own ": ".
fn prompt_label(prefix: &str) -> &str {
    let prefix = prefix.trim_end();
    if no_color() {
        prefix.trim_end_matches(':')
    } else {
        prefix
    }
}

/// Asks for the subject; `header_for` builds the header it is checked in.
/// A draft is the default: Enter takes it, typing replaces it, and Tab puts
/// it on the line to edit. `initial` puts text on the line instead.
fn prompt_subject(
    theme: &dyn Theme,
    label: &str,
    draft: Option<&str>,
    initial: Option<&str>,
    header_for: impl Fn(&str) -> String,
) -> Result<String> {
    let completion = draft.filter(|_| initial.is_none()).map(EditDraft);
    let mut input = Input::<String>::with_theme(theme).with_prompt(label);
    if let Some(c) = &completion {
        input = input.default(c.0.to_string()).completion_with(c);
    }
    if let Some(text) = initial {
        input = input.with_initial_text(text);
    }
    let subject = input
        .validate_with(|s: &String| -> Result<(), String> {
            message::check_subject(s)?;
            message::check_header_len(&header_for(s))
        })
        .interact_text()?;
    Ok(subject.trim().to_string())
}

/// Tab (or → at the end of an empty line) fills in the subject draft.
struct EditDraft<'a>(&'a str);

impl Completion for EditDraft<'_> {
    fn get(&self, input: &str) -> Option<String> {
        // Only on an empty line: dialoguer redraws from the cursor, which
        // would be wrong anywhere but the end.
        input.is_empty().then(|| self.0.to_string())
    }
}
