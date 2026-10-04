//! gca: suggests a Conventional Commit type for the changes you are about to
//! commit, then commits them with `git commit`.

mod commitlint;
mod draft;
mod features;
mod git;
mod heuristics;
mod history;
mod hook;
mod message;
mod model;
mod subject;
mod t5draft;

use anyhow::{bail, Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use console::style;
use dialoguer::theme::{ColorfulTheme, SimpleTheme, Theme};
use dialoguer::{Completion, Confirm, Input, Select};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use commitlint::Rules;
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

    /// Commit type, such as feat or fix (or one the project's commitlint
    /// config allows). Skips the type prompt.
    #[arg(short = 't', long = "type", value_name = "TYPE", value_parser = type_word)]
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

    /// Draft the subject with this model file (overrides gca.draftModel and
    /// GCA_DRAFT_MODEL; see `gca config draft-model`).
    #[cfg(feature = "t5")]
    #[arg(long, value_name = "FILE")]
    draft_model: Option<PathBuf>,

    /// Rank the types with this type model file too (overrides gca.typeModel
    /// and GCA_TYPE_MODEL; see `gca config type-model`).
    #[cfg(feature = "t5")]
    #[arg(long, value_name = "FILE")]
    type_model: Option<PathBuf>,

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
    /// Make and check the subject model's file (for developers).
    #[cfg(feature = "t5")]
    #[command(hide = true)]
    DraftModel {
        #[command(subcommand)]
        action: DraftModelCmd,
    },
    /// Make and check the type model's file (for developers).
    #[cfg(feature = "t5")]
    #[command(hide = true)]
    TypeModel {
        #[command(subcommand)]
        action: TypeModelCmd,
    },
}

#[cfg(feature = "t5")]
#[derive(Subcommand, Debug)]
enum TypeModelCmd {
    /// Write a checkpoint folder of draft_model/train_type.py (the encoder,
    /// head.pt, types.json, tokenizer.json) as the model file gca loads.
    Convert {
        checkpoint: PathBuf,
        out: PathBuf,
        /// Keep every weight in 32-bit floats, to check the runtime
        /// against transformers.
        #[arg(long)]
        float: bool,
    },
    /// Print each type's probability for each JSON line {"id", "i"} of model
    /// inputs, one JSON line each, with the time it took.
    Bench {
        model: PathBuf,
        inputs: PathBuf,
        /// Only the first N inputs.
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[cfg(feature = "t5")]
#[derive(Subcommand, Debug)]
enum DraftModelCmd {
    /// Write a fine-tuned checkpoint folder (config.json, model.safetensors,
    /// tokenizer.json) as the model file gca loads.
    Convert {
        checkpoint: PathBuf,
        out: PathBuf,
        /// Keep every weight in 32-bit floats (four times the size), to
        /// check the runtime against transformers.
        #[arg(long)]
        float: bool,
    },
    /// Draft a subject for each JSON line {"id", "i"} of model inputs and
    /// print one JSON line each, with the time it took.
    Bench {
        model: PathBuf,
        inputs: PathBuf,
        /// Only the first N inputs.
        #[arg(long)]
        limit: Option<usize>,
        /// Also print the input's token ids.
        #[arg(long)]
        ids: bool,
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
    /// The subject model file: gca drafts subjects with it when it is sure
    /// enough of one. "" turns it off.
    #[cfg(feature = "t5")]
    DraftModel {
        path: Option<String>,
        /// Store the setting in this repository only.
        #[arg(long)]
        local: bool,
    },
    /// The type model file: gca ranks the types with it and the built-in
    /// model together. "" turns it off.
    #[cfg(feature = "t5")]
    TypeModel {
        path: Option<String>,
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
        #[cfg(feature = "t5")]
        Some(Cmd::DraftModel { action }) => draft_model_cmd(action),
        #[cfg(feature = "t5")]
        Some(Cmd::TypeModel { action }) => type_model_cmd(action),
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

#[cfg(feature = "t5")]
fn draft_model_cmd(action: &DraftModelCmd) -> Result<ExitCode> {
    use std::io::{BufRead, Write};
    match action {
        DraftModelCmd::Convert {
            checkpoint,
            out,
            float,
        } => {
            let done = t5draft::runtime::convert(checkpoint, out, !float)?;
            println!(
                "wrote {} ({:.1} MB): {} tensors, {} of them in 8-bit blocks",
                out.display(),
                done.bytes as f64 / 1e6,
                done.tensors,
                done.quantized
            );
        }
        DraftModelCmd::Bench {
            model,
            inputs,
            limit,
            ids,
        } => {
            let t0 = std::time::Instant::now();
            let drafter = t5draft::runtime::Drafter::load(model)?;
            eprintln!("loaded in {} ms", t0.elapsed().as_millis());
            let file = std::fs::File::open(inputs)
                .with_context(|| format!("could not open {}", inputs.display()))?;
            let mut out = std::io::stdout().lock();
            for line in std::io::BufReader::new(file)
                .lines()
                .take(limit.unwrap_or(usize::MAX))
            {
                let row: serde_json::Value = serde_json::from_str(&line?)?;
                let input = row["i"].as_str().context("an input line has no \"i\"")?;
                let t0 = std::time::Instant::now();
                let tokens = drafter.tokens(input)?;
                let done = drafter.generate(&tokens)?;
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let mut rec = serde_json::json!({
                    "id": row["id"], "greedy": done.draft.subject, "confidence": done.draft.confidence,
                    "ms": ms, "in_tokens": tokens.len(), "out_tokens": done.steps,
                });
                if *ids {
                    rec["ids"] = serde_json::json!(tokens);
                }
                writeln!(out, "{rec}")?;
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(feature = "t5")]
fn type_model_cmd(action: &TypeModelCmd) -> Result<ExitCode> {
    use std::io::{BufRead, Write};
    match action {
        TypeModelCmd::Convert {
            checkpoint,
            out,
            float,
        } => {
            let done = t5draft::runtime::convert_classifier(checkpoint, out, !float)?;
            println!(
                "wrote {} ({:.1} MB): {} tensors, {} of them in 8-bit blocks",
                out.display(),
                done.bytes as f64 / 1e6,
                done.tensors,
                done.quantized
            );
        }
        TypeModelCmd::Bench {
            model,
            inputs,
            limit,
        } => {
            let t0 = std::time::Instant::now();
            let classifier = t5draft::runtime::Classifier::load(model)?;
            eprintln!("loaded in {} ms", t0.elapsed().as_millis());
            let file = std::fs::File::open(inputs)
                .with_context(|| format!("could not open {}", inputs.display()))?;
            let mut out = std::io::stdout().lock();
            for line in std::io::BufReader::new(file)
                .lines()
                .take(limit.unwrap_or(usize::MAX))
            {
                let row: serde_json::Value = serde_json::from_str(&line?)?;
                let input = row["i"].as_str().context("an input line has no \"i\"")?;
                let t0 = std::time::Instant::now();
                let tokens = classifier.tokens(input)?;
                let probs = classifier.probabilities_of(&tokens)?;
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let p: serde_json::Map<String, serde_json::Value> = classifier
                    .classes()
                    .iter()
                    .zip(&probs)
                    .map(|(c, &v)| (c.clone(), serde_json::json!(v)))
                    .collect();
                writeln!(
                    out,
                    "{}",
                    serde_json::json!({"id": row["id"], "p": p, "ms": ms, "in_tokens": tokens.len()})
                )?;
            }
        }
    }
    Ok(ExitCode::SUCCESS)
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
    let rules = project_rules();
    if let Some(kind) = cli.kind.as_deref().filter(|k| !rules.allows(k)) {
        Cli::command()
            .error(
                clap::error::ErrorKind::InvalidValue,
                format!(
                    "unknown type {kind:?}; the types are {}",
                    rules.type_list().join(", ")
                ),
            )
            .exit();
    }
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
    // The type model, if there is one and the type is to be suggested,
    // loads while the history is read.
    let type_model = type_model_path(cli)
        .filter(|_| cli.kind.is_none() || cli.dry_run)
        .map(t5draft::TypeModel::start);

    let model = load_model(cli.model.as_deref())?;
    let drafted = draft::draft(&change.files, &change.diff);
    // The subject, when it is known before the type: -m, or else the draft.
    let subject = cli
        .message
        .first()
        .map(|m| m.trim())
        .filter(|m| !m.is_empty())
        .or(drafted.as_ref().map(|d| d.subject.as_str()));
    let log = git::recent_log(history::DEPTH);
    let paths: Vec<String> = change.files.iter().map(|f| f.path.clone()).collect();
    let me = git::author_ident();
    let learned = Learned::read(&model, &log, &paths, me.as_deref());
    let typed = type_model.and_then(|m| m.probabilities(&change.diff, &log, &paths, me.as_deref()));
    let Ranking {
        types: ranked,
        preselect,
        with_subject,
        with_type_model,
        with_history,
        with_file_history,
        with_own_commits,
    } = rank(
        &model,
        &change,
        drafted.as_ref(),
        subject,
        typed.as_deref(),
        &learned,
        &rules,
        cli.topk.into(),
    )?;
    let hint = history::scope_hint(&log, &paths, me.as_deref());

    // The subject model, if there is one, loads and drafts in the background
    // for the type and scope gca suggests while the prompts are open.
    let guess_kind = cli
        .kind
        .clone()
        .unwrap_or_else(|| ranked[preselect].0.to_string());
    let guess_scope = match &cli.scope {
        Some(s) => message::normalize_scope(s).ok().flatten(),
        None => hint.suggestion.clone().filter(|_| hint.repo_uses_scopes),
    };
    let mut background = draft_model_path(cli)
        .filter(|_| cli.message.is_empty() && (cli.dry_run || interactive()))
        .map(|path| {
            let input = t5draft::Change {
                diff: change.diff.clone(),
                log: log.clone(),
                staged: paths.clone(),
                me: me.clone(),
            };
            t5draft::Background::start(path, input, &guess_kind, guess_scope.as_deref())
        });

    if cli.dry_run {
        let model_draft = background
            .take()
            .and_then(|b| b.finish(&guess_kind, guess_scope.as_deref()));
        let offered = offered_draft(model_draft.as_ref(), drafted.as_ref());
        if cli.json {
            let extras = JsonExtras {
                hint: &hint,
                offered,
                model_draft: model_draft
                    .as_ref()
                    .map(|d| (d, guess_kind.as_str(), guess_scope.as_deref())),
                ranked_with_subject: subject.filter(|_| with_subject),
                ranked_with_type_model: with_type_model,
                ranked_with_history: with_history,
                ranked_with_file_history: with_file_history,
                ranked_with_own_commits: with_own_commits,
                rules: &rules,
            };
            print_json(&change, &ranked, preselect, &extras)?;
        } else {
            print!("{}", summary(&change));
            for (i, (label, p)) in ranked.iter().enumerate() {
                let mark = if i == preselect { ">" } else { " " };
                println!("{mark} {label:<9} ({:5.1}%)", p * 100.0);
            }
            if let Some(scope) = &hint.suggestion {
                println!("scope: {scope}");
            }
            if let Some((subject, _)) = offered {
                println!("subject: {subject}");
            }
            if let Some(d) = &model_draft {
                let why = d.why_not_offered().map(|w| format!(", {w}"));
                println!(
                    "model draft: {} (confidence {:.2}{})",
                    d.subject,
                    d.confidence,
                    why.unwrap_or_default()
                );
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

    // Subject first: ask for it, then rank the types with it.
    let mut typed_subject = None;
    let (ranked, preselect) = if order == Order::SubjectFirst && ask_subject {
        // The type and scope are not chosen yet; the shortest header they
        // could make must fit, and a longer one is caught below.
        let shortest = cli.kind.as_deref().unwrap_or_else(|| {
            let types = rules.type_list();
            types.into_iter().min_by_key(|t| t.len()).unwrap_or("ci")
        });
        let scope_given = cli
            .scope
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let model_draft = background
            .take()
            .and_then(|b| b.finish(&guess_kind, guess_scope.as_deref()));
        if INTERRUPTED.load(Ordering::SeqCst) {
            return Ok(interrupted());
        }
        let offered =
            offered_draft(model_draft.as_ref(), drafted.as_ref()).map(|(s, _)| s.to_string());
        let subject = prompt_subject(
            &*theme,
            "Subject",
            offered.as_deref(),
            None,
            rules.header_max(),
            |s| message::header(shortest, scope_given, cli.breaking, s),
        )?;
        let ranking = rank(
            &model,
            &change,
            drafted.as_ref(),
            Some(&subject),
            typed.as_deref(),
            &learned,
            &rules,
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
        None => match choose_type(&*theme, &ranked, preselect, &rules)? {
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
                Some(s)
                    if message::check_header_len(&header_for(&s), rules.header_max()).is_ok() =>
                {
                    s
                }
                Some(s) => {
                    eprintln!(
                        "{}",
                        style(format!(
                            "with this type and scope the header is over {} characters; shorten the subject",
                            rules.header_max()
                        ))
                        .yellow()
                    );
                    prompt_subject(
                        &*theme,
                        prompt_label(&prefix),
                        None,
                        Some(&s),
                        rules.header_max(),
                        header_for,
                    )?
                }
                None => {
                    let model_draft = background
                        .take()
                        .and_then(|b| b.finish(&kind, scope.as_deref()));
                    if INTERRUPTED.load(Ordering::SeqCst) {
                        return Ok(interrupted());
                    }
                    let offered = offered_draft(model_draft.as_ref(), drafted.as_ref())
                        .map(|(s, _)| s.to_string());
                    prompt_subject(
                        &*theme,
                        prompt_label(&prefix),
                        offered.as_deref(),
                        None,
                        rules.header_max(),
                        header_for,
                    )?
                }
            };
            (subject, Vec::new())
        }
    };
    message::check_subject(&subject).map_err(anyhow::Error::msg)?;
    let header = message::header(&kind, scope.as_deref(), cli.breaking, &subject);
    message::check_header_len(&header, rules.header_max()).map_err(anyhow::Error::msg)?;

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
        (Mode::Paths, _) => git::add_paths(index, paths).map_err(|e| unmatched_path(e, paths))?,
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
    let type_model = t5draft::type_model_path(None).map(t5draft::TypeModel::start);
    let model = load_model(None)?;
    let drafted = draft::draft(&change.files, &change.diff);
    let draft_subject = drafted.as_ref().map(|d| d.subject.as_str());
    let subject = match source {
        hook::Source::Message => Some(first),
        hook::Source::Editor => draft_subject,
    };
    let log = git::recent_log(history::DEPTH);
    let paths: Vec<String> = change.files.iter().map(|f| f.path.clone()).collect();
    let me = git::author_ident();
    let learned = Learned::read(&model, &log, &paths, me.as_deref());
    let typed = type_model.and_then(|m| m.probabilities(&change.diff, &log, &paths, me.as_deref()));
    let rules = project_rules();
    let ranking = rank(
        &model,
        &change,
        drafted.as_ref(),
        subject,
        typed.as_deref(),
        &learned,
        &rules,
        3,
    )?;
    let hint = history::scope_hint(&log, &paths, me.as_deref());
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

/// Arguments are paths unless they name a command, so a mistyped command, or
/// one an older gca lacks, reaches git as a path; say that plainly.
fn unmatched_path(e: anyhow::Error, paths: &[String]) -> anyhow::Error {
    let text = format!("{e:#}");
    match paths
        .iter()
        .find(|p| text.contains(&format!("pathspec '{p}' did not match")))
    {
        Some(p) => anyhow::anyhow!(
            "no file matches {p:?}; gca's commands are config, completions and hook (see gca --help)"
        ),
        None => e,
    }
}

/// The type model file, if one is set and this build can run it.
fn type_model_path(cli: &Cli) -> Option<PathBuf> {
    #[cfg(feature = "t5")]
    let flag = cli.type_model.as_deref();
    #[cfg(not(feature = "t5"))]
    let flag = {
        let _ = cli;
        None
    };
    t5draft::type_model_path(flag)
}

/// The subject model file, if one is set and this build can run it.
fn draft_model_path(cli: &Cli) -> Option<PathBuf> {
    #[cfg(feature = "t5")]
    let flag = cli.draft_model.as_deref();
    #[cfg(not(feature = "t5"))]
    let flag = {
        let _ = cli;
        None
    };
    t5draft::model_path(flag)
}

/// The subject draft to offer, and who wrote it: the model's when it is
/// sure enough of it, else the rules' for mechanical changes.
fn offered_draft<'a>(
    model: Option<&'a t5draft::ModelDraft>,
    rules: Option<&'a Draft>,
) -> Option<(&'a str, &'static str)> {
    match model.filter(|d| d.is_offered()) {
        Some(d) => Some((d.subject.as_str(), "model")),
        None => rules.map(|d| (d.subject.as_str(), "rules")),
    }
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
    /// Whether the type model's probabilities were averaged in.
    with_type_model: bool,
    /// How many of the project's recent typed commits were taken into account.
    with_history: usize,
    /// How many of those touched a file in this change.
    with_file_history: usize,
    /// How many of your own recent commits were read again.
    with_own_commits: usize,
}

/// The type model's probabilities, if there are any, are averaged with the
/// diff model's, a known subject's model is combined with them, the result is
/// tilted toward the types the project itself uses, then toward the ones its
/// commits to the same files used, then by the types you chose for your own
/// recent commits, and types its commitlint config does not allow drop out.
/// A path rule
/// (docs, test, ci) or the type a subject draft implies (a release is
/// `chore`) is pre-selected even when ranked lower, as long as the model
/// knows that type.
#[allow(clippy::too_many_arguments)]
fn rank<'m>(
    model: &'m Model,
    change: &Change,
    drafted: Option<&Draft>,
    subject: Option<&str>,
    typed: Option<&[(String, f64)]>,
    learned: &Learned,
    rules: &'m Rules,
    topk: usize,
) -> Result<Ranking<'m>> {
    let Learned { habits, own } = learned;
    let mut probs = diff_probs(model, &change.diff, &change.stats)?;
    let mut with_type_model = false;
    if let Some(mixed) =
        typed.and_then(|t| with_type_model_probs(&model.payload.classes, &probs, t))
    {
        probs = mixed;
        with_type_model = true;
    }
    let subjects = match subject {
        Some(_) => Some(SubjectModel::embedded().context("the built-in subject model is invalid")?),
        None => None,
    };
    let mut with_subject = false;
    if let (Some(subject), Some(subjects)) = (subject, &subjects) {
        if let Some(combined) = subjects.combine(&model.payload.classes, &probs, subject) {
            probs = combined;
            with_subject = true;
        }
    }
    let (weight, file_weight, own_weight) = if with_subject {
        (
            history::HABIT_WEIGHT_WITH_SUBJECT,
            history::FILE_HABIT_WEIGHT_WITH_SUBJECT,
            history::OWN_WEIGHT_WITH_SUBJECT,
        )
    } else {
        (
            history::HABIT_WEIGHT,
            history::FILE_HABIT_WEIGHT,
            history::OWN_WEIGHT,
        )
    };
    let classes = &model.payload.classes;
    let mut with_history = 0;
    if let Some(tilted) = history::weigh_by_habits(
        classes,
        &probs,
        &habits.project,
        weight,
        history::PROJECT_PSEUDO_COMMITS,
    ) {
        probs = tilted;
        with_history = habits.project.values().sum();
    }
    // Then toward the types of the commits that touched the same files.
    let mut with_file_history = 0;
    if let Some(tilted) = history::weigh_by_habits(
        classes,
        &probs,
        &habits.same_files,
        file_weight,
        history::FILE_PSEUDO_COMMITS,
    ) {
        probs = tilted;
        with_file_history = habits.same_files.values().sum();
    }
    // Then by how the types you gave your own recent commits differ from
    // what gca would have shown for them, with the subject if this ranking
    // has one.
    let shown: Vec<Vec<f64>> = own
        .iter()
        .map(|c| match (&subjects, with_subject) {
            (Some(subjects), true) => subjects
                .combine(classes, &c.probs, &c.subject)
                .unwrap_or_else(|| c.probs.clone()),
            _ => c.probs.clone(),
        })
        .collect();
    let chosen: Vec<&str> = own.iter().map(|c| c.kind.as_str()).collect();
    let mut with_own_commits = 0;
    if let Some(tilted) = history::weigh_by_own_choices(
        classes,
        &probs,
        &chosen,
        &shown,
        own_weight,
        history::OWN_PSEUDO_COMMITS,
    ) {
        probs = tilted;
        with_own_commits = own.len();
    }
    // Types the project's commitlint config does not allow drop out.
    if rules.types.is_some() {
        for (p, class) in probs.iter_mut().zip(&model.payload.classes) {
            if !rules.allows(class) {
                *p = 0.0;
            }
        }
        let total: f64 = probs.iter().sum();
        if total > 0.0 {
            probs.iter_mut().for_each(|p| *p /= total);
        }
    }
    let mut ranked: Vec<(&'m str, f64)> = model
        .topk(&probs, topk)
        .into_iter()
        .filter(|(label, _)| rules.allows(label))
        .collect();
    if ranked.is_empty() {
        // The project's types are all unknown to the model: list them as they are.
        ranked = rules
            .type_list()
            .into_iter()
            .take(topk)
            .map(|t| (t, 0.0))
            .collect();
    }
    let paths: Vec<String> = change.files.iter().map(|f| f.path.clone()).collect();
    let rule = heuristics::classify(&paths)
        .or_else(|| drafted.and_then(|d| d.kind))
        .filter(|label| model.payload.classes.iter().any(|class| class == label))
        .filter(|label| rules.allows(label));
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
        with_type_model,
        with_history,
        with_file_history,
        with_own_commits,
    })
}

/// The diff model's and the type model's probabilities averaged in log
/// space, half each, as draft_model/eval_type.py scores them. Your own
/// earlier commits are still read again by the diff model alone: on the
/// held-out commits that ranks as well as reading them with both, without
/// running the type model ten more times. `None` if the type model lacks
/// one of the types.
fn with_type_model_probs(
    classes: &[String],
    probs: &[f64],
    typed: &[(String, f64)],
) -> Option<Vec<f64>> {
    let scores = classes
        .iter()
        .zip(probs)
        .map(|(class, p)| {
            let q = typed.iter().find(|(t, _)| t == class)?.1;
            Some(0.5 * (p + 1e-12).ln() + 0.5 * (q + 1e-12).ln())
        })
        .collect::<Option<Vec<f64>>>()?;
    Some(subject::softmax(&scores))
}

/// The diff model's probabilities for a change.
fn diff_probs(model: &Model, diff: &str, stats: &git::Stats) -> Result<Vec<f64>> {
    let diff: String = diff.chars().take(MAX_DIFF_CHARS).collect();
    let numeric = [
        stats.files_changed as f64,
        stats.additions as f64,
        stats.deletions as f64,
        stats.additions as f64 / (stats.deletions as f64 + 1.0),
    ];
    model
        .predict_proba(&model.build_features(&diff, numeric))
        .context("model inference failed")
}

/// What the recent history says about the type, besides the change itself.
struct Learned {
    habits: history::Habits,
    own: Vec<OwnCommit>,
}

impl Learned {
    fn read(model: &Model, log: &[history::LogEntry], paths: &[String], me: Option<&str>) -> Self {
        Learned {
            habits: history::habits(log, paths),
            own: me.map(|me| own_commits(model, log, me)).unwrap_or_default(),
        }
    }
}

/// One of your recent commits: the type you gave it, its subject, and what
/// the diff model says about its change.
struct OwnCommit {
    kind: String,
    subject: String,
    probs: Vec<f64>,
}

/// Your latest typed commits among the recent ones, read again by the diff
/// model. Empty when you have none; a commit that cannot be read is left out.
fn own_commits(model: &Model, log: &[history::LogEntry], me: &str) -> Vec<OwnCommit> {
    let mine = history::own_commits(log, me, &model.payload.classes);
    if mine.is_empty() {
        return Vec::new();
    }
    let shas: Vec<&str> = mine.iter().map(|(e, _)| e.sha.as_str()).collect();
    let Ok(changes) = git::commit_changes(&shas, MAX_DIFF_CHARS) else {
        return Vec::new();
    };
    mine.into_iter()
        .filter_map(|(entry, kind)| {
            let change = changes.get(&entry.sha)?;
            let probs = diff_probs(model, &change.diff, &change.stats).ok()?;
            Some(OwnCommit {
                kind: kind.to_string(),
                subject: entry.subject.clone(),
                probs,
            })
        })
        .collect()
}

/// The ranked types, plus "other type" for the rest. `None` if cancelled.
fn choose_type(
    theme: &dyn Theme,
    ranked: &[(&str, f64)],
    preselect: usize,
    rules: &Rules,
) -> Result<Option<String>> {
    let others: Vec<&str> = rules
        .type_list()
        .into_iter()
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

/// What the dry run's JSON reports besides the change and the ranking.
struct JsonExtras<'a> {
    hint: &'a ScopeHint,
    /// The subject draft gca offers, and who wrote it ("model" or "rules").
    offered: Option<(&'a str, &'static str)>,
    /// What the model wrote, for which type and scope, offered or not.
    model_draft: Option<(&'a t5draft::ModelDraft, &'a str, Option<&'a str>)>,
    ranked_with_subject: Option<&'a str>,
    ranked_with_type_model: bool,
    ranked_with_history: usize,
    ranked_with_file_history: usize,
    ranked_with_own_commits: usize,
    rules: &'a Rules,
}

fn print_json(
    change: &Change,
    ranked: &[(&str, f64)],
    preselect: usize,
    extras: &JsonExtras,
) -> Result<()> {
    let JsonExtras {
        hint,
        offered,
        model_draft,
        ranked_with_subject,
        ranked_with_type_model,
        ranked_with_history,
        ranked_with_file_history,
        ranked_with_own_commits,
        rules,
    } = extras;
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
        "subject_draft": offered.map(|(s, _)| s),
        "subject_draft_source": offered.map(|(_, who)| who),
        "model_draft": model_draft.map(|(d, kind, scope)| serde_json::json!({
            "subject": d.subject,
            "confidence": d.confidence,
            "offered": d.is_offered(),
            "repeats": d.repeats,
            "type": kind,
            "scope": scope,
        })),
        "ranked_with_subject": ranked_with_subject,
        "ranked_with_type_model": ranked_with_type_model,
        "ranked_with_history": ranked_with_history,
        "ranked_with_file_history": ranked_with_file_history,
        "ranked_with_own_commits": ranked_with_own_commits,
        "commitlint": rules.file.as_ref().map(|file| serde_json::json!({
            "config": file.display().to_string(),
            "types": rules.types,
            "header_max_length": rules.header_max,
        })),
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

/// `-t`: any word that can be a type; the project's rules decide the rest.
fn type_word(word: &str) -> Result<String, String> {
    if commitlint::is_type(word) {
        Ok(word.to_string())
    } else {
        Err("a type is a lowercase word, such as feat or fix".into())
    }
}

/// The project's commitlint rules, found from here up to the top of the
/// work tree; the defaults without a config.
fn project_rules() -> Rules {
    match (std::env::current_dir(), git::toplevel()) {
        (Ok(cwd), Ok(top)) => commitlint::load(&cwd, &top),
        _ => Rules::default(),
    }
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
    #[cfg(feature = "t5")]
    match what {
        ConfigCmd::DraftModel { path, local } => {
            return config_model_file(
                t5draft::MODEL_KEY,
                path.as_deref(),
                *local,
                "(not set: no model drafts)",
                "subjects are drafted by the rules only",
            )
        }
        ConfigCmd::TypeModel { path, local } => {
            return config_model_file(
                t5draft::TYPE_MODEL_KEY,
                path.as_deref(),
                *local,
                "(not set: the built-in model ranks alone)",
                "the built-in model ranks the types alone",
            )
        }
        _ => {}
    }
    let (key, value, local, unset) = match what {
        ConfigCmd::Push { mode, local } => (PUSH_KEY, mode, *local, "never (default)"),
        ConfigCmd::Remote { name, local } => (
            REMOTE_KEY,
            name,
            *local,
            "(not set: the upstream, then origin)",
        ),
        ConfigCmd::Order { order, local } => (ORDER_KEY, order, *local, "type-first (default)"),
        #[cfg(feature = "t5")]
        ConfigCmd::DraftModel { .. } | ConfigCmd::TypeModel { .. } => unreachable!("handled above"),
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

/// `gca config draft-model [PATH]` and `gca config type-model [PATH]`: the
/// path is stored absolute, so it works from any directory; "" removes the
/// setting.
#[cfg(feature = "t5")]
fn config_model_file(
    key: &str,
    path: Option<&str>,
    local: bool,
    unset: &str,
    removed: &str,
) -> Result<ExitCode> {
    let Some(path) = path else {
        match git::get_config_path(key) {
            Some(p) => println!("{key} = {p}"),
            None => println!("{key} = {unset}"),
        }
        return Ok(ExitCode::SUCCESS);
    };
    if local {
        git::ensure_work_tree()?;
    }
    if path.trim().is_empty() {
        git::unset_config(key, local)?;
        println!("{key} removed; {removed}");
        return Ok(ExitCode::SUCCESS);
    }
    let file = std::path::absolute(path).with_context(|| format!("not a usable path: {path}"))?;
    if !file.is_file() {
        bail!("no file at {}", file.display());
    }
    let shown = file.display().to_string();
    git::set_config(key, &shown, local)?;
    let place = if local {
        "this repository"
    } else {
        "global config"
    };
    println!("{key} = {shown}  ({place})");
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
    max_header: usize,
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
            message::check_header_len(&header_for(s), max_header)
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
