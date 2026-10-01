//! Subject drafts written by a small sequence-to-sequence model
//! (CodeT5-small fine-tuned on Conventional Commits, see draft_model/), for
//! any change, not only the mechanical ones the rules in draft.rs cover.
//!
//! The model reads the chosen type and scope, the files, the headers of a
//! few earlier commits and the changed lines ([`input`]), and writes the
//! subject. Its draft is offered only when it is sure enough of it
//! ([`THRESHOLD`]); otherwise gca falls back to the rules.
//!
//! The model runs only in builds with the `t5` feature, from a file the
//! user downloads and names ([`model_path`]). It loads and drafts in the
//! background while the type and scope prompts are open, for the type and
//! scope gca suggests; another choice drafts again.

use std::path::{Path, PathBuf};

use crate::history::LogEntry;

pub mod input;
#[cfg(feature = "t5")]
pub mod runtime;

/// git config key and environment variable naming the model file.
pub const MODEL_KEY: &str = "gca.draftModel";
pub const MODEL_ENV: &str = "GCA_DRAFT_MODEL";

/// The lowest mean log-probability per token at which a draft is offered.
/// On 10,005 commits from projects held out from training, 14.5% of drafts
/// reached it; of those, 34% were the author's subject exactly and 56%
/// saved at least half the typing (draft_model/README.md).
pub const THRESHOLD: f32 = -0.3;

/// A subject the model wrote, and how sure it was: the mean log-probability
/// of its tokens, 0 for certain.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDraft {
    pub subject: String,
    pub confidence: f32,
}

impl ModelDraft {
    pub fn is_confident(&self) -> bool {
        self.confidence >= THRESHOLD && !self.subject.trim().is_empty()
    }
}

/// The model file: `--draft-model`, else GCA_DRAFT_MODEL, else gca.draftModel.
/// An empty GCA_DRAFT_MODEL turns the model off. Always `None` in builds
/// without the model.
pub fn model_path(flag: Option<&Path>) -> Option<PathBuf> {
    if !cfg!(feature = "t5") {
        return None;
    }
    if let Some(path) = flag {
        return Some(path.to_path_buf());
    }
    if let Some(value) = std::env::var_os(MODEL_ENV) {
        return (!value.is_empty()).then(|| PathBuf::from(value));
    }
    crate::git::get_config_path(MODEL_KEY).map(PathBuf::from)
}

/// What the model needs besides the type and scope.
#[cfg_attr(not(feature = "t5"), allow(dead_code))]
pub struct Change {
    pub diff: String,
    pub log: Vec<LogEntry>,
    pub staged: Vec<String>,
    pub me: Option<String>,
}

impl Change {
    #[cfg_attr(not(feature = "t5"), allow(dead_code))]
    fn input(&self, kind: &str, scope: Option<&str>) -> String {
        let history = input::history(&self.log, &self.staged, kind, self.me.as_deref());
        input::format(&self.diff, kind, scope, &history)
    }
}

/// A draft being written in the background.
pub struct Background {
    #[cfg(feature = "t5")]
    path: PathBuf,
    #[cfg(feature = "t5")]
    guess: (String, Option<String>),
    #[cfg(feature = "t5")]
    work: std::thread::JoinHandle<anyhow::Result<(runtime::Drafter, Change, ModelDraft)>>,
}

impl Background {
    /// Loads the model and drafts for `kind` and `scope` on another thread.
    #[cfg(feature = "t5")]
    pub fn start(path: PathBuf, change: Change, kind: &str, scope: Option<&str>) -> Self {
        let guess = (kind.to_string(), scope.map(String::from));
        let (file, (k, s)) = (path.clone(), guess.clone());
        let work = std::thread::spawn(move || {
            let drafter = runtime::Drafter::load(&file)?;
            let draft = drafter.draft(&change.input(&k, s.as_deref()))?;
            Ok((drafter, change, draft))
        });
        Background { path, guess, work }
    }

    #[cfg(not(feature = "t5"))]
    pub fn start(_path: PathBuf, _change: Change, _kind: &str, _scope: Option<&str>) -> Self {
        Background {}
    }

    /// The draft for the type and scope finally chosen: the one written in
    /// the background if they are the ones it guessed, or a new one. `None`
    /// if the model could not be used; the reason goes to stderr.
    pub fn finish(self, kind: &str, scope: Option<&str>) -> Option<ModelDraft> {
        #[cfg(feature = "t5")]
        {
            let result = self
                .work
                .join()
                .map_err(|_| anyhow::anyhow!("it stopped unexpectedly"))
                .and_then(|r| r)
                .and_then(|(drafter, change, draft)| {
                    if (kind, scope) == (self.guess.0.as_str(), self.guess.1.as_deref()) {
                        Ok(draft)
                    } else {
                        drafter.draft(&change.input(kind, scope))
                    }
                });
            match result {
                Ok(draft) => Some(draft),
                Err(e) => {
                    eprintln!(
                        "{} could not use the subject model {}: {e:#}",
                        console::style("warning:").yellow().bold(),
                        self.path.display()
                    );
                    None
                }
            }
        }
        #[cfg(not(feature = "t5"))]
        {
            let _ = (kind, scope);
            None
        }
    }
}
