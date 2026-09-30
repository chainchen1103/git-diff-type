//! `gca hook install`: a prepare-commit-msg hook, so plain `git commit`,
//! editors and git GUIs get the suggested type too.
//!
//! The hook calls `gca hook run`, which edits the message git is about to
//! use. It never stops a commit: whatever goes wrong, it exits 0.

use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::git;

/// First comment of the hook script; marks the hook as gca's.
pub const MARKER: &str = "# Added by `gca hook install`";

/// The hook script. It prefers the gca that installed it, then any on PATH.
pub fn script(gca: &str) -> String {
    let quoted: String = gca
        .chars()
        .flat_map(|c| match c {
            '"' | '$' | '`' | '\\' => vec!['\\', c],
            c => vec![c],
        })
        .collect();
    format!(
        "#!/bin/sh\n\
         {MARKER}; `gca hook uninstall` removes it.\n\
         # Puts gca's suggested Conventional Commit type in the commit message.\n\
         # It never stops a commit. Skip it for one commit with GCA_HOOK=0.\n\
         gca=\"{quoted}\"\n\
         [ -x \"$gca\" ] || gca=$(command -v gca) || exit 0\n\
         \"$gca\" hook run \"$@\" || true\n"
    )
}

pub fn install() -> Result<()> {
    let path = git::hook_path("prepare-commit-msg")?;
    let existing = std::fs::read_to_string(&path).ok();
    if let Some(text) = &existing {
        if !text.contains(MARKER) {
            bail!(
                "{} already exists and is not gca's; to use both, add this line to it:\n  \
                 gca hook run \"$@\" || true",
                path.display()
            );
        }
    }
    let exe = std::env::current_exe().context("could not find gca's own path")?;
    // Git for Windows runs hooks with its sh, which reads C:/... paths.
    let exe = exe.to_string_lossy().replace('\\', "/");
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("could not create {}", dir.display()))?;
    }
    std::fs::write(&path, script(&exe))
        .with_context(|| format!("could not write {}", path.display()))?;
    make_executable(&path)?;
    let verb = if existing.is_some() {
        "updated"
    } else {
        "installed"
    };
    println!("{verb} {}", path.display());
    Ok(())
}

pub fn uninstall() -> Result<()> {
    let path = git::hook_path("prepare-commit-msg")?;
    match std::fs::read_to_string(&path) {
        Err(_) => println!("no prepare-commit-msg hook in {}", path.display()),
        Ok(text) if !text.contains(MARKER) => {
            bail!("{} is not gca's; leaving it alone", path.display())
        }
        Ok(_) => {
            std::fs::remove_file(&path)
                .with_context(|| format!("could not remove {}", path.display()))?;
            println!("removed {}", path.display());
        }
    }
    Ok(())
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
        .with_context(|| format!("could not make {} executable", path.display()))
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<()> {
    Ok(())
}

/// Where the message came from, as git tells the hook.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Source {
    /// `git commit -m`/`-F`, or a GUI committing its message box: the
    /// message is used as it is, without an editor.
    Message,
    /// Git opens an editor on the message (no source, or a template).
    Editor,
}

impl Source {
    /// `None` for merges, squashes and reused or amended commits, which keep
    /// their own messages.
    pub fn from_git(source: Option<&str>) -> Option<Self> {
        match source {
            None | Some("") | Some("template") => Some(Self::Editor),
            Some("message") => Some(Self::Message),
            _ => None,
        }
    }
}

/// A first line gca leaves alone: it already starts with a type or another
/// `word:` prefix, or git or an autosquash wrote it.
pub fn keeps(first: &str) -> bool {
    let line = first.trim_start();
    let prefixed = line
        .split_once(": ")
        .map(|(head, _)| head.trim_end_matches('!'))
        .map(|head| head.split_once('(').map_or(head, |(word, _)| word))
        .is_some_and(|word| {
            !word.is_empty()
                && word
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        });
    prefixed
        || ["Revert \"", "Merge ", "fixup! ", "squash! ", "amend! "]
            .iter()
            .any(|p| line.starts_with(p))
}

/// The message with the suggested type in front of the subject, or `None` to
/// leave it as it is.
///
/// `prefix` is `type(scope): `. A message from `-m` or a message box gets the
/// prefix in front of its first line. When git opens an editor, an empty
/// first line gets the prefix and the subject draft, if any, and `note` (a
/// comment with the ranking) goes below the first line.
pub fn edit(
    text: &str,
    source: Source,
    prefix: &str,
    draft: Option<&str>,
    note: Option<&str>,
) -> Option<String> {
    let (first, rest) = match text.split_once('\n') {
        Some((first, rest)) => (first, Some(rest)),
        None => (text, None),
    };
    let (first, eol) = match first.strip_suffix('\r') {
        Some(first) => (first, "\r\n"),
        None => (first, "\n"),
    };
    if keeps(first) {
        return None;
    }
    let mut out = match source {
        Source::Message if first.trim().is_empty() => return None,
        Source::Message => format!("{prefix}{}", first.trim()),
        Source::Editor if first.trim().is_empty() => format!("{prefix}{}", draft.unwrap_or("")),
        // A template's first line: keep it, but still show the ranking.
        Source::Editor if note.is_some() => first.to_string(),
        Source::Editor => return None,
    };
    if let (Source::Editor, Some(note)) = (source, note) {
        out.push_str(eol);
        out.push_str(note);
    }
    if let Some(rest) = rest {
        out.push_str(eol);
        out.push_str(rest);
    }
    Some(out)
}

/// `# gca suggests: fix (21%), feat (15%), refactor (13%)`
pub fn note(ranked: &[(&str, f64)]) -> String {
    let list: Vec<String> = ranked
        .iter()
        .map(|(kind, p)| format!("{kind} ({:.0}%)", p * 100.0))
        .collect();
    format!("# gca suggests: {}", list.join(", "))
}

/// Whether git will drop `#` comment lines from an edited message.
pub fn comments_are_stripped() -> bool {
    let char_ok = |key| git::get_config(key).map_or(true, |v| v == "#");
    let cleanup_ok = git::get_config("commit.cleanup")
        .map_or(true, |v| matches!(v.as_str(), "default" | "strip"));
    char_ok("core.commentChar") && char_ok("core.commentString") && cleanup_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIT_TEMPLATE: &str = "\n# Please enter the commit message for your changes. Lines starting\n# with '#' will be ignored, and an empty message aborts the commit.\n#\n# On branch main\n";

    #[test]
    fn a_message_gets_the_type_in_front() {
        assert_eq!(
            edit(
                "make parsing faster\n",
                Source::Message,
                "perf: ",
                None,
                None
            )
            .as_deref(),
            Some("perf: make parsing faster\n")
        );
        assert_eq!(
            edit(
                "rename x\n\nWhy it moved.\n",
                Source::Message,
                "refactor(core): ",
                None,
                Some("# ignored")
            )
            .as_deref(),
            Some("refactor(core): rename x\n\nWhy it moved.\n")
        );
        assert_eq!(
            edit("fix typo\r\n", Source::Message, "docs: ", None, None).as_deref(),
            Some("docs: fix typo\r\n")
        );
        assert_eq!(edit("\n", Source::Message, "fix: ", None, None), None);
    }

    #[test]
    fn typed_or_generated_messages_are_left_alone() {
        for first in [
            "fix: already typed",
            "feat(cli)!: breaking",
            "WIP: not ready",
            "net: fix a leak",
            "Revert \"feat: x\"",
            "Merge branch 'main'",
            "fixup! feat: x",
            "squash! x",
        ] {
            assert!(keeps(first), "{first}");
            assert_eq!(edit(first, Source::Message, "fix: ", None, None), None);
        }
        for first in [
            "fix crash on start",
            "update README: add install steps",
            "call foo(): bar",
        ] {
            assert!(!keeps(first), "{first}");
        }
    }

    #[test]
    fn an_editor_gets_the_prefix_the_draft_and_the_ranking() {
        let note = note(&[("chore", 0.4), ("build", 0.25), ("fix", 0.1)]);
        assert_eq!(note, "# gca suggests: chore (40%), build (25%), fix (10%)");
        let out = edit(
            GIT_TEMPLATE,
            Source::Editor,
            "chore(deps): ",
            Some("bump zod from 3.22.0 to 3.23.8"),
            Some(&note),
        )
        .unwrap();
        assert!(out.starts_with(
            "chore(deps): bump zod from 3.22.0 to 3.23.8\n# gca suggests: chore (40%)"
        ));
        assert!(out.ends_with(GIT_TEMPLATE.trim_start_matches('\n')));
        // no draft: the cursor's line starts with the type
        let out = edit(GIT_TEMPLATE, Source::Editor, "fix: ", None, None).unwrap();
        assert!(out.starts_with("fix: \n# Please enter"));
        // a template with text keeps it
        let out = edit("[JIRA-1] \n", Source::Editor, "fix: ", None, Some("# n")).unwrap();
        assert_eq!(out, "[JIRA-1] \n# n\n");
        assert_eq!(
            edit("[JIRA-1] \n", Source::Editor, "fix: ", None, None),
            None
        );
    }

    #[test]
    fn sources() {
        assert_eq!(Source::from_git(None), Some(Source::Editor));
        assert_eq!(Source::from_git(Some("template")), Some(Source::Editor));
        assert_eq!(Source::from_git(Some("message")), Some(Source::Message));
        for other in ["merge", "squash", "commit"] {
            assert_eq!(Source::from_git(Some(other)), None);
        }
    }

    #[test]
    fn the_script_quotes_the_path() {
        let s = script("/opt/my \"tools\"/gca");
        assert!(s.starts_with("#!/bin/sh\n"));
        assert!(s.contains(MARKER));
        assert!(s.contains("gca=\"/opt/my \\\"tools\\\"/gca\""));
        assert!(s.contains("|| true"));
    }
}
