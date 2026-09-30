//! End-to-end tests: run the real `gca` binary against throwaway repositories.
//! Git is isolated from the machine's own config so user settings such as
//! `gca.push=auto` or `diff.noprefix` cannot leak in.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use tempfile::TempDir;

struct Repo {
    dir: TempDir,
    home: TempDir,
}

impl Repo {
    fn new() -> Self {
        let repo = Repo {
            dir: TempDir::new().unwrap(),
            home: TempDir::new().unwrap(),
        };
        fs::write(repo.home.path().join(".gitconfig"), "").unwrap();
        repo.git(&["init", "-q"]);
        repo.git(&["symbolic-ref", "HEAD", "refs/heads/main"]);
        repo.git(&["config", "user.name", "Test"]);
        repo.git(&["config", "user.email", "test@example.com"]);
        repo.git(&["config", "commit.gpgsign", "false"]);
        repo.write("README.md", "# demo\n");
        repo.write("src/lib.rs", "pub fn one() -> u32 {\n    1\n}\n");
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-q", "-m", "chore: initial commit"]);
        repo
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    fn env(&self, cmd: &mut Command) {
        let home = self.home.path();
        cmd.current_dir(self.path())
            .env("HOME", home)
            .env("USERPROFILE", home)
            .env("XDG_CONFIG_HOME", home)
            .env("GIT_CONFIG_GLOBAL", home.join(".gitconfig"))
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("NO_COLOR", "1")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE");
    }

    fn git(&self, args: &[&str]) -> String {
        let mut cmd = Command::new("git");
        self.env(&mut cmd);
        let out = cmd.args(args).output().unwrap();
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    fn gca(&self, args: &[&str]) -> Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_gca"));
        self.env(&mut cmd);
        cmd.args(args).stdin(Stdio::null()).output().unwrap()
    }

    fn write(&self, rel: &str, content: &str) {
        let p = self.path().join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, content).unwrap();
    }

    fn head_message(&self) -> String {
        self.git(&["log", "-1", "--format=%B"])
            .trim_end()
            .to_string()
    }

    fn head_files(&self) -> Vec<String> {
        self.git(&["show", "--name-only", "--format=", "HEAD"])
            .lines()
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect()
    }

    fn commit_count(&self) -> usize {
        self.git(&["rev-list", "--count", "HEAD"])
            .trim()
            .parse()
            .unwrap()
    }

    fn staged(&self) -> String {
        self.git(&["diff", "--cached", "--name-only"])
    }
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn json(out: &Output) -> serde_json::Value {
    serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("bad JSON ({e}): {}", stdout(out)))
}

#[test]
fn nothing_staged_stages_nothing_and_explains() {
    let repo = Repo::new();
    repo.write("src/lib.rs", "pub fn one() -> u32 {\n    2\n}\n");
    repo.write("notes.txt", "todo\n");

    let out = repo.gca(&["-t", "fix", "-m", "change one"]);

    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(
        err.contains("nothing to commit: no changes are staged"),
        "{err}"
    );
    assert!(err.contains("gca -a"), "{err}");
    assert_eq!(repo.commit_count(), 1);
    assert_eq!(repo.staged(), "");
}

#[test]
fn all_commits_tracked_changes_but_not_new_files() {
    let repo = Repo::new();
    repo.write("src/lib.rs", "pub fn one() -> u32 {\n    2\n}\n");
    repo.write("notes.txt", "todo\n");

    let out = repo.gca(&["-a", "-t", "fix", "-m", "return two"]);

    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(repo.head_message(), "fix: return two");
    assert_eq!(repo.head_files(), ["src/lib.rs"]);
    assert!(repo.git(&["status", "--short"]).contains("?? notes.txt"));
}

#[test]
fn all_and_paths_conflict() {
    let repo = Repo::new();
    let out = repo.gca(&["-a", "src/lib.rs"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn an_unknown_command_is_reported_as_a_missing_path() {
    let repo = Repo::new();
    let out = repo.gca(&["hooks", "install"]);
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(err.contains("no file matches \"hooks\""), "{err}");
    assert!(err.contains("gca --help"), "{err}");
    assert_eq!(repo.commit_count(), 1);
}

#[test]
fn paths_commit_only_those_and_keep_other_staged_work() {
    let repo = Repo::new();
    repo.write("README.md", "# demo\n\nmore\n");
    repo.git(&["add", "README.md"]);
    repo.write("src/new.rs", "pub fn two() {}\n");

    let out = repo.gca(&["src/new.rs", "-t", "feat", "-m", "add two"]);

    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(repo.head_files(), ["src/new.rs"]);
    assert_eq!(repo.staged().trim(), "README.md");
}

#[test]
fn header_scope_breaking_body_and_signoff() {
    let repo = Repo::new();
    repo.write("src/lib.rs", "pub fn one() -> u64 {\n    1\n}\n");
    repo.git(&["add", "-A"]);

    let out = repo.gca(&[
        "-t",
        "feat",
        "--scope",
        "api",
        "-b",
        "-s",
        "-m",
        "return u64",
        "-m",
        "Callers must widen.",
    ]);

    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(
        repo.head_message(),
        "feat(api)!: return u64\n\nCallers must widen.\n\nSigned-off-by: Test <test@example.com>"
    );
}

#[test]
fn rejects_bad_input_without_committing() {
    let repo = Repo::new();
    repo.write("src/lib.rs", "// changed\n");
    repo.git(&["add", "-A"]);

    let long = "x".repeat(120);
    for args in [
        vec!["-t", "fix", "-m", "   "],
        vec!["-t", "fix", "-m", long.as_str()],
        vec!["-t", "fix", "--scope", "a(b)", "-m", "subject"],
    ] {
        let out = repo.gca(&args);
        assert_eq!(out.status.code(), Some(1), "{args:?}: {}", stderr(&out));
    }
    let out = repo.gca(&["-t", "feature", "-m", "subject"]);
    assert_eq!(out.status.code(), Some(2), "unknown type is a usage error");
    assert_eq!(repo.commit_count(), 1);
}

#[test]
fn without_a_terminal_it_asks_for_flags() {
    let repo = Repo::new();
    repo.write("src/lib.rs", "// changed\n");
    repo.git(&["add", "-A"]);

    let out = repo.gca(&[]);

    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("--type"), "{}", stderr(&out));
    assert_eq!(repo.commit_count(), 1);
}

#[test]
fn yes_accepts_the_top_suggestion() {
    let repo = Repo::new();
    repo.write("docs/guide.md", "# Guide\n\nHow to install.\n");
    repo.git(&["add", "-A"]);

    let out = repo.gca(&["-y", "-m", "add install guide"]);

    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(repo.head_message(), "docs: add install guide");
}

#[test]
fn dry_run_json_leaves_the_index_alone() {
    let repo = Repo::new();
    repo.write("src/lib.rs", "pub fn one() -> u32 {\n    3\n}\n");

    let out = repo.gca(&["-a", "--dry-run", "--json"]);

    assert!(out.status.success(), "{}", stderr(&out));
    let v = json(&out);
    assert_eq!(v["files"][0]["path"], "src/lib.rs");
    assert_eq!(v["files"][0]["status"], "M");
    assert_eq!(v["suggestions"].as_array().unwrap().len(), 3);
    assert!(v["preselected"].is_string());
    assert_eq!(repo.staged(), "", "dry run must not stage");
    let leftovers: Vec<_> = fs::read_dir(repo.path().join(".git"))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("gca-index"))
        .collect();
    assert!(leftovers.is_empty(), "scratch index left behind");
}

#[test]
fn mechanical_changes_get_a_subject_draft() {
    let repo = Repo::new();
    let manifest = |version: &str, zod: &str| {
        format!(
            "{{\n  \"name\": \"demo\",\n  \"version\": \"{version}\",\n  \"dependencies\": {{\n    \"zod\": \"^{zod}\"\n  }}\n}}\n"
        )
    };
    repo.write("package.json", &manifest("1.0.0", "3.22.0"));
    repo.git(&["add", "-A"]);
    repo.git(&["commit", "-q", "-m", "chore: add zod"]);

    repo.write("package.json", &manifest("1.0.0", "3.23.8"));
    repo.git(&["add", "-A"]);
    let v = json(&repo.gca(&["--dry-run", "--json"]));
    assert_eq!(v["subject_draft"], "bump zod from 3.22.0 to 3.23.8");
    let text = stdout(&repo.gca(&["--dry-run"]));
    assert!(
        text.contains("subject: bump zod from 3.22.0 to 3.23.8"),
        "{text}"
    );
    repo.git(&["commit", "-q", "-m", "chore(deps): bump zod"]);

    // A release draft also pre-selects chore.
    repo.write("package.json", &manifest("1.1.0", "3.23.8"));
    repo.write("CHANGELOG.md", "## 1.1.0\n\n- Newer zod.\n");
    repo.git(&["add", "-A"]);
    let v = json(&repo.gca(&["--dry-run", "--json"]));
    assert_eq!(v["subject_draft"], "release v1.1.0");
    assert_eq!(v["preselected"], "chore");
    repo.git(&["commit", "-q", "-m", "chore: release v1.1.0"]);

    // A moved file is one rename, even with git's rename detection turned off.
    repo.git(&["config", "diff.renames", "false"]);
    repo.git(&["mv", "src/lib.rs", "src/core.rs"]);
    let v = json(&repo.gca(&["--dry-run", "--json"]));
    assert_eq!(v["subject_draft"], "rename lib.rs to core.rs");
    assert_eq!(
        v["files"],
        serde_json::json!([{"status": "R", "path": "src/core.rs", "from": "src/lib.rs"}])
    );
    assert_eq!(
        (v["additions"].clone(), v["deletions"].clone()),
        (0.into(), 0.into())
    );
    let text = stdout(&repo.gca(&["--dry-run"]));
    assert!(text.starts_with("1 file "), "{text}");
}

#[test]
fn the_subject_steers_the_suggested_type() {
    let repo = Repo::new();
    repo.write(
        "src/lib.rs",
        "pub fn one() -> u32 {\n    let n = 1;\n    n\n}\n",
    );
    repo.git(&["add", "-A"]);

    let v = json(&repo.gca(&["--dry-run", "--json"]));
    assert!(v["ranked_with_subject"].is_null(), "{v}");
    assert_ne!(v["suggestions"][0]["type"], "perf");

    let v = json(&repo.gca(&["--dry-run", "--json", "-m", "make one faster"]));
    assert_eq!(v["ranked_with_subject"], "make one faster");
    assert_eq!(v["suggestions"][0]["type"], "perf");

    // --yes takes the type the diff and the subject suggest together.
    let out = repo.gca(&["-y", "-m", "make one faster"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(repo.head_message(), "perf: make one faster");
}

fn git_with(repo: &Repo, env: &[(&str, &str)], args: &[&str]) -> Output {
    let mut cmd = Command::new("git");
    repo.env(&mut cmd);
    cmd.envs(env.iter().copied());
    let out = cmd.args(args).output().unwrap();
    assert!(out.status.success(), "git {args:?}: {}", stderr(&out));
    out
}

/// A message's type, if it starts with one of the eleven.
fn type_of(message: &str) -> Option<&str> {
    let (head, _) = message.split_once(": ")?;
    let kind = head.split('(').next()?.trim_end_matches('!');
    [
        "feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore",
        "revert",
    ]
    .contains(&kind)
    .then_some(kind)
}

#[test]
fn the_hook_types_plain_git_commits() {
    let repo = Repo::new();
    let out = repo.gca(&["hook", "install"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let hook = repo.path().join(".git/hooks/prepare-commit-msg");
    assert!(fs::read_to_string(&hook)
        .unwrap()
        .contains("gca hook install"));

    // git commit -a -m: the subject gets the type gca -y would pick
    repo.write(
        "src/lib.rs",
        "pub fn one() -> u32 {\n    let n = 1;\n    n\n}\n",
    );
    git_with(&repo, &[], &["commit", "-q", "-a", "-m", "make one faster"]);
    assert_eq!(repo.head_message(), "perf: make one faster");

    // a type the author wrote stays, and GCA_HOOK=0 skips the hook
    repo.write("README.md", "# demo\n\nmore\n");
    git_with(
        &repo,
        &[],
        &["commit", "-q", "-a", "-m", "docs: explain more"],
    );
    assert_eq!(repo.head_message(), "docs: explain more");
    repo.write("README.md", "# demo\n\neven more\n");
    git_with(
        &repo,
        &[("GCA_HOOK", "0")],
        &["commit", "-q", "-a", "-m", "no type"],
    );
    assert_eq!(repo.head_message(), "no type");

    // in the editor the first line starts with the type; the ranking is a comment
    let append =
        r#"f() { printf '%sreturn two\n' "$(head -n 1 "$1")" > "$1.tmp" && mv "$1.tmp" "$1"; }; f"#;
    repo.write("src/lib.rs", "pub fn one() -> u32 {\n    2\n}\n");
    git_with(&repo, &[("GIT_EDITOR", append)], &["commit", "-q", "-a"]);
    let message = repo.head_message();
    assert!(type_of(&message).is_some(), "{message}");
    assert!(message.ends_with(": return two"), "{message}");

    // a subject draft is filled in whole
    repo.write(
        "package.json",
        "{\n  \"dependencies\": {\n    \"zod\": \"^3.22.0\"\n  }\n}\n",
    );
    git_with(&repo, &[("GCA_HOOK", "0")], &["add", "-A"]);
    git_with(
        &repo,
        &[("GCA_HOOK", "0")],
        &["commit", "-q", "-m", "chore: add zod"],
    );
    repo.write(
        "package.json",
        "{\n  \"dependencies\": {\n    \"zod\": \"^3.23.8\"\n  }\n}\n",
    );
    git_with(&repo, &[("GIT_EDITOR", "true")], &["commit", "-q", "-a"]);
    let message = repo.head_message();
    assert!(type_of(&message).is_some(), "{message}");
    assert!(
        message.ends_with(": bump zod from 3.22.0 to 3.23.8"),
        "{message}"
    );
    assert!(!message.contains("gca suggests"), "{message}");

    // gca's own commits are left as they are
    repo.write("src/lib.rs", "pub fn one() -> u32 {\n    3\n}\n");
    let out = repo.gca(&["-a", "-t", "fix", "-m", "return three"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(repo.head_message(), "fix: return three");

    let out = repo.gca(&["hook", "uninstall"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(!hook.exists());
}

#[test]
fn the_hook_leaves_other_hooks_alone() {
    let repo = Repo::new();
    let hook = repo.path().join(".git/hooks/prepare-commit-msg");
    fs::write(&hook, "#!/bin/sh\necho mine\n").unwrap();

    let out = repo.gca(&["hook", "install"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("gca hook run"), "{}", stderr(&out));
    let out = repo.gca(&["hook", "uninstall"]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(fs::read_to_string(&hook).unwrap(), "#!/bin/sh\necho mine\n");
}

#[test]
fn the_hook_goes_where_core_hooks_path_points() {
    let repo = Repo::new();
    repo.git(&["config", "core.hooksPath", ".githooks"]);
    fs::create_dir_all(repo.path().join("src/deep")).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_gca"));
    repo.env(&mut cmd);
    let out = cmd
        .current_dir(repo.path().join("src/deep"))
        .args(["hook", "install"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(repo.path().join(".githooks/prepare-commit-msg").exists());
    assert!(!repo.path().join(".git/hooks/prepare-commit-msg").exists());
}

#[test]
fn the_project_s_own_types_tilt_the_ranking() {
    let perf_probability = |kind: &str| {
        let repo = Repo::new();
        for i in 0..30 {
            repo.write("notes.txt", &format!("{i}\n"));
            repo.git(&["add", "-A"]);
            repo.git(&["commit", "-q", "-m", &format!("{kind}: step {i}")]);
        }
        repo.write(
            "src/lib.rs",
            "pub fn one() -> u32 {\n    let n = 1;\n    n\n}\n",
        );
        repo.git(&["add", "-A"]);
        let v = json(&repo.gca(&["--dry-run", "--json", "--topk", "11"]));
        // the thirty commits and the initial one
        assert_eq!(v["ranked_with_history"], 31);
        v["suggestions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["type"] == "perf")
            .unwrap()["probability"]
            .as_f64()
            .unwrap()
    };
    let (perf_project, fix_project) = (perf_probability("perf"), perf_probability("fix"));
    assert!(
        perf_project > 1.3 * fix_project,
        "{perf_project} vs {fix_project}"
    );
}

#[test]
fn a_commitlint_config_sets_the_types_and_the_header_limit() {
    let repo = Repo::new();
    repo.write(
        ".commitlintrc.json",
        r#"{"rules": {"type-enum": [2, "always", ["feat", "fix", "deps"]],
                      "header-max-length": [2, "always", 40]}}"#,
    );
    repo.git(&["add", "-A"]);
    repo.git(&["commit", "-q", "-m", "chore: add commitlint config"]);
    repo.write(
        "src/lib.rs",
        "pub fn one() -> u32 {\n    let n = 1;\n    n\n}\n",
    );
    repo.git(&["add", "-A"]);

    let v = json(&repo.gca(&["--dry-run", "--json", "--topk", "11"]));
    let types: Vec<&str> = v["suggestions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["type"].as_str().unwrap())
        .collect();
    assert!(
        !types.is_empty() && types.iter().all(|t| ["feat", "fix"].contains(t)),
        "{types:?}"
    );
    assert_eq!(
        v["commitlint"]["types"],
        serde_json::json!(["feat", "fix", "deps"])
    );
    assert_eq!(v["commitlint"]["header_max_length"], 40);

    // a type the project left out is a usage error; one it added is fine
    let out = repo.gca(&["-t", "chore", "-m", "tidy"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("feat, fix, deps"), "{}", stderr(&out));
    let out = repo.gca(&[
        "-t",
        "deps",
        "-m",
        "bump everything that can be bumped today",
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("within 40"), "{}", stderr(&out));
    let out = repo.gca(&["-t", "deps", "-m", "bump zod"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(repo.head_message(), "deps: bump zod");
}

#[test]
fn ordinary_changes_get_no_subject_draft() {
    let repo = Repo::new();
    repo.write("src/lib.rs", "pub fn one() -> u32 {\n    2\n}\n");
    repo.git(&["add", "-A"]);

    let v = json(&repo.gca(&["--dry-run", "--json"]));
    assert!(v["subject_draft"].is_null(), "{v}");
    let text = stdout(&repo.gca(&["--dry-run"]));
    assert!(!text.contains("subject:"), "{text}");
}

#[test]
fn diff_settings_do_not_change_the_prediction() {
    let repo = Repo::new();
    repo.write(
        "src/parser.rs",
        "pub fn parse(s: &str) -> Vec<&str> {\n    s.split(',').collect()\n}\n",
    );
    repo.git(&["add", "-A"]);
    let baseline = json(&repo.gca(&["--dry-run", "--json"]))["suggestions"].clone();

    for (key, value) in [
        ("diff.noprefix", "true"),
        ("diff.mnemonicPrefix", "true"),
        ("diff.relative", "true"),
        ("color.ui", "always"),
        ("diff.context", "10"),
    ] {
        repo.git(&["config", key, value]);
        let got = json(&repo.gca(&["--dry-run", "--json"]))["suggestions"].clone();
        assert_eq!(got, baseline, "prediction changed with {key}={value}");
        repo.git(&["config", "--unset", key]);
    }
}

#[test]
fn scope_is_suggested_from_history() {
    let repo = Repo::new();
    for (file, subject) in [
        ("src/cli.rs", "feat(cli): add flags"),
        ("src/cli.rs", "fix(cli): quote paths"),
        ("src/model.rs", "feat(model): load json"),
        ("README.md", "docs: usage"),
        ("src/cli.rs", "refactor(cli): split parser"),
    ] {
        repo.write(file, &format!("// {subject}\n"));
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-q", "-m", subject]);
    }
    repo.write("src/cli.rs", "// next\n");
    repo.git(&["add", "-A"]);

    let v = json(&repo.gca(&["--dry-run", "--json"]));

    assert_eq!(v["repo_uses_scopes"], true);
    assert_eq!(v["scope"], "cli");

    let out = repo.gca(&["-y", "-m", "tidy"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        repo.head_message().contains("(cli): tidy"),
        "{}",
        repo.head_message()
    );
}

#[test]
fn without_a_terminal_it_names_the_missing_scope() {
    let repo = Repo::new();
    for i in 0..5 {
        repo.write("src/cli.rs", &format!("// {i}\n"));
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-q", "-m", &format!("fix(cli): bug {i}")]);
    }
    repo.write("src/cli.rs", "// next\n");
    repo.git(&["add", "-A"]);

    let out = repo.gca(&["-t", "fix", "-m", "tidy"]);
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(err.contains("--scope") && !err.contains("--type,"), "{err}");
    assert_eq!(repo.commit_count(), 6);

    let out = repo.gca(&["-t", "fix", "--scope", "", "-m", "tidy"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(repo.head_message(), "fix: tidy");
}

#[test]
fn no_scope_when_the_project_does_not_use_them() {
    let repo = Repo::new();
    for i in 0..6 {
        repo.write("src/lib.rs", &format!("// {i}\n"));
        repo.git(&["add", "-A"]);
        repo.git(&["commit", "-q", "-m", &format!("fix: bug {i}")]);
    }
    repo.write("src/lib.rs", "// next\n");
    repo.git(&["add", "-A"]);

    let v = json(&repo.gca(&["--dry-run", "--json"]));

    assert_eq!(v["repo_uses_scopes"], false);
    assert!(v["scope"].is_null());
}

#[test]
fn refuses_to_run_during_a_merge() {
    let repo = Repo::new();
    let head = repo.git(&["rev-parse", "HEAD"]);
    fs::write(repo.path().join(".git/MERGE_HEAD"), head).unwrap();
    repo.write("src/lib.rs", "// changed\n");
    repo.git(&["add", "-A"]);

    let out = repo.gca(&["-t", "fix", "-m", "x"]);

    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("merge is in progress"),
        "{}",
        stderr(&out)
    );
    assert_eq!(repo.commit_count(), 1);
}

fn with_remote(repo: &Repo) -> PathBuf {
    let remote = repo.path().join("remote.git");
    repo.git(&["init", "-q", "--bare", remote.to_str().unwrap()]);
    repo.git(&["remote", "add", "origin", remote.to_str().unwrap()]);
    repo.write(".gitignore", "remote.git/\n");
    repo.git(&["add", ".gitignore"]);
    repo.git(&["commit", "-q", "-m", "chore: ignore remote"]);
    remote
}

fn remote_head(repo: &Repo, remote: &Path) -> Option<String> {
    let mut cmd = Command::new("git");
    repo.env(&mut cmd);
    let out = cmd
        .args([
            "--git-dir",
            remote.to_str().unwrap(),
            "rev-parse",
            "--verify",
            "-q",
            "refs/heads/main",
        ])
        .output()
        .unwrap();
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[test]
fn does_not_push_unless_asked() {
    let repo = Repo::new();
    let remote = with_remote(&repo);
    repo.write("src/lib.rs", "// changed\n");
    repo.git(&["add", "-A"]);

    let out = repo.gca(&["-t", "fix", "-m", "local only"]);

    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(remote_head(&repo, &remote), None);
}

#[test]
fn push_sets_the_upstream_when_missing() {
    let repo = Repo::new();
    let remote = with_remote(&repo);
    repo.write("src/lib.rs", "// changed\n");
    repo.git(&["add", "-A"]);

    let out = repo.gca(&["-t", "fix", "-m", "ship it", "--push"]);

    assert!(out.status.success(), "{}", stderr(&out));
    let head = repo.git(&["rev-parse", "HEAD"]).trim().to_string();
    assert_eq!(remote_head(&repo, &remote), Some(head));
    assert_eq!(repo.git(&["config", "branch.main.remote"]).trim(), "origin");
}

#[test]
fn push_setting_is_read_from_git_config() {
    let repo = Repo::new();
    let remote = with_remote(&repo);
    let out = repo.gca(&["config", "push", "auto", "--local"]);
    assert!(out.status.success(), "{}", stderr(&out));
    repo.write("src/lib.rs", "// changed\n");
    repo.git(&["add", "-A"]);

    let out = repo.gca(&["-t", "fix", "-m", "auto push"]);

    assert!(out.status.success(), "{}", stderr(&out));
    assert!(remote_head(&repo, &remote).is_some());
}

#[test]
fn order_setting_is_stored_and_checked() {
    let repo = Repo::new();
    let out = repo.gca(&["config", "order"]);
    assert!(stdout(&out).contains("gca.order = type-first (default)"));
    let out = repo.gca(&["config", "order", "subject-first", "--local"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(repo.git(&["config", "gca.order"]).trim(), "subject-first");
    let out = repo.gca(&["config", "order", "sideways"]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "only the two orders are accepted"
    );

    // Flags that answer every prompt make the order moot.
    repo.write("src/lib.rs", "// changed\n");
    repo.git(&["add", "-A"]);
    let out = repo.gca(&["-t", "fix", "-m", "with flags"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(repo.head_message(), "fix: with flags");

    // A bad value stops gca before anything is staged or committed.
    repo.git(&["config", "gca.order", "sideways"]);
    repo.write("src/lib.rs", "// changed again\n");
    let out = repo.gca(&["-a", "-t", "fix", "-m", "bad order"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("gca.order"), "{}", stderr(&out));
    assert_eq!(repo.commit_count(), 2);
    assert_eq!(repo.staged(), "");
}

#[cfg(unix)]
#[test]
fn hooks_run_and_no_verify_skips_them() {
    use std::os::unix::fs::PermissionsExt;
    let repo = Repo::new();
    let hook = repo.path().join(".git/hooks/commit-msg");
    fs::write(&hook, "#!/bin/sh\necho 'rejected by hook' >&2\nexit 1\n").unwrap();
    fs::set_permissions(&hook, fs::Permissions::from_mode(0o755)).unwrap();
    repo.write("src/lib.rs", "// changed\n");
    repo.git(&["add", "-A"]);

    let out = repo.gca(&["-t", "fix", "-m", "blocked"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("rejected by hook"),
        "{}",
        stderr(&out)
    );
    assert_eq!(repo.commit_count(), 1);
    assert!(
        !repo.staged().is_empty(),
        "changes stay staged after a failed commit"
    );

    let out = repo.gca(&["-t", "fix", "-m", "skip hooks", "-n"]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(repo.commit_count(), 2);
}

#[test]
fn completions_and_version() {
    let repo = Repo::new();
    let out = repo.gca(&["completions", "bash"]);
    assert!(out.status.success());
    assert!(stdout(&out).contains("gca"));
    let out = repo.gca(&["--version"]);
    assert_eq!(
        stdout(&out).trim(),
        format!("gca {}", env!("CARGO_PKG_VERSION"))
    );
}

// ---------------------------------------------------------------------------
// Edge cases around previews and the index, carried over from the earlier
// CLI tests: unborn repositories, subdirectories, split and custom indexes,
// linked worktrees, and settings that must fail before anything is staged.

fn run_git(repo: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", repo.join("test-global-config"))
        .env_remove("GIT_INDEX_FILE")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn unborn_repo() -> TempDir {
    let dir = TempDir::new().unwrap();
    run_git(dir.path(), &["init", "--quiet"]);
    run_git(dir.path(), &["config", "user.name", "CLI tests"]);
    run_git(
        dir.path(),
        &["config", "user.email", "tests@example.invalid"],
    );
    run_git(dir.path(), &["config", "core.autocrlf", "false"]);
    dir
}

fn run_gca(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gca"))
        .args(args)
        .current_dir(dir)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", dir.join("test-global-config"))
        .env("NO_COLOR", "1")
        .env_remove("GIT_INDEX_FILE")
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn preview(out: Output) -> serde_json::Value {
    assert!(out.status.success(), "{}", stderr(&out));
    json(&out)
}

#[test]
fn dry_run_on_unborn_repo_leaves_index_absent() {
    let repo = unborn_repo();
    fs::write(repo.path().join("README.md"), "hello\n").unwrap();
    let v = preview(run_gca(repo.path(), &["--dry-run", "--json", "README.md"]));
    assert_eq!(v["preselected"], "docs");
    assert_eq!(v["additions"], 1);
    assert!(!repo.path().join(".git/index").exists());
}

#[test]
fn dry_run_uses_staged_bytes_and_leaves_the_index_alone() {
    let repo = unborn_repo();
    fs::write(repo.path().join("README.md"), "staged\n").unwrap();
    run_git(repo.path(), &["add", "README.md"]);
    fs::write(repo.path().join("README.md"), "staged\nunstaged\n").unwrap();
    fs::write(repo.path().join("code.rs"), "fn main() {}\n").unwrap();
    let index = fs::read(repo.path().join(".git/index")).unwrap();
    let v = preview(run_gca(repo.path(), &["--dry-run", "--json"]));
    assert_eq!(v["preselected"], "docs");
    assert_eq!(v["additions"], 1);
    assert_eq!(v["files"].as_array().unwrap().len(), 1);
    assert_eq!(fs::read(repo.path().join(".git/index")).unwrap(), index);
}

#[test]
fn dry_run_with_paths_previews_only_those_paths() {
    let repo = unborn_repo();
    fs::write(repo.path().join("code.rs"), "fn main() {}\n").unwrap();
    run_git(repo.path(), &["add", "code.rs"]);
    fs::write(repo.path().join("README.md"), "one\ntwo\n").unwrap();
    let index = fs::read(repo.path().join(".git/index")).unwrap();
    let v = preview(run_gca(repo.path(), &["--dry-run", "--json", "README.md"]));
    assert_eq!(v["preselected"], "docs");
    assert_eq!(v["additions"], 2);
    assert_eq!(v["files"][0]["path"], "README.md");
    assert_eq!(fs::read(repo.path().join(".git/index")).unwrap(), index);
}

#[test]
fn preview_from_a_subdirectory_ignores_diff_display_settings() {
    let repo = unborn_repo();
    fs::create_dir(repo.path().join("docs")).unwrap();
    fs::write(repo.path().join("docs/guide.txt"), "guide\n").unwrap();
    run_git(repo.path(), &["config", "diff.relative", "true"]);
    run_git(repo.path(), &["config", "diff.noprefix", "true"]);
    let v = preview(run_gca(
        &repo.path().join("docs"),
        &["--dry-run", "--json", "."],
    ));
    assert_eq!(v["preselected"], "docs");
    assert_eq!(v["files"][0]["path"], "docs/guide.txt");
    assert!(!repo.path().join(".git/index").exists());
}

#[test]
fn preview_handles_a_split_index_without_changing_it() {
    let repo = unborn_repo();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    run_git(repo.path(), &["add", "README.md"]);
    run_git(repo.path(), &["update-index", "--split-index"]);
    fs::write(repo.path().join("README.md"), "guide\nmore\n").unwrap();
    let index = fs::read(repo.path().join(".git/index")).unwrap();
    let v = preview(run_gca(repo.path(), &["--dry-run", "--json", "-a"]));
    assert_eq!(v["preselected"], "docs");
    assert_eq!(fs::read(repo.path().join(".git/index")).unwrap(), index);
}

#[test]
fn invalid_model_fails_before_staging() {
    let repo = unborn_repo();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    let out = run_gca(
        repo.path(),
        &[
            "--model",
            "missing.json",
            "README.md",
            "-t",
            "docs",
            "-m",
            "x",
        ],
    );
    assert!(!out.status.success());
    assert!(stderr(&out).contains("missing.json"), "{}", stderr(&out));
    assert!(!repo.path().join(".git/index").exists());
}

#[test]
fn invalid_push_setting_fails_before_staging() {
    let repo = unborn_repo();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    run_git(repo.path(), &["config", "gca.push", "typo"]);
    let out = run_gca(repo.path(), &["README.md", "-t", "docs", "-m", "x"]);
    assert!(!out.status.success());
    assert!(
        stderr(&out).contains("invalid gca.push"),
        "{}",
        stderr(&out)
    );
    assert!(!repo.path().join(".git/index").exists());
}

#[test]
fn empty_preview_exits_without_interaction() {
    let repo = unborn_repo();
    let out = run_gca(repo.path(), &["--dry-run"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("nothing to commit"),
        "{}",
        stderr(&out)
    );
    assert!(!repo.path().join(".git/index").exists());
}

#[test]
fn preview_respects_a_custom_index() {
    let repo = unborn_repo();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    run_git(repo.path(), &["add", "README.md"]);
    let index = fs::read(repo.path().join(".git/index")).unwrap();
    let alternate = repo.path().join(".git/alternate-index");
    fs::write(&alternate, &index).unwrap();
    fs::write(repo.path().join("README.md"), "new\ncontents\n").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_gca"))
        .args(["--dry-run", "--json", "README.md"])
        .current_dir(repo.path())
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", repo.path().join("test-global-config"))
        .env("GIT_INDEX_FILE", &alternate)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(preview(out)["additions"], 2);
    assert_eq!(fs::read(alternate).unwrap(), index);
    assert_eq!(fs::read(repo.path().join(".git/index")).unwrap(), index);
}

#[test]
fn preview_works_in_a_linked_worktree() {
    let repo = unborn_repo();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    run_git(repo.path(), &["add", "README.md"]);
    run_git(
        repo.path(),
        &[
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
    );
    let worktree_dir = TempDir::new().unwrap();
    let worktree = worktree_dir.path().join("checkout");
    run_git(
        repo.path(),
        &["worktree", "add", "--detach", worktree.to_str().unwrap()],
    );
    fs::write(worktree.join("README.md"), "guide\nadded\n").unwrap();
    let index_path = run_git(&worktree, &["rev-parse", "--git-path", "index"]);
    let index_path = worktree.join(index_path.trim());
    let index = fs::read(&index_path).unwrap();
    let v = preview(run_gca(&worktree, &["--dry-run", "--json", "-a"]));
    assert_eq!(v["additions"], 1);
    assert_eq!(fs::read(&index_path).unwrap(), index);
    assert!(run_git(&worktree, &["diff", "--cached", "--name-only"]).is_empty());
}

#[test]
fn path_rule_only_preselects_types_the_model_has() {
    let repo = unborn_repo();
    let fixture_dir = TempDir::new().unwrap();
    let model_path = fixture_dir.path().join("binary.json");
    let vectorizer = serde_json::json!({
        "vocabulary": {}, "token_pattern": "\\w+", "lowercase": true, "binary": true
    });
    let model = serde_json::json!({
        "schema_version": 1, "classes": ["feat", "fix"],
        "tfidf": {"vocabulary": {}, "idf": [], "token_pattern": "\\w+", "lowercase": true,
                  "norm": "l2", "sublinear_tf": false},
        "path_bow": vectorizer, "ext_bow": vectorizer,
        "scaler": {"mean": vec![0.0; 4], "scale": vec![1.0; 4]},
        "calibrated_folds": [{"coef": vec![vec![0.0; 5]], "intercept": [0.0],
                              "sigmoid_a": [1.0], "sigmoid_b": [0.0]}]
    });
    fs::write(&model_path, serde_json::to_vec(&model).unwrap()).unwrap();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    let v = preview(run_gca(
        repo.path(),
        &[
            "--dry-run",
            "--json",
            "--model",
            model_path.to_str().unwrap(),
            "README.md",
        ],
    ));
    assert_eq!(v["preselected"], "feat");
    let types: Vec<&str> = v["suggestions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["type"].as_str().unwrap())
        .collect();
    assert!(!types.contains(&"docs"), "{types:?}");
}
