use std::fs;
use std::path::Path;
use std::process::{Command, Output, Stdio};

fn git(repo: &Path, args: &[&str]) -> String {
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

fn repo() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    git(dir.path(), &["init", "--quiet"]);
    git(dir.path(), &["config", "user.name", "CLI tests"]);
    git(
        dir.path(),
        &["config", "user.email", "tests@example.invalid"],
    );
    git(dir.path(), &["config", "core.autocrlf", "false"]);
    dir
}

fn gca(repo: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_gca"))
        .args(args)
        .current_dir(repo)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", repo.join("test-global-config"))
        .env_remove("GIT_INDEX_FILE")
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn success(output: Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn dry_run_on_unborn_repo_leaves_index_absent() {
    let repo = repo();
    fs::write(repo.path().join("README.md"), "hello\n").unwrap();
    let output = success(gca(repo.path(), &["--dry-run"]));
    assert!(output.contains("Suggested type: docs"));
    assert!(output.contains("+1 / -0 lines in 1 files"));
    assert!(!repo.path().join(".git/index").exists());
    assert!(git(repo.path(), &["diff", "--cached", "--name-only"]).is_empty());
}

#[test]
fn dry_run_preserves_staged_bytes_and_ignores_unstaged_files() {
    let repo = repo();
    fs::write(repo.path().join("README.md"), "staged\n").unwrap();
    git(repo.path(), &["add", "README.md"]);
    fs::write(repo.path().join("README.md"), "staged\nunstaged\n").unwrap();
    fs::write(repo.path().join("code.rs"), "fn main() {}\n").unwrap();
    let index = fs::read(repo.path().join(".git/index")).unwrap();
    let output = success(gca(repo.path(), &["--dry-run"]));
    assert!(output.contains("Suggested type: docs"));
    assert!(output.contains("+1 / -0 lines in 1 files"));
    assert_eq!(fs::read(repo.path().join(".git/index")).unwrap(), index);
}

#[test]
fn dry_run_with_paths_previews_only_requested_changes() {
    let repo = repo();
    fs::write(repo.path().join("code.rs"), "fn main() {}\n").unwrap();
    git(repo.path(), &["add", "code.rs"]);
    fs::write(repo.path().join("README.md"), "one\ntwo\n").unwrap();
    let index = fs::read(repo.path().join(".git/index")).unwrap();
    let output = success(gca(repo.path(), &["--dry-run", "README.md"]));
    assert!(output.contains("Suggested type: docs"));
    assert!(output.contains("+2 / -0 lines in 1 files"));
    assert_eq!(fs::read(repo.path().join(".git/index")).unwrap(), index);
}

#[test]
fn preview_from_subdirectory_ignores_diff_display_settings() {
    let repo = repo();
    fs::create_dir(repo.path().join("docs")).unwrap();
    fs::write(repo.path().join("docs/guide.txt"), "guide\n").unwrap();
    git(repo.path(), &["config", "diff.relative", "true"]);
    git(repo.path(), &["config", "diff.noprefix", "true"]);
    let output = success(gca(&repo.path().join("docs"), &["--dry-run", "."]));
    assert!(output.contains("Suggested type: docs"));
    assert!(output.contains("+1 / -0 lines in 1 files"));
    assert!(!repo.path().join(".git/index").exists());
}

#[test]
fn preview_handles_split_index_without_changing_it() {
    let repo = repo();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    git(repo.path(), &["add", "README.md"]);
    git(repo.path(), &["update-index", "--split-index"]);
    let index = fs::read(repo.path().join(".git/index")).unwrap();
    let output = success(gca(repo.path(), &["--dry-run"]));
    assert!(output.contains("Suggested type: docs"));
    assert_eq!(fs::read(repo.path().join(".git/index")).unwrap(), index);
}

#[test]
fn invalid_model_fails_before_staging() {
    let repo = repo();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    let output = gca(repo.path(), &["--model", "missing.json", "README.md"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("failed to load model"));
    assert!(!repo.path().join(".git/index").exists());
}

#[test]
fn invalid_push_setting_fails_before_staging() {
    let repo = repo();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    git(repo.path(), &["config", "gca.push", "typo"]);
    let output = gca(repo.path(), &["README.md"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid gca.push"));
    assert!(!repo.path().join(".git/index").exists());
}

#[test]
fn empty_preview_exits_without_interaction() {
    let repo = repo();
    let output = success(gca(repo.path(), &["--dry-run"]));
    assert!(output.contains("nothing to commit"));
    assert!(!repo.path().join(".git/index").exists());
}

#[test]
fn list_leaves_index_unchanged() {
    let repo = repo();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    let output = success(gca(repo.path(), &["list"]));
    assert!(output.contains("README.md"));
    assert!(!repo.path().join(".git/index").exists());
}

#[test]
fn preview_respects_custom_index() {
    let repo = repo();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    git(repo.path(), &["add", "README.md"]);
    let index = fs::read(repo.path().join(".git/index")).unwrap();
    let alternate = repo.path().join(".git/alternate-index");
    fs::write(&alternate, &index).unwrap();
    fs::write(repo.path().join("README.md"), "new\ncontents\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_gca"))
        .args(["--dry-run", "README.md"])
        .current_dir(repo.path())
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", repo.path().join("test-global-config"))
        .env("GIT_INDEX_FILE", &alternate)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(success(output).contains("+2 / -0 lines in 1 files"));
    assert_eq!(fs::read(alternate).unwrap(), index);
    assert_eq!(fs::read(repo.path().join(".git/index")).unwrap(), index);
}

#[test]
fn preview_works_in_linked_worktree() {
    let repo = repo();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    git(repo.path(), &["add", "README.md"]);
    git(
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
    let worktree_dir = tempfile::tempdir().unwrap();
    let worktree = worktree_dir.path().join("checkout");
    git(
        repo.path(),
        &["worktree", "add", "--detach", worktree.to_str().unwrap()],
    );
    fs::write(worktree.join("README.md"), "guide\nadded\n").unwrap();
    let index_path = git(&worktree, &["rev-parse", "--git-path", "index"]);
    let index = fs::read(index_path.trim()).unwrap();
    let output = success(gca(&worktree, &["--dry-run"]));
    assert!(output.contains("+1 / -0 lines in 1 files"));
    assert_eq!(fs::read(index_path.trim()).unwrap(), index);
    assert!(git(&worktree, &["diff", "--cached", "--name-only"]).is_empty());
}

#[test]
fn heuristic_does_not_add_classes_missing_from_external_model() {
    let repo = repo();
    let fixture_dir = tempfile::tempdir().unwrap();
    let model_path = fixture_dir.path().join("binary.json");
    let vectorizer = serde_json::json!({
        "vocabulary": {}, "token_pattern": "\\w+", "lowercase": true, "binary": true
    });
    let model = serde_json::json!({
        "schema_version": 1, "classes": ["feat", "fix"],
        "tfidf": {"vocabulary": {}, "idf": [], "token_pattern": "\\w+", "lowercase": true, "norm": "l2", "sublinear_tf": false},
        "path_bow": vectorizer, "ext_bow": vectorizer,
        "scaler": {"mean": vec![0.0; 4], "scale": vec![1.0; 4]},
        "calibrated_folds": [{"coef": vec![vec![0.0; 5]], "intercept": [0.0], "sigmoid_a": [1.0], "sigmoid_b": [0.0]}]
    });
    fs::write(&model_path, serde_json::to_vec(&model).unwrap()).unwrap();
    fs::write(repo.path().join("README.md"), "guide\n").unwrap();
    let output = success(gca(
        repo.path(),
        &["--dry-run", "--model", model_path.to_str().unwrap()],
    ));
    assert!(output.contains("Suggested type: feat"));
    assert!(!output.contains("docs"));
}
