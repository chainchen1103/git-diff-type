// Path-based pre-classifier. Fires only when every staged file matches the
// same category, so mixed commits fall through to the ML model.
use once_cell::sync::Lazy;
use regex::Regex;

pub struct Hit {
    pub label: &'static str,
    pub reason: String,
}

static DOC: Lazy<Vec<Regex>> = Lazy::new(|| {
    [
        r"(?i).*\.(md|mdx|rst|adoc)$",
        r"(?i)(^|/)(README|CHANGELOG|CONTRIBUTING|AUTHORS|LICENSE|NOTICE|CODE_OF_CONDUCT|SECURITY|MAINTAINERS)(\.[^/]*)?$",
        r"(?i)(^|/)docs?/",
        r"(?i)(^|/)documentation/",
    ]
    .iter()
    .map(|p| Regex::new(p).unwrap())
    .collect()
});

static TEST: Lazy<Vec<Regex>> = Lazy::new(|| {
    [
        r"(^|/)tests?/",
        r"(^|/)__tests__/",
        r"(^|/)spec/",
        r"(^|/)e2e/",
        r"(^|/)test_[^/]+\.py$",
        r".*_test\.(py|go|rb)$",
        r".*\.(test|spec)\.(ts|tsx|js|jsx|mjs|cjs)$",
        r".*Test\.java$",
        r".*Tests?\.cs$",
        r".*_spec\.rb$",
    ]
    .iter()
    .map(|p| Regex::new(p).unwrap())
    .collect()
});

static CI: Lazy<Vec<Regex>> = Lazy::new(|| {
    [
        r"^\.github/(workflows|actions)/",
        r"^\.gitlab-ci\.ya?ml$",
        r"^\.circleci/",
        r"^azure-pipelines.*\.ya?ml$",
        r"^Jenkinsfile$",
        r"^\.travis\.ya?ml$",
        r"^\.drone\.ya?ml$",
        r"^\.buildkite/",
        r"^buildkite\.ya?ml$",
        r"^appveyor\.ya?ml$",
        r"^codecov\.ya?ml$",
        r"^\.pre-commit-config\.ya?ml$",
    ]
    .iter()
    .map(|p| Regex::new(p).unwrap())
    .collect()
});

fn all_match(files: &[String], patterns: &[Regex]) -> bool {
    !files.is_empty() && files.iter().all(|f| patterns.iter().any(|p| p.is_match(f)))
}

pub fn classify(files: &[String]) -> Option<Hit> {
    if files.is_empty() {
        return None;
    }
    if all_match(files, &CI) {
        return Some(Hit { label: "ci", reason: format!("all {} file(s) under CI config paths", files.len()) });
    }
    if all_match(files, &DOC) {
        return Some(Hit { label: "docs", reason: format!("all {} file(s) are documentation", files.len()) });
    }
    if all_match(files, &TEST) {
        return Some(Hit { label: "test", reason: format!("all {} file(s) are test files", files.len()) });
    }
    None
}
