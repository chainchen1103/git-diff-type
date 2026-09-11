// Path-based pre-classifier. Fires only when every staged file matches the
// same category, so mixed commits fall through to the ML model.
use regex::RegexSet;
use std::sync::LazyLock;

static DOC: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        r"(?i).*\.(md|mdx|rst|adoc)$",
        r"(?i)(^|/)(README|CHANGELOG|CONTRIBUTING|AUTHORS|LICENSE|NOTICE|CODE_OF_CONDUCT|SECURITY|MAINTAINERS)(\.[^/]*)?$",
        r"(?i)(^|/)docs?/",
        r"(?i)(^|/)documentation/",
    ])
    .unwrap()
});

static TEST: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
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
    ])
    .unwrap()
});

static CI: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
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
    ])
    .unwrap()
});

fn all_match(files: &[String], patterns: &RegexSet) -> bool {
    !files.is_empty() && files.iter().all(|f| patterns.is_match(f))
}

pub fn classify(files: &[String]) -> Option<&'static str> {
    if all_match(files, &CI) {
        Some("ci")
    } else if all_match(files, &DOC) {
        Some("docs")
    } else if all_match(files, &TEST) {
        Some("test")
    } else {
        None
    }
}
