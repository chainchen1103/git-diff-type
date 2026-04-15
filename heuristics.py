"""Path-based commit classifier. Fires only when every staged file fits one
category, so mixed commits fall through to the ML model."""
import re
from typing import List, Optional, Tuple

DOC_PATTERNS = [
    re.compile(r".*\.(md|mdx|rst|adoc)$", re.I),
    re.compile(
        r"(^|/)(README|CHANGELOG|CONTRIBUTING|AUTHORS|LICENSE|NOTICE|"
        r"CODE_OF_CONDUCT|SECURITY|MAINTAINERS)(\.[^/]*)?$",
        re.I,
    ),
    re.compile(r"(^|/)docs?/", re.I),
    re.compile(r"(^|/)documentation/", re.I),
]

TEST_PATTERNS = [
    re.compile(r"(^|/)tests?/"),
    re.compile(r"(^|/)__tests__/"),
    re.compile(r"(^|/)spec/"),
    re.compile(r"(^|/)e2e/"),
    re.compile(r"(^|/)test_[^/]+\.py$"),
    re.compile(r".*_test\.(py|go|rb)$"),
    re.compile(r".*\.(test|spec)\.(ts|tsx|js|jsx|mjs|cjs)$"),
    re.compile(r".*Test\.java$"),
    re.compile(r".*Tests?\.cs$"),
    re.compile(r".*_spec\.rb$"),
]

CI_PATTERNS = [
    re.compile(r"^\.github/(workflows|actions)/"),
    re.compile(r"^\.gitlab-ci\.ya?ml$"),
    re.compile(r"^\.circleci/"),
    re.compile(r"^azure-pipelines.*\.ya?ml$"),
    re.compile(r"^Jenkinsfile$"),
    re.compile(r"^\.travis\.ya?ml$"),
    re.compile(r"^\.drone\.ya?ml$"),
    re.compile(r"^\.buildkite/"),
    re.compile(r"^buildkite\.ya?ml$"),
    re.compile(r"^appveyor\.ya?ml$"),
    re.compile(r"^codecov\.ya?ml$"),
    re.compile(r"^\.pre-commit-config\.ya?ml$"),
]


def _all_match(files: List[str], patterns: List[re.Pattern]) -> bool:
    if not files:
        return False
    return all(any(p.search(f) for p in patterns) for f in files)


def heuristic_classify(files: List[str]) -> Optional[Tuple[str, str]]:
    """Return (label, reason) when all files match one high-precision rule."""
    if not files:
        return None
    if _all_match(files, CI_PATTERNS):
        return ("ci", f"all {len(files)} file(s) under CI config paths")
    if _all_match(files, DOC_PATTERNS):
        return ("docs", f"all {len(files)} file(s) are documentation")
    if _all_match(files, TEST_PATTERNS):
        return ("test", f"all {len(files)} file(s) are test files")
    return None
