//! Conventional Commit headers: the allowed types, building and checking a
//! header, and reading the type and scope back out of existing subjects.

use regex::Regex;
use std::sync::LazyLock;

/// The eleven types the model knows, in the order commit tools usually list them.
pub const TYPES: [&str; 11] = [
    "feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert",
];

/// commitlint's default `header-max-length`; a project's config can change it.
pub const MAX_HEADER_LEN: usize = 100;

static SUBJECT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([a-z]+)(?:\(([^()\r\n]+)\))?!?: \S").unwrap());

/// `type(scope)!: subject`, leaving out the parts that are not set.
pub fn header(kind: &str, scope: Option<&str>, breaking: bool, subject: &str) -> String {
    let mut h = String::from(kind);
    if let Some(scope) = scope {
        h.push('(');
        h.push_str(scope);
        h.push(')');
    }
    if breaking {
        h.push('!');
    }
    h.push_str(": ");
    h.push_str(subject.trim());
    h
}

/// Trimmed scope, or `None` for an empty one.
pub fn normalize_scope(input: &str) -> Result<Option<String>, String> {
    let s = input.trim();
    if s.is_empty() {
        return Ok(None);
    }
    if s.chars()
        .any(|c| c == '(' || c == ')' || c == '\n' || c == '\r')
    {
        return Err(format!(
            "scope {s:?} cannot contain parentheses or line breaks"
        ));
    }
    Ok(Some(s.to_string()))
}

pub fn check_subject(subject: &str) -> Result<(), String> {
    let s = subject.trim();
    if s.is_empty() {
        return Err("the subject cannot be empty".into());
    }
    if s.contains('\n') || s.contains('\r') {
        return Err("the subject must be one line; pass the body as another -m".into());
    }
    Ok(())
}

pub fn check_header_len(header: &str, max: usize) -> Result<(), String> {
    let n = header.chars().count();
    if n > max {
        return Err(format!(
            "the header is {n} characters; keep it within {max}"
        ));
    }
    Ok(())
}

/// Type and scope of a subject that follows Conventional Commits.
pub fn parse_subject(subject: &str) -> Option<(&str, Option<&str>)> {
    let caps = SUBJECT.captures(subject)?;
    let kind = caps.get(1)?.as_str();
    if !TYPES.contains(&kind) {
        return None;
    }
    let scope = caps
        .get(2)
        .map(|m| m.as_str().trim())
        .filter(|s| !s.is_empty());
    Some((kind, scope))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_parts() {
        assert_eq!(
            header("fix", None, false, "handle empty diff"),
            "fix: handle empty diff"
        );
        assert_eq!(
            header("feat", Some("cli"), false, " add -a "),
            "feat(cli): add -a"
        );
        assert_eq!(
            header("feat", Some("api"), true, "drop v1"),
            "feat(api)!: drop v1"
        );
        assert_eq!(
            header("refactor", None, true, "rename"),
            "refactor!: rename"
        );
    }

    #[test]
    fn scope_rules() {
        assert_eq!(normalize_scope("  "), Ok(None));
        assert_eq!(normalize_scope(" cli "), Ok(Some("cli".into())));
        assert_eq!(normalize_scope("core, cli"), Ok(Some("core, cli".into())));
        assert!(normalize_scope("a(b)").is_err());
    }

    #[test]
    fn subject_rules() {
        assert!(check_subject("add flag").is_ok());
        assert!(check_subject("   ").is_err());
        assert!(check_subject("one\ntwo").is_err());
        assert!(check_header_len(&"x".repeat(MAX_HEADER_LEN), MAX_HEADER_LEN).is_ok());
        assert!(check_header_len(&"x".repeat(MAX_HEADER_LEN + 1), MAX_HEADER_LEN).is_err());
        assert!(check_header_len(&"x".repeat(51), 50).is_err());
    }

    #[test]
    fn parses_existing_subjects() {
        assert_eq!(
            parse_subject("feat(cli): add -a"),
            Some(("feat", Some("cli")))
        );
        assert_eq!(parse_subject("fix!: breaking fix"), Some(("fix", None)));
        assert_eq!(
            parse_subject("chore(deps)!: bump"),
            Some(("chore", Some("deps")))
        );
        assert_eq!(parse_subject("Merge branch 'main'"), None);
        assert_eq!(parse_subject("feature: not a type"), None);
        assert_eq!(parse_subject("feat:missing space"), None);
    }
}
