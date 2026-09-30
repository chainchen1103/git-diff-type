//! The project's commitlint rules that shape a header: which types it allows
//! (`type-enum`) and how long a header may be (`header-max-length`). Only
//! rules at level 2 count, since commitlint rejects the commit for those.
//!
//! JSON configs are read as JSON. YAML, JavaScript and TypeScript configs
//! cannot be run here, so gca looks for the two rules written out literally,
//! as most configs write them, and otherwise keeps its defaults.

use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use crate::message;

/// Where commitlint looks for its config in each directory, in its order.
const PLACES: [&str; 17] = [
    "package.json",
    ".commitlintrc",
    ".commitlintrc.json",
    ".commitlintrc.yaml",
    ".commitlintrc.yml",
    ".commitlintrc.js",
    ".commitlintrc.cjs",
    ".commitlintrc.mjs",
    "commitlint.config.js",
    "commitlint.config.cjs",
    "commitlint.config.mjs",
    ".commitlintrc.ts",
    ".commitlintrc.cts",
    ".commitlintrc.mts",
    "commitlint.config.ts",
    "commitlint.config.cts",
    "commitlint.config.mts",
];

/// @commitlint/config-angular's types: the conventional ones without `chore`.
const ANGULAR_TYPES: [&str; 10] = [
    "build", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style", "test",
];

#[derive(Debug, Default, PartialEq)]
pub struct Rules {
    /// The config the rules came from, if the project has one.
    pub file: Option<PathBuf>,
    /// The types the project allows, when it lists them.
    pub types: Option<Vec<String>>,
    pub header_max: Option<usize>,
}

impl Rules {
    pub fn allows(&self, kind: &str) -> bool {
        match &self.types {
            Some(types) => types.iter().any(|t| t == kind),
            None => message::TYPES.contains(&kind),
        }
    }

    /// The allowed types, in the config's order or the usual one.
    pub fn type_list(&self) -> Vec<&str> {
        match &self.types {
            Some(types) => types.iter().map(String::as_str).collect(),
            None => message::TYPES.to_vec(),
        }
    }

    pub fn header_max(&self) -> usize {
        self.header_max.unwrap_or(message::MAX_HEADER_LEN)
    }
}

/// The rules of the first config found from `from` up to `top`.
pub fn load(from: &Path, top: &Path) -> Rules {
    let mut dir = Some(from);
    while let Some(d) = dir {
        for name in PLACES {
            let path = d.join(name);
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            if let Some(mut rules) = parse(name, &text) {
                rules.file = Some(path);
                return rules;
            }
        }
        if d == top {
            break;
        }
        dir = d.parent();
    }
    Rules::default()
}

/// `None` when the file is not a commitlint config (a package.json without
/// a `commitlint` key).
fn parse(name: &str, text: &str) -> Option<Rules> {
    if name == "package.json" {
        let json: serde_json::Value = serde_json::from_str(text).ok()?;
        return json.get("commitlint").map(from_json);
    }
    if name == ".commitlintrc" || name.ends_with(".json") {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
            return Some(from_json(&json));
        }
    }
    Some(from_text(text))
}

fn from_json(config: &serde_json::Value) -> Rules {
    let rule = |name: &str| {
        let r = config.get("rules")?.get(name)?.as_array()?;
        let enforced = r.first()?.as_u64()? == 2 && r.get(1)?.as_str()? == "always";
        enforced.then(|| r.get(2).cloned()).flatten()
    };
    let types = rule("type-enum").and_then(|v| {
        v.as_array().map(|items| {
            items
                .iter()
                .filter_map(|t| t.as_str())
                .filter(|t| is_type(t))
                .map(String::from)
                .collect::<Vec<_>>()
        })
    });
    let extends_angular = config
        .get("extends")
        .is_some_and(|e| e.to_string().contains("config-angular"));
    Rules {
        file: None,
        types: types
            .filter(|t| !t.is_empty())
            .or_else(|| extends_angular.then(angular)),
        header_max: rule("header-max-length")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize),
    }
}

static TYPE_ENUM: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"["']?type-enum["']?\s*:\s*\[\s*(\d)\s*,\s*["']?(always|never)["']?\s*,\s*\[([^\]]*)\]"#,
    )
    .unwrap()
});
static HEADER_MAX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"["']?header-max-length["']?\s*:\s*\[\s*(\d)\s*,\s*["']?(always|never)["']?\s*,\s*(\d+)\s*\]"#)
        .unwrap()
});
static QUOTED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"["'`]([A-Za-z0-9_-]+)["'`]"#).unwrap());

/// Rules written out literally in YAML, JavaScript or TypeScript.
fn from_text(text: &str) -> Rules {
    let enforced = |c: &regex::Captures| &c[1] == "2" && &c[2] == "always";
    let types = TYPE_ENUM
        .captures(text)
        .filter(|c| enforced(c))
        .map(|c| {
            let list = c.get(3).map_or("", |m| m.as_str());
            let quoted: Vec<String> = QUOTED
                .captures_iter(list)
                .map(|q| q[1].to_string())
                .collect();
            if quoted.is_empty() {
                // YAML flow style: [feat, fix]
                list.split(',')
                    .map(str::trim)
                    .filter(|t| is_type(t))
                    .map(String::from)
                    .collect()
            } else {
                quoted.into_iter().filter(|t| is_type(t)).collect()
            }
        })
        .filter(|t: &Vec<String>| !t.is_empty());
    Rules {
        file: None,
        types: types.or_else(|| text.contains("config-angular").then(angular)),
        header_max: HEADER_MAX
            .captures(text)
            .filter(|c| enforced(c))
            .and_then(|c| c[3].parse().ok()),
    }
}

fn angular() -> Vec<String> {
    ANGULAR_TYPES.iter().map(|t| t.to_string()).collect()
}

/// A word that can be a commit type.
pub fn is_type(word: &str) -> bool {
    let mut chars = word.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn types(r: &Rules) -> Vec<&str> {
        r.types
            .as_ref()
            .map(|t| t.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    #[test]
    fn json_configs() {
        let r = parse(
            ".commitlintrc.json",
            r#"{"extends": ["@commitlint/config-conventional"],
                "rules": {"type-enum": [2, "always", ["feat", "fix", "deps"]],
                          "header-max-length": [2, "always", 72]}}"#,
        )
        .unwrap();
        assert_eq!(types(&r), ["feat", "fix", "deps"]);
        assert_eq!(r.header_max, Some(72));
        assert!(r.allows("deps") && !r.allows("chore"));

        let pkg =
            r#"{"name": "x", "commitlint": {"rules": {"header-max-length": [2, "always", 120]}}}"#;
        let r = parse("package.json", pkg).unwrap();
        assert_eq!((r.types.clone(), r.header_max), (None, Some(120)));
        assert!(r.allows("chore"));
        assert_eq!(parse("package.json", r#"{"name": "x"}"#), None);
    }

    #[test]
    fn warnings_and_disabled_rules_do_not_count() {
        let r = parse(
            ".commitlintrc.json",
            r#"{"rules": {"type-enum": [1, "always", ["feat"]], "header-max-length": [0, "always", 50]}}"#,
        )
        .unwrap();
        assert_eq!(r, Rules::default());
        let never = parse(
            ".commitlintrc.json",
            r#"{"rules": {"type-enum": [2, "never", ["wip"]]}}"#,
        );
        assert_eq!(never.unwrap().types, None);
    }

    #[test]
    fn javascript_typescript_and_yaml_written_out() {
        let js = "module.exports = {\n  extends: ['@commitlint/config-conventional'],\n  rules: {\n    'type-enum': [\n      2,\n      'always',\n      ['feat', 'fix', // bugs\n       \"docs\", `chore`],\n    ],\n    'header-max-length': [2, 'always', 80],\n  },\n};\n";
        let r = parse("commitlint.config.js", js).unwrap();
        assert_eq!(types(&r), ["feat", "fix", "docs", "chore"]);
        assert_eq!(r.header_max, Some(80));

        let yaml = "extends:\n  - '@commitlint/config-conventional'\nrules:\n  type-enum: [2, always, [feat, fix, ops]]\n  header-max-length: [2, always, 90]\n";
        let r = parse(".commitlintrc.yml", yaml).unwrap();
        assert_eq!(types(&r), ["feat", "fix", "ops"]);
        assert_eq!(r.header_max, Some(90));

        // rules built from variables cannot be read here: defaults
        let r = parse(
            "commitlint.config.ts",
            "export default { rules: { 'type-enum': [2, 'always', TYPES] } };",
        )
        .unwrap();
        assert_eq!(r, Rules::default());
    }

    #[test]
    fn the_angular_preset_has_no_chore() {
        let r = parse(
            "commitlint.config.cjs",
            "module.exports = { extends: ['@commitlint/config-angular'] };",
        )
        .unwrap();
        assert!(!r.allows("chore") && r.allows("feat"));
        let r = parse(
            ".commitlintrc.json",
            r#"{"extends": "@commitlint/config-angular"}"#,
        )
        .unwrap();
        assert!(!r.allows("chore"));
    }

    #[test]
    fn found_from_a_subdirectory_up_to_the_top() {
        let top = tempfile::tempdir().unwrap();
        let sub = top.path().join("packages/app");
        std::fs::create_dir_all(&sub).unwrap();
        assert_eq!(load(&sub, top.path()), Rules::default());
        std::fs::write(
            top.path().join(".commitlintrc.json"),
            r#"{"rules": {"header-max-length": [2, "always", 72]}}"#,
        )
        .unwrap();
        let r = load(&sub, top.path());
        assert_eq!(r.header_max, Some(72));
        assert_eq!(r.file, Some(top.path().join(".commitlintrc.json")));
        assert_eq!(r.header_max(), 72);
        assert_eq!(Rules::default().header_max(), message::MAX_HEADER_LEN);
    }

    #[test]
    fn type_words() {
        assert!(is_type("feat") && is_type("deps-dev") && is_type("v2"));
        assert!(!is_type("Feat") && !is_type("") && !is_type("2x") && !is_type("a b"));
    }
}
