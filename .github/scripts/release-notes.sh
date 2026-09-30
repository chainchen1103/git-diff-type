#!/bin/sh
# Print the notes of a release: its section of CHANGELOG.md, without the
# heading. Takes the version, with or without a leading v. Fails when
# CHANGELOG.md has no section for it.
#
#   sh .github/scripts/release-notes.sh v0.4.0 > notes.md
set -eu

version="${1#v}"
notes=$(awk -v v="$version" '
    /^## / { if (found) exit; if ($2 == v) { found = 1; next } }
    found
' CHANGELOG.md | sed '/./,$!d')
if [ -z "$notes" ]; then
    echo "CHANGELOG.md has no section for $version" >&2
    exit 1
fi
printf '%s\n' "$notes"
