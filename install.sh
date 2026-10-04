#!/bin/sh
# Install gca on macOS or Linux from the latest GitHub release:
#
#   curl -fsSL https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.sh | sh
#
# Run it again to upgrade. To remove gca:
#
#   curl -fsSL https://raw.githubusercontent.com/chainchen1103/git-diff-type/main/install.sh | sh -s -- --uninstall
#
# Settings, as environment variables:
#   GCA_VERSION=v0.6.1      install this release instead of the latest
#   GCA_INSTALL_DIR=DIR     install into DIR instead of ~/.local/bin
#   GCA_NO_MODIFY_PATH=1    leave your shell profile alone
#   GCA_DOWNLOAD_URL=URL    download from URL instead of GitHub (a mirror, or a test)
#
# The download is checked against the release's SHA256SUMS before anything is
# installed. Prebuilt binaries cover macOS (Apple silicon and Intel) and Linux
# on x86_64; elsewhere, build gca with Rust (see the README).
#
# Everything runs inside main(), so a download cut off halfway runs nothing.

set -eu

REPO=chainchen1103/git-diff-type
RAW=https://raw.githubusercontent.com/$REPO/main
MARKER="# added by the gca installer"

say() { printf '%s\n' "$*"; }
die() { printf 'gca install: %s\n' "$*" >&2; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }

usage() {
    say "usage: install.sh [--uninstall]"
    say "settings: GCA_VERSION, GCA_INSTALL_DIR, GCA_NO_MODIFY_PATH, GCA_DOWNLOAD_URL"
}

# The release asset for this machine, named as .github/workflows/release.yml
# names it.
target() {
    os=$(uname -s)
    arch=$(uname -m)
    case "$os" in
    Darwin)
        # A shell running under Rosetta reports x86_64 on Apple silicon.
        if [ "$arch" = arm64 ] || [ "$(sysctl -n hw.optional.arm64 2>/dev/null || true)" = 1 ]; then
            say aarch64-apple-darwin
        elif [ "$arch" = x86_64 ]; then
            say x86_64-apple-darwin
        else
            unsupported "$os" "$arch"
        fi
        ;;
    Linux)
        case "$arch" in
        x86_64 | amd64) say x86_64-unknown-linux-musl ;;
        *) unsupported "$os" "$arch" ;;
        esac
        ;;
    MINGW* | MSYS* | CYGWIN*)
        die "on Windows, run this in PowerShell instead: irm $RAW/install.ps1 | iex"
        ;;
    *) unsupported "$os" "$arch" ;;
    esac
}

unsupported() {
    die "there is no prebuilt gca for $1 $2; build it with Rust 1.80 or later:
  cargo install --git https://github.com/$REPO gca-rs --bin gca"
}

download_base() {
    if [ -n "${GCA_DOWNLOAD_URL:-}" ]; then
        say "${GCA_DOWNLOAD_URL%/}"
    elif [ -n "${GCA_VERSION:-}" ]; then
        say "https://github.com/$REPO/releases/download/$GCA_VERSION"
    else
        say "https://github.com/$REPO/releases/latest/download"
    fi
}

# fetch URL FILE
fetch() {
    if has curl; then
        if [ -n "${GCA_DOWNLOAD_URL:-}" ]; then
            curl -fsSL --retry 3 -o "$2" "$1"
        else
            curl -fsSL --retry 3 --proto '=https' --proto-redir '=https' --tlsv1.2 -o "$2" "$1"
        fi
    elif has wget; then
        wget -q -O "$2" "$1"
    else
        die "curl or wget is needed to download gca"
    fi || die "could not download $1
  (is there a release yet? see https://github.com/$REPO/releases)"
}

sha256() {
    if has sha256sum; then
        sha256sum "$1" | cut -d ' ' -f 1
    elif has shasum; then
        shasum -a 256 "$1" | cut -d ' ' -f 1
    else
        die "sha256sum or shasum is needed to check the download"
    fi
}

# verify FILE NAME SUMS: FILE's hash must be the one SUMS lists for NAME.
verify() {
    expected=$(awk -v name="$2" '$2 == name || $2 == "*" name { print $1; exit }' "$3")
    [ -n "$expected" ] || die "SHA256SUMS lists no $2; nothing was installed"
    actual=$(sha256 "$1")
    [ "$actual" = "$expected" ] ||
        die "checksum mismatch for $2 (expected $expected, got $actual); nothing was installed"
}

# The file that sets PATH for the user's login shell, and the line to add.
profile() {
    case "$1" in
    "$HOME"/*) shown="\$HOME/${1#"$HOME"/}" ;;
    *) shown=$1 ;;
    esac
    case "$(basename "${SHELL:-sh}")" in
    zsh)
        rc="${ZDOTDIR:-$HOME}/.zshrc"
        line="export PATH=\"$shown:\$PATH\""
        ;;
    bash)
        if [ "$(uname -s)" = Darwin ]; then rc="$HOME/.bash_profile"; else rc="$HOME/.bashrc"; fi
        line="export PATH=\"$shown:\$PATH\""
        ;;
    fish)
        rc="$HOME/.config/fish/conf.d/gca.fish"
        line="fish_add_path \"$1\""
        ;;
    *)
        rc="$HOME/.profile"
        line="export PATH=\"$shown:\$PATH\""
        ;;
    esac
}

add_to_path() {
    case ":${PATH:-}:" in
    *":$1:"*) return ;;
    esac
    profile "$1"
    if [ "${GCA_NO_MODIFY_PATH:-0}" = 1 ]; then
        say "$1 is not on your PATH; add it with:  $line"
        return
    fi
    if ! grep -qsxF "$line" "$rc"; then
        mkdir -p "$(dirname "$rc")"
        printf '\n%s\n%s\n' "$MARKER" "$line" >>"$rc"
        say "added $1 to PATH in $rc"
    fi
    say "open a new terminal to use gca, or run:  $line"
}

# Remove the marker and the line after it from every profile the installer
# may have written, leaving the rest of each file as it was.
remove_from_path() {
    for rc in "${ZDOTDIR:-$HOME}/.zshrc" "$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.profile"; do
        if [ ! -f "$rc" ] || ! grep -qF "$MARKER" "$rc"; then continue; fi
        kept=$(awk -v marker="$MARKER" 'drop { drop = 0; next } $0 == marker { drop = 1; next } { print }' "$rc")
        printf '%s\n' "$kept" >"$rc"
        say "removed the PATH line from $rc"
    done
    fish="$HOME/.config/fish/conf.d/gca.fish"
    if [ -f "$fish" ]; then
        rm -f "$fish"
        say "removed $fish"
    fi
}

install_gca() {
    dir=$1
    asset="gca-$(target)"
    base=$(download_base)
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    say "downloading $asset from $base"
    fetch "$base/$asset" "$tmp/$asset"
    fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS"
    verify "$tmp/$asset" "$asset" "$tmp/SHA256SUMS"
    chmod 755 "$tmp/$asset"
    "$tmp/$asset" --version >/dev/null 2>&1 || die "the downloaded gca does not run on this machine"

    mkdir -p "$dir"
    # Copy next to the target, then rename, so a running gca is never half-written.
    cp "$tmp/$asset" "$dir/.gca.new"
    mv -f "$dir/.gca.new" "$dir/gca"
    say "installed $("$dir/gca" --version) to $dir/gca"

    add_to_path "$dir"
    found=$(command -v gca 2>/dev/null || true)
    if [ -n "$found" ] && [ "$found" != "$dir/gca" ]; then
        say "note: $found comes first on your PATH and will run instead"
    fi
    has git || say "note: gca runs git, which is not installed yet"
    say "stage some changes and run gca; see gca --help"
}

uninstall_gca() {
    dir=$1
    if [ -f "$dir/gca" ]; then
        rm -f "$dir/gca"
        say "removed $dir/gca"
    else
        say "gca is not installed in $dir"
    fi
    remove_from_path
    found=$(command -v gca 2>/dev/null || true)
    if [ -n "$found" ] && [ "$found" != "$dir/gca" ]; then
        say "note: another gca is still installed at $found"
    fi
    # where `gca model install` puts the models
    models=${GCA_MODELS_DIR:-}
    if [ -z "$models" ]; then
        case "$(uname -s)" in
        Darwin) models="$HOME/Library/Application Support/gca/models" ;;
        *) models="${XDG_DATA_HOME:-$HOME/.local/share}/gca/models" ;;
        esac
    fi
    if [ -d "$models" ]; then
        say "the models gca downloaded stay in $models; delete that folder to remove them"
    fi
    say "settings stay in your git config; remove them with:  git config --global --remove-section gca"
}

main() {
    uninstall=0
    for arg in "$@"; do
        case "$arg" in
        --uninstall) uninstall=1 ;;
        -h | --help)
            usage
            return
            ;;
        *) die "unknown option $arg (see --help)" ;;
        esac
    done
    [ -n "${HOME:-}" ] || die "HOME is not set"
    dir=${GCA_INSTALL_DIR:-$HOME/.local/bin}
    if [ "$uninstall" = 1 ]; then
        uninstall_gca "$dir"
    else
        install_gca "$dir"
    fi
}

main "$@"
