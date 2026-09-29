#!/bin/sh
# CI check for install.sh: serve a locally built gca as a release, then
# install, reinstall and uninstall it. Run from the repository root after
# `cargo build --bin gca`.
set -eu

case "$(uname -s)" in
Linux) asset=gca-x86_64-unknown-linux-musl ;;
Darwin)
    if [ "$(uname -m)" = arm64 ]; then asset=gca-aarch64-apple-darwin; else asset=gca-x86_64-apple-darwin; fi
    ;;
*)
    echo "unsupported runner" >&2
    exit 1
    ;;
esac

release="$RUNNER_TEMP/release"
mkdir -p "$release"
cp gca-rs/target/debug/gca "$release/$asset"
(cd "$release" && shasum -a 256 "$asset" >SHA256SUMS)
python3 -m http.server 8765 --bind 127.0.0.1 --directory "$release" >/dev/null 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT
sleep 2

export GCA_DOWNLOAD_URL=http://127.0.0.1:8765 GCA_INSTALL_DIR="$RUNNER_TEMP/bin"
fail() {
    echo "$*" >&2
    exit 1
}
marks() { cat "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc" "$HOME/.profile" 2>/dev/null | grep -c "gca installer" || true; }

sh install.sh
"$GCA_INSTALL_DIR/gca" --version
[ "$(marks)" = 1 ] || fail "expected one PATH line in the shell profile, found $(marks)"
sh install.sh
[ "$(marks)" = 1 ] || fail "reinstalling added another PATH line"
sh install.sh --uninstall
[ ! -e "$GCA_INSTALL_DIR/gca" ] || fail "gca is still installed"
[ "$(marks)" = 0 ] || fail "the PATH line was left in the shell profile"
echo "install.sh works on $(uname -s)"
