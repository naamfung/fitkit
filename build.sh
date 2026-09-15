#!/usr/bin/env bash
# build.sh — build every CLI under cmd/ into bin/ (POSIX shell: Git Bash / WSL / Linux).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p bin

GO="${GO:-go}"

# Windows (Git Bash/MSYS/Cygwin) appends .exe; POSIX keeps the extensionless name.
EXT=""
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) EXT=".exe" ;;
esac

build() {
    # build <cmd-dir> <bin-name>
    local cmd="$1" name="$2"
    echo "==> building $name"
    "$GO" build -trimpath -o "bin/$name$EXT" "./cmd/$cmd"
}

build fitting    fitting
build fitfidelity fitfidelity
build fitcalibrate fitcalibrate
build fitregistry fitregistry
build fitdry      fitdry
build fitpoc      fitpoc
build plancheck   plancheck

echo "done:"
ls -l bin