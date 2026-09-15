#!/usr/bin/env bash
# build.sh — build every CLI under cmd/ into bin/ (Git Bash / Windows).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p bin

GO="${GO:-go}"

build() {
    # build <cmd-dir> <bin-name>
    local cmd="$1" name="$2"
    echo "==> building $name"
    "$GO" build -trimpath -o "bin/$name.exe" "./cmd/$cmd"
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