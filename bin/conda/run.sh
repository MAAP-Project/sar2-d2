#!/usr/bin/env bash

set -Eeuo pipefail

conda=${CONDA_EXE:-conda}
thisdir=$(dirname "$(readlink -f "$0")")

# We must be sure the environment exists before we can run things in it.
prefix=$("${thisdir}"/install.sh)
envname=$(basename "${prefix}")

# All arguments are passed directly to `conda run`
"${conda}" run --no-capture-output --name "${envname}" "${@}"
