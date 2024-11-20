#!/usr/bin/env bash

set -Eeuo pipefail

conda=${CONDA_EXE:-conda}
thisdir=$(dirname "$(readlink -f "$0")")

if [[ ! -v SAR2D2_ENV ]]; then
    echo "ERROR: The SAR2D2_ENV environment variable must be set to the name of the conda environment to run in." 1>&2
    exit 1
fi

# All arguments are passed directly to `conda run`
"${conda}" run --no-capture-output --name "${SAR2D2_ENV}" "${@}"
