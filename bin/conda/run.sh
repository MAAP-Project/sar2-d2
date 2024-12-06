#!/usr/bin/env bash

set -Eeuo pipefail

conda=${CONDA_EXE:-conda}

if [[ ! -v SAR2D2_ENV ]]; then
    echo "ERROR: The SAR2D2_ENV environment variable must be set to the name of the conda environment to run in." 1>&2
    echo "" 1>&2
    echo "For example:" 1>&2
    echo "" 1>&2
    echo "    SAR2D2_ENV=my_env command arg1 arg2 ..." 1>&2
    echo "" 1>&2
    exit 1
fi

"${conda}" list -n "${SAR2D2_ENV}" cuda

# All arguments are passed directly to `conda run`
"${conda}" run --no-capture-output --name "${SAR2D2_ENV}" "${@}"
