#!/usr/bin/env bash

set -Eeuo pipefail

conda=${CONDA_EXE:-conda}
thisdir=$(dirname "$(readlink -f "$0")")
envname=$("${thisdir}/name.sh")

# Create the conda environment if it doesn't exist.
if ! "${conda}" env list | grep -q "/${envname}$"; then
    # Initialize the environment with conda and conda-lock so we can use
    # conda-lock to install dependencies listed in the lock file.
    set -x
    "${conda}" create --no-default-packages --quiet --name "${envname}" --solver libmamba --yes conda~=24.0 conda-lock~=2.0
fi
