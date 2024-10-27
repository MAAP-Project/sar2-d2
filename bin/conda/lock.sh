#!/usr/bin/env bash

set -Eeuo pipefail

thisdir=$(dirname "$(readlink -f "$0")")
basedir=$(dirname "$(dirname "${thisdir}")")

"${thisdir}"/create.sh

# If the lock file does not exist or it is outdated, regenerate it.
if [[ 
    ! -f "${basedir}/conda-lock.yml" ||
    "${basedir}/environment.yml" -nt "${basedir}/conda-lock.yml" ||
    "${basedir}/environment-dev.yml" -nt "${basedir}/conda-lock.yml" ]]; then

    if [[ "${PRE_COMMIT:-}" == "1" ]]; then
        # In pre-commit.ci, conda is not available, so we cannot generate the
        # lock file.  Instead, we must simply issue an error message and fail
        # the build.
        echo >&2 "The conda-lock.yml file is out of date.  Run 'bin/conda/lock.sh' to update it, then commit and push the changes."
        exit 1
    fi

    conda=${CONDA_EXE:-conda}
    envname=$("${thisdir}"/name.sh)

    # We must call "conda run" directly, rather than using our run.sh script,
    # otherwise we end up in an infinite scripting loop.
    "${conda}" run --no-capture-output --name "${envname}" conda lock --mamba -f "${basedir}/environment.yml" -f "${basedir}/environment-dev.yml"
fi
