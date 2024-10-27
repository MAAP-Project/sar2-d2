#!/usr/bin/env bash

set -Eeuo pipefail

thisdir=$(dirname "$(readlink -f "$0")")
basedir=$(dirname "$(dirname "${thisdir}")")

# We must make sure the lock file is up to date before installing the
# dependencies listed within it.
# "${thisdir}"/lock.sh

envname=$("${thisdir}"/name.sh)
envdir=$("${thisdir}"/prefix.sh)

#-------------------------------------------------------------------------------
# conda lock is not behaving well with isce3/cuda, so commenting out for now.
#
# # If the conda lock file is newer than (-nt) the conda environment (i.e., the
# # environment's "prefix" directory), install the dependencies from the lock file.
# if [[ "${basedir}"/conda-lock.yml -nt "${envdir}" ]]; then
#     # Since there is at least one package (maap-py) that is not available on
#     # conda-forge, we need to use pip to install it (conda does this for us), so
#     # we must set PIP_REQUIRE_VENV=0 to avoid complaints about installing packages
#     # outside of a virtual environment, in case the user has set that env var to a
#     # "truthy" value for direct pip usage.
#     PIP_REQUIRE_VENV=0 "${thisdir}"/run.sh conda lock install --name "${envname}" "$@" "${basedir}"/conda-lock.yml
# fi
#-------------------------------------------------------------------------------

if [[ "${basedir}"/environment.yml -nt "${envdir}" || "${basedir}"/environment-dev.yml -nt "${envdir}" ]]; then
    conda=${CONDA_EXE:-conda}
    set -x
    PIP_REQUIRE_VENV=0 "${conda}" env update --quiet --solver libmamba \
        --name "${envname}" --file "${basedir}"/environment.yml

    # Allow --no-dev flag (from nasa/build.sh) to *prevent* installation of
    # development dependencies, since we don't need (nor want) them in the DPS.
    if [[ "${1:-}" != "--no-dev" ]]; then
        PIP_REQUIRE_VENV=0 "${conda}" env update --quiet --solver libmamba \
            --name "${envname}" --file "${basedir}"/environment-dev.yml
    fi
fi
