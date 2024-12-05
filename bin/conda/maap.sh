#!/usr/bin/env bash

set -Eeuo pipefail

conda=${CONDA_EXE:-conda}
thisdir=$(dirname "$(readlink -f "$0")")
env_file=${thisdir}/maap.yml
prefix=${HOME}/.conda/envs/maap

# If the maap.yml file is newer than the conda env directory (prefix), then
# update the conda env.  This should occur only the very first time this script
# is executed, creating the conda env.  All subsequent uses of this script will
# should run without updating the environment, as the maap.yml should not need
# to be updated.

if [[ "${env_file}" -nt "${prefix}" ]]; then
    echo 1>&2
    echo 1>&2 "Performing initial preparation for running MAAP algorithm and job commands."
    echo 1>&2 "Subsequent algorithm and job commands will run without pause."
    echo 1>&2
    echo 1>&2 "This one-time step will take only a few moments ..."
    echo 1>&2

    "${conda}" env update --solver=libmamba --prefix="${prefix}" --file="${env_file}" >/dev/null
fi

"${conda}" run --no-capture-output --prefix "${prefix}" "${@}"
