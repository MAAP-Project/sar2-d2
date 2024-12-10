#!/usr/bin/env bash

# Install dependencies from the environment*.yml files into the conda environment,
# first creating the environment, if necessary.  If the environment is already up to
# date with the environment*.yml files, nothing happens. Regardless, outputs the conda
# environment absolute path of the environment directory (a.k.a., the environment
# "prefix").

set -Eeuo pipefail

conda=${CONDA_EXE:-conda}
thisdir=$(dirname "$(readlink -f "$0")")
basedir=$(dirname "$(dirname "${thisdir}")")

# Make sure that the SAR2D2_ENV shell environment variable
# is set. It can be set in the JN, build.sh, run-dps.sh, and conda/run.sh.
if [[ ! -v SAR2D2_ENV ]]; then
    echo "ERROR: The SAR2D2_ENV environment variable must be set to the name of the conda environment to create/update." 1>&2
    exit 1
fi

function environment_prefix() {
    # We must call "conda run" directly, rather than using our run.sh script,
    # otherwise we end up in an infinite scripting loop.

    # The value of `result` will either be the environment "prefix" (directory) when the
    # environment exists (captured from stdout), or an error message (captured from
    # stderr) containing the directory as a suffix when the environment does not exist.
    result=$("${conda}" run --no-capture-output --name "${SAR2D2_ENV}" printenv CONDA_PREFIX 2>&1 || true)

    # In either case, the "prefix" directory starts with a foward-slash (`/`) and
    # continues to the end of the captured output, so we tell grep to give us only
    # the directory path, ignoring any prefix, if the result is an error message.
    echo -n "${result}" | grep --only-matching "/.*$"
}

function update_environment() {
    (
        set -x
        PIP_REQUIRE_VENV=0 "${conda}" env update --quiet --solver libmamba \
            --name "${SAR2D2_ENV}" --file "${1}"
    )
}

prefix=$(environment_prefix)

if [[ $# -gt 0 ]]; then
    env_files=("${@}")
else
    env_files=("${basedir}"/environment.yml "${basedir}"/environment-dev.yml)
fi

# Create list of env files that have been created/modified (i.e., are newer than [-nt])
# the conda environment.
if [[ ! -d "${prefix}" ]]; then
    # The conda env doesn't exist, so the environment must be updated with all of the
    # env files.  (The first update will force creation of the environment.)
    updated_env_files=("${env_files[@]}")
else
    # The conda env exists, so collect each env file that has been created or modified
    # since the env was created or last updated.
    for env_file in "${env_files[@]}"; do
        if [[ "${env_file}" -nt "${prefix}" ]]; then
            updated_env_files=("${updated_env_files[@]}" "${env_file}")
        fi
    done
fi

# Update the environment with each env file that is newer than the environment.
for updated_env_file in "${updated_env_files[@]}"; do
    update_environment "${updated_env_file}"
done

# Touch the conda env dir, just in case updating the environment did not produce
# any changes (i.e., all dependencies are already satisified), otherwise, running
# this script again, may still consider the environment to be out of date, and would
# needlessly re-run the `conda env update` command for at least one of the
# environment*.yml files.
touch "${prefix}"

echo "${prefix}"
