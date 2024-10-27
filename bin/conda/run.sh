#!/usr/bin/env bash

set -Eeuo pipefail

conda=${CONDA_EXE:-conda}
thisdir=$(dirname "$(readlink -f "$0")")
envname=$("${thisdir}"/name.sh)

# We must be sure the environment exists before we can run things in it.
"${thisdir}"/install.sh

# All arguments are passed to conda run, except --quiet, which is used to
# suppress printing commands and arguments as they are executed.

command=("${conda}" run --no-capture-output --name "${envname}")
quiet=

while ((${#})); do
    case "${1}" in
    --quiet)
        quiet=1
        ;;
    *)
        command+=("${1}")
        ;;
    esac
    shift
done

[[ -n "${quiet}" ]] || set -x
"${command[@]}"
