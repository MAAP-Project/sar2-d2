#!/usr/bin/env bash

set -Eeuo pipefail

conda=${CONDA_EXE:-conda}
thisdir=$(dirname "$(readlink -f "$0")")
envname=$("${thisdir}/name.sh")

"${thisdir}"/create.sh

# We must call "conda run" directly, rather than using our run.sh script,
# otherwise we end up in an infinite scripting loop.
"${conda}" run --no-capture-output --name "${envname}" printenv CONDA_PREFIX
