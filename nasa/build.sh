#!/usr/bin/env bash

set -Eeuo pipefail

basedir=$(dirname "$(dirname "$(readlink -f "$0")")")

# Build *without* development dependencies.  Unfortunately, DPS does not allow
# us to pass arguments to an algorithm's "build" script, so this script is
# required as a wrapper for DPS to call.

"${basedir}"/bin/conda/install.sh "${basedir}/environment.yml"
