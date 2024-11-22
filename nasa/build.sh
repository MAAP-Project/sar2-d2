#!/usr/bin/env bash

set -Eeuo pipefail

basedir=$(dirname "$(dirname "$(readlink -f "$0")")")

# Build *without* development dependencies.  Unfortunately, DPS does not allow
# us to pass arguments to an algorithm's "build" script, so this script is
# required as a wrapper for DPS to call.

# When we build the algorithm container, we need to build the
# custom environment into the algorithm container.
# Later, when we use the run-dps.sh script to run in DPS,
# make sure that script uses the same conda environment.
# Note: This is where the name of the conda environment gets set.
# (It is not set per the `name` in the environment.yml.)
SAR2D2_ENV=sar2-d2 "${basedir}"/bin/conda/install.sh "${basedir}/environment.yml"
