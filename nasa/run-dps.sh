#!/usr/bin/env bash

set -Eeuo pipefail

thisdir=$(dirname "$(readlink -f "$0")")

# For DPS, hardcode the `SAR2D2_ENV` variable to match the
# name of the custom environment in the environment.yml file
# and in the build.sh script. This ensures we're using the correct
# environment setup and dependencies during processing.
# After, call the standard run.sh script.
SAR2D2_ENV=sar2-d2 "${thisdir}"/run.sh "$@"
