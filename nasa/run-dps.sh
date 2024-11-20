#!/usr/bin/env bash

set -Eeuo pipefail

thisdir=$(dirname "$(readlink -f "$0")")

SAR2D2_ENV=sar2-d2 "${thisdir}"/run.sh "$@"
