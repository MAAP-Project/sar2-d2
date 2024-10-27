#!/usr/bin/env bash

set -Eeuo pipefail

basedir=$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")
envname=$(basename "${basedir}")

echo "${envname}"
