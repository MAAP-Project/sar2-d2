#!/usr/bin/env bash

set -Eeuo pipefail

basedir=$(dirname "$(dirname "$(readlink -f "$0")")")
input_dir="${PWD}/input"
output_dir="${PWD}/output"

if [[ $# == 0 || "$1" == "-h" ]]; then
    echo "usage: nasa/run.sh [-h] CALIBRATION_FILE LEFT BOTTOM RIGHT TOP

SAR2-D2: Synthetic Aperture Radar Remote Disturbance Detector

positional arguments:
  CALIBRATION_FILE      path to calibration file
  LEFT                  left longitude of your desired bounding box
  BOTTOM                bottom latitude of your desired bounding box
  RIGHT                 right longitude of your desired bounding box
  TOP                   top latitude of your desired bounding box

options:
  -h, --help            show this help message and exit"
    exit 1
fi

if test -d "${input_dir}"; then
    # RUNNING IN DPS
    #
    # There is an `input` sub-directory of the current working directory, so assume the
    # calibration file is the sole file within the `input` sub-directory that the DPS
    # automatically downloaded for us, based upon the URL given as the value of the
    # calibration_file job input.
    ls -l "${input_dir}" >&2

    calibration_file="$(ls "${input_dir}"/*)"
    bbox="$1"
else
    # RUNNING IN DEVELOPMENT ENVIRONMENT
    #
    # There is no `input` sub-directory of the current working directory, so simply
    # pass all arguments through to the Python script.  This is useful for testing in a
    # non-DPS environment, where there is no `input` directory since the DPS creates
    # the `input` directory.  In this case, we must pass a path to a "local" file,
    # which can be a path to a file in a shared bucket in the ADE.
    calibration_file="$1"

    if [[ $# -eq 5 ]]; then
        # Allow us to pass the coordinates as individual arguments, so we don't need to
        # remember to put quotes around them:
        #
        #   nasa/run.sh path/to/calibration-file LEFT BOTTOM RIGHT TOP
        #
        bbox="$2 $3 $4 $5"
    else
        # Also allow us to put quotes around the 4 coordinates:
        #
        #   nasa/run.sh path/to/calibration-file "LEFT BOTTOM RIGHT TOP"
        #
        bbox="$2"
    fi
fi

# WARNING: DO NOT place quotes around ${bbox} in the following command.  The
# bbox should be 4 coordinates, each as individual (repeated) bbox arguments,
# thus, we do not want to use quotes because that would cause the 4 individual
# coordinates to instead be treated as a single string argument, not 4 float
# arguments.

# shellcheck disable=SC2086
"${basedir}"/bin/conda/run.sh python "${basedir}/sar2_d2/cli.py" \
    --calibration-file "${calibration_file}" \
    --bbox ${bbox} \
    --output-dir "${output_dir}"
