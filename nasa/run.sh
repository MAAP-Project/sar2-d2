#!/usr/bin/env bash

set -Eeuo pipefail

basedir=$(dirname "$(dirname "$(readlink -f "$0")")")
input_dir="${PWD}/input"
output_dir="${PWD}/output"

if [[ $# == 0 || "$1" == "-h" ]]; then
    echo "usage: nasa/run.sh [-h] INPUT_FILE LEFT BOTTOM RIGHT TOP INPUT_PRODUCT_TYPE OUTPUT_PRODUCT_TYPE GCOV_POSTING

SAR2-D2: Synthetic Aperture Radar Remote Disturbance Detector

positional arguments:
  INPUT_FILE            URL http path to input file
  LEFT                  left longitude of bounding box of the input file
  BOTTOM                bottom latitude of the input file
  RIGHT                 right longitude of the input file
  TOP                   top latitude of the input file
  INPUT_PRODUCT_TYPE    Product type of the input file. One of: 'alos1', 'l0b', 'rslc'.
                        Will be processed through the ALOS1->L0B->RSLC->GCOV pipeline,
                        ending with type specified by 'out_type'.
  OUTPUT_PRODUCT_TYPE   Product type of the output file. One of: 'l0b', 'rslc', 'gcov'.
                        In combination with 'in_type', determines which processing steps
                        occur.
  GCOV_POSTING          Posting for pixel spacing for GCOV (square pixels). Must be
                        in units of EPSG 4326 (same as DEM and bbox coordinates).
                        (100 meters is ~0.000833334 degrees.)"
    exit 1
fi

# if test -d "${input_dir}"; then
#     # RUNNING IN DPS
#     #
#     # There is an `input` sub-directory of the current working directory, so assume the
#     # input file is the sole file within the `input` sub-directory that the DPS
#     # automatically downloaded for us, based upon the URL given as the value of the
#     # in_file job input.
#     ls -l "${input_dir}" >&2

#     in_file="$(ls "${input_dir}"/*)"
#     bbox="$1"
#     in_type="$2"
#     out_type="$3"
#     posting="$4"
# else
#     # RUNNING IN DEVELOPMENT ENVIRONMENT
#     #
#     # There is no `input` sub-directory of the current working directory, so simply
#     # pass all arguments through to the Python script.  This is useful for testing in a
#     # non-DPS environment, where there is no `input` directory since the DPS creates
#     # the `input` directory.  In this case, we must pass a path to a "local" file,
#     # which can be a path to a file in a shared bucket in the ADE.
#     in_file="$1"

#     if [[ $# -eq 8 ]]; then
#         # Allow us to pass the coordinates as individual arguments, so we don't need to
#         # remember to put quotes around them:
#         #
#         #   nasa/run.sh path/to/in-file LEFT BOTTOM RIGHT TOP  [...]
#         #
#         bbox="$2 $3 $4 $5"
#         in_type="$6"
#         out_type="$7"
#         posting="$8"
#     else
#         # Also allow us to put quotes around the 4 coordinates:
#         #
#         #   nasa/run.sh path/to/in-file "LEFT BOTTOM RIGHT TOP" [...]
#         #
#         bbox="$2"
#         in_type="$3"
#         out_type="$4"
#         posting="$5"
#     fi
# fi

## BEGIN HACK ##
# HACK: DPS cannot download ALOS-1 data from ASF into input directory;
# this is being handled internal to the algorithm for now.
# We'll simply always pass in a http URL for the granule
in_file="$1"

if [[ $# -eq 8 ]]; then
    # Allow us to pass the coordinates as individual arguments, so we don't need to
    # remember to put quotes around them:
    #
    #   nasa/run.sh path/to/calibration-file LEFT BOTTOM RIGHT TOP
    #
    bbox="$2 $3 $4 $5"
    in_type="$6"
    out_type="$7"
    posting="$8"
else
    # Also allow us to put quotes around the 4 coordinates:
    #
    #   nasa/run.sh path/to/calibration-file "LEFT BOTTOM RIGHT TOP"
    #
    bbox="$2"
    in_type="$3"
    out_type="$4"
    posting="$5"
fi
## END HACK ##

# WARNING: DO NOT place quotes around ${bbox} in the following command.  The
# bbox should be 4 coordinates, each as individual (repeated) bbox arguments,
# thus, we do not want to use quotes because that would cause the 4 individual
# coordinates to instead be treated as a single string argument, not 4 float
# arguments.

# shellcheck disable=SC2086
"${basedir}"/bin/conda/run.sh python "${basedir}/src/alos1_2_gcov/alos1_to_gcov.py" \
    --in-file "${in_file}" \
    --bbox ${bbox} \
    --out-dir "${output_dir}" \
    --in_type "${in_type}" \
    --out_type "${out_type}" \
    --gcov_posting "${posting}"
# --dem /projects/alos1-dist/sar2-d2/nasa/output/dem.tif
