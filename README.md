# SAR2-D2: Synthetic Aperture Radar Remote Disturbance Detector

## Running the NASA algorithm locally

You can run the NASA algorithm locally like so:

```plain
nasa/run.sh CALIBRATION_FILE LEFT BOTTOM RIGHT TOP
```

where:

- `CALIBRATION_FILE` is the path to calibration file on the file system
- `LEFT` is the left longitude of your desired bounding box
- `BOTTOM` is the bottom latitude of your desired bounding box
- `RIGHT` is the right longitude of your desired bounding box
- `TOP` is the top latitude of your desired bounding box

As a convenience, this will _automatically_ do the following for you:

1. Create a conda environment named `sar2-d2`, if it does not already exist.
1. Install all dependencies in `environment.yml` and `environment-dev.yml`
   whenever the environment is (re-)created or one (or both) of the files have
   been modified since the environment was (re-)created.
1. Run the code with the `sar2-d2` environment automatically activated (i.e.,
   there is no need for you to manually activate the environment).

## Registering the algorithm

To register the algorithm (or re-register it after making code changes), run the
following command:

```plain
nasa/algorithm/register.py
```

This will kick off a process to build a new algorithm image, and will output a
link where you can check the progress of this process, which may take several
minutes.  The new version of the algorithm will not be available for submitting
jobs until this process completes successfully.  Once the process completes
successfully, the new version of the algorithm is considered "registered."

## Submitting a job

Once the algorithm is registered (or re-registered), you may submit a job using
the following command:

```plain
nasa/job/submit.py CALIBRATION_FILE LEFT BOTTOM RIGHT TOP
```

where the arguments are the same as those used when running the algorithm
locally (see above), with the following difference: if `CALIBRATION_FILE` is
_not_ a URL, it will automatically be converted to a URL if its path starts with
`/projects/my-public-bucket` or `/projects/shared-bucket/USERNAME` (where
`USERNAME` is the username of any MAAP user, not necessarily your username).
