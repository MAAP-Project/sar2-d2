# SAR2-D2: Synthetic Aperture Radar Remote Disturbance Detector

## Running the NASA algorithm locally

If you need to create a new environment, or update an existing one, with
dependencies from the `environment.yml` file, run the following, where
`<conda-env>` is the name of either an existing environment you want to
update, or the name of a new environment to create:

```plain
conda env --name <conda-env> update
```

You can run the NASA algorithm locally like so:

```plain
SAR2D2_ENV=<conda-env> nasa/run.sh CALIBRATION_FILE LEFT BOTTOM RIGHT TOP
```

where:

- `<conda-env>` is the name of the conda environment to use, which should
  be the same name used with the previous "update" command, if you ran it
- `CALIBRATION_FILE` is the path to calibration file on the file system
- `LEFT` is the left longitude of your desired bounding box
- `BOTTOM` is the bottom latitude of your desired bounding box
- `RIGHT` is the right longitude of your desired bounding box
- `TOP` is the top latitude of your desired bounding box

## Registering the algorithm

To register the algorithm (or re-register it after making code changes), run the
following command:

```plain
nasa/algorithm/register.py
```

> [!NOTE]
>
> The *first* time you run this command, it will *automatically* create an
> isolated conda environment separate from your main environment.  This will
> take a few moments, but subsequent algorithm and job commands will run much
> more quickly, as the environment will be reused.  Further, the environment is
> created in a location that persists across instance restarts.
>
> This separate environment is used automatically behind the scenes for running
> all algorithm and job scripts that use `maap-py`.  This environment is built
> to include *only* `maap-py` and its dependencies, and you should never need to
> use this environment directly, nor even know where it is.

This will kick off a process to build a new algorithm image, and will output a
link where you can check the progress of this process, which may take several
minutes.  The new version of the algorithm will not be available for submitting
jobs until this process completes successfully.  Once the process completes
successfully, the new version of the algorithm is considered "registered."

> [!NOTE]
>
> The *version* of the algorithm that is registered is set to the name of the
> current branch that you have checked out, unless you have added a tag to the
> current commit, in which case the version is set to the value of the tag.

## Submitting a job

Once the algorithm is registered (or re-registered), you may submit a job using
the following command:

```plain
nasa/job/submit.py ARGS...
```

As with the registration command above, the *version* of the algorithm that is
used is either the name of the current branch, or the value of the tag of the
current commit, if there is a tag.  However, you *can* specify the `--version`
option to override this, if you ever want to run a different version of the
algorithm.

The expected `ARGS` are *automatically* determined by the inputs defined in the
`nasa/algorithm.yml` file.  Therefore, to see the expected arguments, run the
following command:

```plain
nasa/job/submit.py -h
```

> [!NOTE]
>
> By default, the queue specified in the `nasa/algorithm.yml` file will be used,
> but you can override this by including the `-q/--queue` option, as described
> in the help message shown by the command above.
>
> Further, for any input defined in `nasa/algorithm.yml` that has a non-empty
> default value specified, that input can be excluded from the command-line,
> and the default value will be used.  To override the default, you can specify
> a value on the command line by using an option rather than a positional
> argument.  Again, the help message from the command above will describe the
> supported options and positional arguments.

> [!WARNING]
>
> Since the expected command line arguments are obtained from the
> `nasa/algorithm.yml` file, if you override the version of the algorithm to run
> by specifying the `--version` option, the job submission will fail if the
> defined inputs are different between the two versions of the algorithm.

Here's an example submission (which may not match the inputs currently defined
in `nasa/algorithm.yml`, but should give you the idea):

```plain
nasa/job/submit.py https://datapool.asf.alaska.edu/L1.0/A3/ALPSRP271200670-L1.0.zip '-117.722 33.825 -118.61 34.483' alos1 l0b
```

To use the "sandbox" queue instead of the queue specified in the YAML file:

```plain
nasa/job/submit.py --queue maap-dps-sandbox https://datapool.asf.alaska.edu/L1.0/A3/ALPSRP271200670-L1.0.zip '-117.722 33.825 -118.61 34.483' alos1 l0b
```

To override the default GCOV posting:

```plain
nasa/job/submit.py --gcov_posting 0.0009 https://datapool.asf.alaska.edu/L1.0/A3/ALPSRP271200670-L1.0.zip '-117.722 33.825 -118.61 34.483' alos1 l0b
```

Of course, you can override the queue and other defaults simultaneously.

If the job was successfully submitted, the command above will print out the job
ID of the newly submitted job, which you can use to check the job status.

> [!TIP]
>
> You can run the submit script from within a notebook, if you wish to use
> values from the notebook as arguments to the script.
>
> Specifically, rather than making a call like so:
>
> ```plain
> maap.submitJob(
>     identifier="test_to_l0b",
>     algo_id="sar2-d2",
>     version="gcov_pipeline",
>     username="niemoell",
>     queue="maap-dps-sandbox",
>     in_file=item,
>     bbox=union_of_bboxes.get_as_str(),
>     in_type="alos1",  # Choose from: 'alos1', 'l0b', 'rslc',
>     out_type="l0b",  # Choose from: 'l0b', 'rslc', 'gcov'
>     gcov_posting="0.000833334"
> )
> ```
>
> You can make a shell call like so:
>
> ```plain
> !job/submit.py --identifier test_to_l0b --queue maap-dps-sandbox {item} '{union_of_bboxes.get_as_str()}' alos1 l0b
> ```
>
> Notice that there's no need to specify `algo_id`, `version`, `username`, nor
> `gcov_posting` because the script automatically uses the correct values,
> although `version` and `gcov_posting` can be overridden, if needed.
>
> **Also notice the need for quotes around the bbox value!**

## Checking the status of a job

After successfully submitting a job, you can use the job ID to check its status:

```plain
nasa/job/status.py JOB_ID
```

where `JOB_ID` is the job ID printed out by the job submission command.

You can also get more information about the job, using the following command:

```plain
nasa/job/result.py JOB_ID
```

If the job has completed, successfully or not, the command will indicate where
the results were written.  If the job failed, the command should also show an
error message, that may or may not be helpful, and may require looking at the
triaged job files for more information.
