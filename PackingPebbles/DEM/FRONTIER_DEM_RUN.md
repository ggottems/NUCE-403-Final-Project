# Running the pebble DEM solver on Frontier

This launcher uses one compute node and one Python process. It runs the existing
Numba CPU solver, using up to 56 physical cores with Frontier's default low-noise
core reservation. It does not accelerate contacts on AMD GPUs or split one bed
across nodes. A larger allocation will not accelerate this version of the solver.

## 1. Prepare the files and environment

Use a working directory in your allocated Orion project scratch area, such as
`/lustre/orion/YOUR_PROJECT/scratch/YOUR_USERNAME/pebble_dem`.
Place `frontier_dem.sbatch` and your current `pebble_dem_optimized.py` there.
The solver remains a single Python file; this Slurm file only launches it.

On a login node, from that directory:

```bash
module load miniforge3
conda create -y -p "$PWD/.conda-dem-frontier" -c conda-forge \
  python=3.11 numpy scipy numba pillow
```

The batch script loads the module and activates that exact environment for each
job. No MPI Python package, ROCm environment, FFmpeg, or graphics display is
required. Pillow is used for snapshots. Run all benchmarks and simulations on
compute nodes through Slurm.

For a different environment or solver location, export `DEM_ENV` or `DEM_SCRIPT`
before submitting. For example, if you kept the original solver filename:

```bash
export DEM_SCRIPT="$PWD/pebble_dem.py"
```

## 2. Benchmark one node

```bash
sbatch -A YOUR_PROJECT frontier_dem.sbatch
```

Replace `YOUR_PROJECT` with your current authorized allocation. Defaults are a
45-minute `batch` job with one node, one task, and 56 physical cores. The script
runs separate processes sequentially for 1, 2, 4, 8, 16, 28, and 56 Numba threads,
always with N=53386, R=21, dp=1 and the same feed as the production run (8 inlets,
release velocity and interval below). This does not generate a packing.

Read `pebble_dem-JOBID.out` and the files in
`frontier_dem_benchmark_JOBID/`. The job ends by printing one `ESTIMATE` line per
thread count: pour time including neighbor rebuilds, plus an upper bound for the
settling, loading and unloading stages. Choose the thread count with the lowest
total. Repeat close contenders when results are noisy. More threads can be slower.

The estimate integrates the measured step cost over the growing bed. It adds one
full neighbor rebuild (serial, measured) each time the fastest falling pebble
travels half the 0.15 skin, about every 80-90 steps. It uses a synthetic lattice
and excludes clearance cleanup, snapshots, checkpoints, history sampling and inlet
blockage. A complete 8-inlet test pour at N=2500 took 1.23 times its estimate, so
multiply by about 1.25. `estimated_pour_core_days` is the older force-only figure.

During the run, `estimate_runtime.py` gives a better projection from the
measured history (see section 3).

## 3. Generate the packing

If 16 threads performs best, for example:

```bash
sbatch -A YOUR_PROJECT -p extended -t 24:00:00 \
  --export=ALL,MODE=run,DEM_THREADS=16 frontier_dem.sbatch
```

Frontier's documented `batch` limit for 1-91 nodes is two hours. The `extended`
partition permits up to 24 hours for jobs of at most 64 nodes. Queue eligibility
depends on your allocation. The 24-hour request is an allocation limit, not a
prediction of completion time. This file deliberately has no debug QOS directive.

Defaults for the production run:

| Setting | Value |
| --- | --- |
| Pebbles | 53386 |
| Cylinder radius / pebble diameter | 21 / 1 |
| Fixed inlets | 8: (0, 0, 65) plus 7 evenly spaced on a ring of radius 15 at z=65 |
| Release velocity | vz = -3 (terminal speed is 1/drag = 3.33) |
| Minimum interval per inlet | 0.35 t* (about 1400 steps) |
| Top load | 20% of total bed weight |
| Final surface clearance | 1e-5 in input length units |
| Snapshots | Every 1000 releases and at stage boundaries |
| Checkpoint | Every 300 wall-clock seconds and at stage boundaries |
| Video | Disabled |

Each inlet feeds independently and waits until its own point is clear; pebbles
released at the same check are inserted together with one neighbor rebuild. The
pre-accelerated release lets a point clear in about 0.33 t* instead of 1.52 t*,
and impact speed is almost unchanged because pebbles falling from z=65 approach
terminal speed anyway. Together this is about 35 times fewer pouring steps than
one inlet released from rest (about 9.3e6 instead of 3.3e8). The bed is built
from 8 merging cones rather than one central cone, and each stream delivers about
one pebble per 0.35 t*. Compare packing fraction and radial porosity against the
single-inlet protocol on a smaller bed if that difference matters.

The ring radius 15 makes the centre point and the 7 ring points each serve about
one eighth of the cross-section. Change the layout with `DEM_INLET_COUNT`,
`DEM_INLET_RADIUS` and `DEM_INLET_Z`, or give exact points with
`DEM_INLETS="x y z; x y z; ..."`. For the original single slow inlet use
`DEM_INLET_COUNT=1,DEM_INLET_VZ=0,DEM_POUR_INTERVAL=0.5`. Extra solver flags
can be passed with `DEM_EXTRA_ARGS`, for example `"--max-settle-time 300"`.

The output prefix is `pb50k_8inlet`; outputs include DAT, CSV, VTK, JSON, a
progress log, a streamed history CSV, snapshot files, and
`pb50k_8inlet_restart.npz`. The releases CSV records the inlet of every pebble.
Final exported geometry is checked by the solver for the specified clearance.
Native soft-contact snapshots can contain the small elastic overlaps intrinsic to
this DEM model.

Change other inputs with exported variables such as `DEM_N`, `DEM_R`, `DEM_DP`,
`DEM_PREFIX`, or `DEM_CHECKPOINT`. The default 65 inlet height is specific to the
53386-pebble example; adjust it for other geometries. The launcher refuses a
fresh run if the checkpoint or final DAT already exists.

To project the remaining time while the job runs (after a few thousand releases),
from the working directory:

```bash
.conda-dem-frontier/bin/python3 estimate_runtime.py pb50k_8inlet_history.csv
```

It fits the measured wall time per step against bed size, including rebuilds,
snapshots and checkpoints, and integrates it to the final count.

## 4. Continue after a pause or wall-time limit

```bash
sbatch -A YOUR_PROJECT -p extended -t 24:00:00 \
  --export=ALL,MODE=resume,DEM_THREADS=16 frontier_dem.sbatch
```

Keep the same working directory, prefix, and checkpoint path. This restores
positions, velocities, friction history, random state, and protocol progress;
it does not regenerate the initial particles. The current solver version must
match the version that wrote the checkpoint.

Slurm is asked to send SIGINT to the running job step about three minutes before
the allocation expires. The solver uses that signal to request a checkpoint and
safe pause at its next simulation check. The periodic atomic checkpoint remains
the fallback if the process is terminated before it can pause. Final clearance
cleanup and export may not respond immediately to a pause request. Resume is
manual; this launcher does not submit a chain of jobs.

Use checkpoints created by this Frontier wrapper for the simplest resume path.
Moving a checkpoint from Windows or another machine can require additional path
overrides for saved output and recording settings.

## Validation and references

The launcher was checked with Bash and a mocked Slurm/module environment using
the real solver for benchmark, run, and resume. No Frontier job was submitted
from this session; the site environment, CPU affinity, signal delivery, and
Frontier performance must be verified on the machine.

- [Frontier user guide: hardware, Slurm, mapping, and partitions](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
- [Python environments on OLCF systems](https://docs.olcf.ornl.gov/software/python/index.html)
- [Slurm sbatch signal options](https://slurm.schedmd.com/sbatch.html#OPT_signal)

Queue and environment documentation checked October 19, 2026.
