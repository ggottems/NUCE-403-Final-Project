# Running the pebble DEM solver on Aurora

This launcher uses one Aurora compute node and one Python process. It runs the
existing Numba CPU solver on up to 102 physical cores of the node's two Intel Xeon
CPU Max 9470C sockets. It does not use the node's GPUs or split one bed across
nodes. A larger allocation will not accelerate this version of the solver.

## 1. Prepare the files and environment

Use a working directory in your project's Flare space, such as
`/lus/flare/projects/YOUR_PROJECT/YOUR_USERNAME/pebble_dem`; ALCF asks that jobs
be submitted from a project directory rather than from home.
Place `aurora_dem.pbs` and your current `pebble_dem_optimized.py` there.
The solver remains a single Python file; this PBS file only launches it.

On a login node, from that directory:

```bash
module load miniforge3
conda create -y -p "$PWD/.conda-dem-aurora" -c conda-forge \
  python=3.11 numpy scipy numba pillow
```

ALCF's Python documentation is built around `module load frameworks`, a conda
environment with GPU builds of the AI frameworks, extended with virtual
environments. That would change the Python version and the NumPy build, so this
workflow uses the `miniforge3` module (listed by `module avail`) to build the same
plain conda-forge environment as on Frontier. Login nodes reach conda-forge
directly. Compute nodes have no outbound network and need the ALCF proxy
(`http://proxy.alcf.anl.gov:3128`), but nothing in this workflow downloads from a
compute node.

The batch script loads no module and does not activate conda; it runs
`$DEM_ENV/bin/python3` directly. It removes `LD_LIBRARY_PATH` for the solver,
because the default Aurora modules put oneAPI's TBB and GCC's OpenMP and C++
runtimes on it (Numba would otherwise also find a TBB threading layer). No MPI,
GPU stack, FFmpeg or display is required. Pillow is used for snapshots. Run all
benchmarks and simulations on compute nodes through PBS.

PBS does not forward your login environment, so pass a different environment or
solver location with `qsub -v`. For example, if you kept the original solver
filename:

```bash
qsub -A YOUR_PROJECT -v DEM_SCRIPT="$PWD/pebble_dem.py" aurora_dem.pbs
```

## 2. Benchmark one node

```bash
qsub -A YOUR_PROJECT aurora_dem.pbs
```

Replace `YOUR_PROJECT` with your Aurora allocation. Defaults are a 15-minute
`debug` job on one node; the whole sweep takes under two minutes. The script runs
separate processes sequentially for 1, 2, 4, 8, 16, 26, 51, and 102 Numba
threads, always with N=53386, R=21, dp=1 and the same feed as the production run
(8 inlets, release velocity and interval below). This does not generate a packing.

Read `pebble_dem.oJOBID` and the files in `aurora_dem_benchmark_JOBID/`. The job
ends by printing one `ESTIMATE` line per thread count: pour time including
neighbor rebuilds, plus an upper bound for the settling, loading and unloading
stages. Choose the thread count with the lowest total. Repeat close contenders
when results are noisy, for example with `-v "DEM_THREAD_COUNTS=8 16 26 8 16 26"`;
a repeated count gets its own log, such as `threads_16_2.txt`. More threads can be
slower.

Each thread is bound to its own physical core, and the second hardware thread of
each core stays idle. An Aurora node has 2 sockets with 52 cores each, but cores
0 and 52 (hardware threads 104 and 156) are reserved for the operating system and
are outside the job's CPU set, so 102 cores are usable. Threads fill socket 0
(cores 1-51) before socket 1 (cores 53-103): up to 51 threads stay on one socket,
and 102 uses both. OpenMP binds the main Python thread to core 1, so the arrays it
allocates, including every neighbor list, live in socket 0 memory; threads on
socket 1 reach them across the socket link. The on-package HBM is not used.

The estimate integrates the measured step cost over the growing bed. It adds one
full neighbor rebuild (serial, measured) each time the fastest falling pebble
travels half the 0.15 skin, about every 89 steps. It uses a synthetic lattice
and excludes clearance cleanup, snapshots, checkpoints, history sampling and inlet
blockage. A complete 8-inlet test pour at N=2500 took 1.23 times its estimate, so
multiply by about 1.25. `estimated_pour_core_days` is the older force-only figure.

During the run, `estimate_runtime.py` gives a better projection from the
measured history (see section 3).

### Results on Aurora, October 20, 2026

Job 8947910 ran the default sweep. The Frontier column is job 5712746, whose
sweep used 28 and 56 threads where this one uses 26 and 51. Step cost and
neighbor rebuild are measured at N=53386; ESTIMATE is the total from the
`ESTIMATE` line, before the 1.25 correction.

| Threads | Aurora ms/step | Aurora rebuild ms | Aurora ESTIMATE h | Frontier threads | Frontier ms/step | Frontier rebuild ms | Frontier ESTIMATE h |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.37 | 54.5 | 10.06 | 1 | 5.74 | 63.6 | 10.71 |
| 2 | 3.31 | 59.7 | 6.55 | 2 | 5.39 | 68.8 | 10.17 |
| 4 | 2.08 | 60.0 | 4.53 | 4 | 3.59 | 69.0 | 7.12 |
| 8 | 1.56 | 59.7 | 3.78 | 8 | 2.43 | 69.3 | 5.18 |
| 16 | 1.54 | 60.1 | **3.67** | 16 | 2.17 | 80.1 | 4.86 |
| 26 | 1.53 | 61.5 | 3.74 | 28 | 2.08 | 78.5 | **4.74** |
| 51 | 2.09 | 61.1 | 4.54 | 56 | 2.14 | 79.6 | 4.87 |
| 102 | 2.86 | 60.3 | 5.88 | | | | |

Job 8947917 repeated 8-32 threads three times, interleaved, on another node. The
mean totals were 3.75 h (8 threads), 3.64 h (12), 3.62 h (16), 3.70 h (20),
3.82 h (26) and 4.00 h (32).

Use 16 threads; 12 is equivalent within the noise. The estimate is about 3.65 h,
or about 4.6 h with the 1.25 correction, against about 5.9 h for Frontier's best.
As on Frontier, scaling flattens after 8 threads. The serial neighbor rebuild
takes 55-62 ms whatever the thread count, about 28% of the estimated pour time at
16 threads. Beyond about 20 threads the step gets slower even within socket 0,
and using socket 1 makes it slower still (102 threads: 2.86 ms/step, 5.88 h), so
the second socket does not help this solver.

## 3. Generate the packing

With 16 threads, as benchmarked above:

```bash
qsub -A YOUR_PROJECT -q capacity -l walltime=12:00:00 \
  -v MODE=run,DEM_THREADS=16 aurora_dem.pbs
```

`debug` jobs are limited to one hour, and `prod` only routes jobs of 256 nodes or
more. The `capacity` queue takes 1-16 nodes for up to 168 hours, with at most 2
running and 5 queued jobs per user, so it is the queue for this one-node job.
The 12-hour request is about 2.5 times the corrected estimate, so one job should
normally finish the packing; if it does not, the job pauses safely before the
limit (section 4). Queue priority also depends on the project's allocation
balance.

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
| Pause before the walltime | SIGINT 180 s before the limit (`DEM_STOP_MARGIN`) |
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
`-v "DEM_INLETS=x y z; x y z; ..."`. For the original single slow inlet use
`-v DEM_INLET_COUNT=1,DEM_INLET_VZ=0,DEM_POUR_INTERVAL=0.5,...`. Extra solver
flags can be passed with `DEM_EXTRA_ARGS`, for example
`-v "MODE=run,DEM_THREADS=N,DEM_EXTRA_ARGS=--max-settle-time 300"`. `qsub -v`
separates variables with commas, so quote the whole list when a value contains
spaces.

The output prefix is `pb50k_8inlet`; outputs include DAT, CSV, VTK, JSON, a
progress log, a streamed history CSV, snapshot files, and
`pb50k_8inlet_restart.npz`. The releases CSV records the inlet of every pebble.
Final exported geometry is checked by the solver for the specified clearance.
Native soft-contact snapshots can contain the small elastic overlaps intrinsic to
this DEM model.

Change other inputs with `qsub -v` variables such as `DEM_N`, `DEM_R`, `DEM_DP`,
`DEM_PREFIX`, or `DEM_CHECKPOINT`. The default 65 inlet height is specific to the
53386-pebble example; adjust it for other geometries. The launcher refuses a
fresh run if the checkpoint or final DAT already exists.

To project the remaining time while the job runs (after a few thousand releases),
from the working directory on a login node:

```bash
.conda-dem-aurora/bin/python3 estimate_runtime.py pb50k_8inlet_history.csv --job-hours 12
```

It fits the measured wall time per step against bed size, including rebuilds,
snapshots and checkpoints, and integrates it to the final count. `--job-hours`
should match the requested walltime.

## 4. Continue after a pause or walltime limit

```bash
qsub -A YOUR_PROJECT -q capacity -l walltime=12:00:00 \
  -v MODE=resume,DEM_THREADS=16 aurora_dem.pbs
```

Keep the same working directory, prefix, and checkpoint path. This restores
positions, velocities, friction history, random state, and protocol progress;
it does not regenerate the initial particles. The current solver version must
match the version that wrote the checkpoint.

PBS cannot send a signal before the walltime the way Slurm's `--signal=INT@180`
did on Frontier, so the job script does it. At startup it reads the job's
walltime and start time with `qstat -f` and, `DEM_STOP_MARGIN` seconds (default
180) before the limit, sends SIGINT to the solver. The solver uses that signal to
request a checkpoint and safe pause at its next simulation check, every 200
steps, then exits normally, and the job log ends with "Solver stopped before
completion". Set `DEM_STOP_MARGIN=0` to disable the signal. If `qstat` cannot be
read, the job says so and only the periodic checkpoints protect the run.

The solver is started directly on the job's node, not through `mpiexec`. On
Aurora a SIGINT sent to a backgrounded `mpiexec` did not reach Python, while a
directly started solver receives it under the same process ID as recorded by the
job script. The periodic atomic checkpoint remains the fallback if the process is
terminated before it can pause. Final clearance cleanup and export may not
respond immediately to a pause request. Resume is manual; this launcher does not
submit a chain of jobs.

Use checkpoints created by this Aurora wrapper for the simplest resume path.
Moving a checkpoint from Frontier, Windows or another machine can require
additional path overrides for saved output and recording settings.

## Validation and references

Checked on Aurora compute nodes on October 20, 2026, with Python 3.11.17,
NumPy 2.4.6, SciPy 1.17.1, Numba 0.68.0 (llvmlite 0.50.0), Pillow 13.0.0 and
GNU OpenMP (libgomp 16.2) from conda-forge:

- `lscpu` and `numactl -H` show 2 sockets x 52 cores x 2 hardware threads and
  4 NUMA nodes: the DDR of each socket (nodes 0 and 1) and the CPU-less HBM of
  each socket (nodes 2 and 3). The job's CPU set is
  `1-51,53-103,105-155,157-207`.
- With `NUMBA_THREADING_LAYER=omp`, Numba loads its GNU OpenMP layer and the
  environment's own `libgomp`. Each thread is bound to its own core, and the main
  thread to core 1. The solver's `--self-test` passes with 4 threads.
- A 7-minute `MODE=run` debug job with a separate test prefix received SIGINT
  180 s before its walltime, wrote the checkpoint at N=6368 and paused with exit
  status 0. A 6-minute `MODE=resume` job restored N=6368 with its contact and
  random histories, continued to N=8200 and paused the same way. The test files
  were then deleted.
- The benchmarks in section 2, jobs 8947910 and 8947917.

- [Running jobs on Aurora: queues, reserved cores, CPU binding](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/)
- [Getting started on Aurora: proxy and file systems](https://docs.alcf.anl.gov/aurora/getting-started-on-aurora/)
- [Python on Aurora](https://docs.alcf.anl.gov/aurora/data-science/python/)
- [ALCF queue and scheduling policy](https://docs.alcf.anl.gov/policies/queue-scheduling/)

Queue and environment documentation checked October 20, 2026.
