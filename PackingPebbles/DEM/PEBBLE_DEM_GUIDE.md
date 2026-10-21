# Pebble DEM 1.4

`pebble_dem.py` is the only source file needed. The particle count is a parameter. Snapshots, history, and checkpoints work without FFmpeg or another Python source file.

## Install in your activated Python environment

```powershell
python -m pip install numpy scipy numba Pillow
```

For optional MP4 encoding, also install `imageio-ffmpeg`.

## Measure performance on your laptop first

AMD lists the Ryzen 5 7535HS as [six cores and twelve threads](https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-5-7535hs.html). More threads can add overhead, so compare the following settings rather than assuming that six threads is fastest:

```powershell
foreach ($t in 1,2,4,6) {
    python pebble_dem.py --benchmark --N 53386 --R 21 --dp 1 --backend numba --threads $t
}
```

This command benchmarks synthetic contacts; it does not generate a packing. Compilation is excluded from the reported timestep cost. Compare the largest-particle-count timing and the printed core-time estimate. The estimate excludes repeated neighbor rebuilding, settling, plate loading, final clearance cleanup, snapshots, and inlet blockage. It is not a promised completion time.

The NumPy backend remains available, with one contact thread. With Numba and `--threads` greater than one, parallel contact evaluation activates at 2,048 particles. Smaller beds use compiled serial blocks. Threads handle independent contact contributions followed by an ordered per-particle gather, avoiding concurrent force updates on the same particle. The current backend uses the CPU; the RTX GPU is not used.

## Generate 53,386 pebbles with sparse snapshots

The following example uses four threads. Replace `4` with the fastest choice from your benchmark.

```powershell
python pebble_dem.py --N 53386 --R 21 --dp 1 --backend numba --threads 4 --inlet 0 0 65 --load-fraction 0.2 --clearance 1e-5 --snapshots pb50k_snapshots --snapshot-every 1000 --history-interval 5 --checkpoint pb50k_restart.npz --checkpoint-interval 300 -o pb50k.dat
```

Every pebble is released individually at exactly `(0,0,65)`. The small seeded variation is in horizontal velocity, never position. There can be several falling pebbles at once; insertion waits for the release point to clear. `--wait-until-settled` additionally waits for every existing grain and is slower. A centrally poured heap can rise above its mean height, which is why this example raises the inlet from 40 to 65. Omitting `--inlet` chooses one fixed central height automatically (about 64.96 for this case).

The dimensions and clearance use the same length unit. No independent bed height is imposed by this command. The final height and porosity are outputs.

## Several fixed release points

Repeat `--inlet X Y Z` to feed from several fixed points. Each point keeps its own feed interval and blockage timer and releases only when its own point is clear; pebbles released at the same check are inserted together with one neighbor rebuild. Points must be at least 1.0001 diameters apart. The releases CSV has an `inlet` column. A negative `--inlet-vz` pre-accelerates each pebble, so its point clears sooner; with `--inlet-vz -3 --pour-interval 0.35` a point releases about every 0.35 t* instead of 1.52 t*. The Frontier launcher uses 8 points (centre plus 7 on a ring of radius 15 at z=65) with these settings. With a single `--inlet`, results are identical to version 1.3.

## Snapshots and output files

The snapshot directory receives PNG, VTK, and small JSON files at the first release, every 1,000 releases, the last release, stage boundaries, and final clearance verification. A successful run of this example produces about 68 snapshot sets. They are written immediately; no trajectory accumulates in RAM for snapshot-only runs. Expected disk use is on the order of hundreds of MB for this example, depending on actual coordinates and logging.

* PNG: a view of the packing, including the fixed inlet and a translucent loading plate.
* VTK: all centers and sphere radii, for ParaView. Snapshot VTK coordinates use the input length unit and the bottom origin. These settings do not follow a different final-export `--origin` selection.
* JSON: the stage, count, time, and validation status.

Intermediate files ending in `_native` are native soft-contact DEM states and can contain small elastic overlaps. The final `_verified` snapshot and final DAT/CSV/VTK coordinates must pass the strict pair, wall, and floor clearance checks. The final cleanup is geometric and is not another mechanical equilibration.

The output stem also receives `.log`, `_history.csv`, `_releases.csv`, `.json`, `.csv`, `.dat`, and `.vtk`. History is streamed to disk, with sampled rows every `--history-interval` numerical units plus release and convergence events. The `.log` is flushed during execution. Released and settled counts are different: a later impact can move previously settled grains.

## Pause and resume

With `--checkpoint` enabled, press **Ctrl+C once** and wait for the safe-pause message before closing Python. A complete state is saved periodically in wall-clock seconds and after stages. An abrupt interruption can lose progress since the latest saved checkpoint.

```powershell
python pebble_dem.py --resume pb50k_restart.npz
```

Resume restores positions, velocities, rotations, neighbor-list/contact histories, random-generator state, pouring schedule, stage progress, and saved arguments. The log continues, and history rows beyond the checkpoint are removed before continuing. The checkpoint is replaced atomically and only the latest version is retained. Checkpoints are compatible with script version 1.4.0.

You may override output and recording options on the resume command. Maximum settling/holding durations may be increased. Geometry and material/protocol parameters cannot be changed midway through a restart. If moving a checkpoint to a different directory or computer, override saved output, snapshot, video, and log paths as needed; saved paths refer to the original machine.

## Optional bounded video

```powershell
python pebble_dem.py --N 146 --R 3.5 --backend numba --video pouring.mp4 --video-interval 0.2 --video-max-frames 600 -o pb146.dat
```

Recording now has a hard trajectory-frame budget, including release-triggered frames. It thins older frames as needed and preserves stage boundaries. This changes playback sampling, not integration or forces. At 53,386 pebbles and the default 600-frame budget, position arrays are bounded to about 386 MB, plus solver, rendering, and metadata memory. Sparse PNG snapshots are still the simplest choice for a long run. The encoder adds a short final hold without storing more trajectory frames.

## Performance and checks

The 146-pebble reference test retained exactly the same final coordinates and integration-step count after the compiled-update optimization. Its pouring stage dropped from about 20 to 7 seconds in one measurement here. Large-bed speedups are smaller and depend on contact workload, neighbor rebuilding, and CPU threading. A 53k fixed-point pour can still require days; no complete 53k packing has been generated or validated here.

Validation covered NumPy/Numba contact-force, torque, friction-history and plate-reaction agreement; threaded contact agreement; complete small threaded trajectories; exact interrupted/restarted trajectories during pouring, loaded holding, and retraction; restored neighbor topology above the parallel threshold; recording invariance; sparse snapshots; bounded video; and strict final geometric clearance.

```powershell
python pebble_dem.py --self-test --backend numba --threads 4
```

The settings describe an artificial settling protocol with internal `d_p=m=g=1`; the numerical simulation time is not an experimentally calibrated physical packing time. No stiffness, timestep, gravity, friction, or clearance requirement was relaxed to obtain the implementation speedups.
