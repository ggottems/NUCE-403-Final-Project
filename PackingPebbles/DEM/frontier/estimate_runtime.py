#!/usr/bin/env python3
"""Project the remaining wall time of a pebble_dem run from its streamed history CSV.

    python estimate_runtime.py pb50k_frontier_history.csv

Each release row records wall time and the step counter. The cost of one step grows
roughly linearly with the number of pebbles in the bed, so the wall time per release
keeps rising and a straight-line extrapolation of releases/hour is too optimistic.
This script fits step_cost(n) = a + b*n on the measured release intervals (which
include neighbor rebuilds, sampling, snapshots and checkpoints) and integrates it to
the target count. Run it after a few thousand releases above 2048 pebbles.
"""
import argparse
import csv
import math

import numpy as np

PARALLEL_MIN = 2048  # The solver switches to threaded contacts at this count.
POST_POUR_TIME = 120 + 10 + 100 + 10 + 30 + 120  # Settle/ramp/hold/ramp/retract/settle caps, t*.


def hours(seconds):
    return f"{seconds/3600:7.1f} h ({seconds/86400:.2f} d)"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("history", help="<prefix>_history.csv written by pebble_dem")
    p.add_argument("--target", type=int, help="Final pebble count; default from the CSV.")
    p.add_argument("--bin", type=int, default=250, help="Releases per timing bin.")
    p.add_argument("--job-hours", type=float, default=24., help="Wall limit per job, for the job count.")
    a = p.parse_args()

    n, wall, steps, target, dt = [], [], [], None, None
    with open(a.history, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["stage"] != "pouring" or row["event"] != "release":
                continue
            n.append(int(float(row["released_count"])))
            wall.append(float(row["wall_elapsed_seconds"]))
            steps.append(int(float(row["total_steps"])))
            target, dt = int(float(row["target_count"])), float(row["timestep"])
    if not n:
        raise SystemExit("No pouring release rows found; is this a _history.csv from a pour run?")
    target = a.target or target
    n, wall, steps = np.array(n), np.array(wall), np.array(steps)

    edges = np.arange(0, len(n), a.bin)
    mid, cost, per_release = [], [], []
    for i, j in zip(edges[:-1], edges[1:]):
        dw, ds, dn = wall[j]-wall[i], steps[j]-steps[i], n[j]-n[i]
        if dw > 0 and ds > 0 and dn > 0:  # Skip bins spanning a restart glitch.
            mid.append(.5*(n[i]+n[j])); cost.append(dw/ds); per_release.append(ds/dn)
    mid, cost, per_release = map(np.array, (mid, cost, per_release))
    use = mid >= PARALLEL_MIN
    if use.sum() < 3:
        print(f"Warning: fewer than 3 timing bins above {PARALLEL_MIN} pebbles; "
              "the fit mixes serial and threaded regimes and is rough.")
        use = np.ones_like(mid, bool)
    if use.sum() < 2:
        raise SystemExit(f"Need at least {2*a.bin} releases to fit; have {len(n)}.")
    b, a0 = np.polyfit(mid[use], cost[use], 1)
    b = max(b, 0.)
    if b == 0.:
        a0 = float(np.mean(cost[use]))
    s_rel = float(np.median(per_release[use][-10:]))

    now = int(n[-1])
    c_now, c_end = a0+b*now, a0+b*target
    pour_left = s_rel*(a0*(target-now)+.5*b*(target**2-now**2))
    post = POST_POUR_TIME/dt*c_end
    total = wall[-1]+pour_left+post

    print(f"Released {now}/{target} pebbles after {hours(wall[-1])} of wall time.")
    print(f"Steps per release (recent median): {s_rel:.0f}   timestep dt* = {dt:.4g}")
    print(f"Fitted step cost: {1e3*a0:.3f} ms + {1e6*b:.4f} us x N"
          f"  -> {1e3*c_now:.3f} ms/step now, {1e3*c_end:.3f} ms/step at N={target}")
    print(f"Remaining pour:            {hours(pour_left)}")
    print(f"Post-pour stages (bound):  {hours(post)}  (settling caps; usually converges sooner)")
    print(f"Projected total wall time: {hours(total)}  excluding final clearance cleanup")
    print(f"About {math.ceil(total/(a.job_hours*3600-300))} job(s) of {a.job_hours:g} h in total.")


if __name__ == "__main__":
    main()
