#!/usr/bin/env python3
"""Standalone contact-force DEM packing of equal spheres in a cylinder.

Required packages: numpy scipy. Optional: numba (speed), matplotlib (plots),
Pillow and imageio-ffmpeg (MP4 animation, or use a system FFmpeg).
No other source files or fixed particle counts. No compiler is needed.

Examples (all lengths in the same unit):
  python pebble_dem.py --N 146 --R 3.5 --dp 1 --load-fraction .2 --clearance 1e-5 -o pb146.dat
  python pebble_dem.py --N 1568 --R 6.525 --dp 1 -o pb1568.dat
  python pebble_dem.py --diameter .06 --core-radius .3915 --height .72 -o filled.dat
  python pebble_dem.py --N 146 --R 3.5 --video packing.mp4 -o pb146.dat
  python pebble_dem.py --N 146 --R 3.5 --inlet 0 0 8 --video pour.mp4 -o pb146.dat
  python pebble_dem.py --N 53386 --R 21 --backend numba --threads 6 --snapshots snapshots --snapshot-every 1000 --checkpoint pb50k_restart.npz -o pb50k.dat
  python pebble_dem.py --resume pb50k_restart.npz
  python pebble_dem.py --benchmark --N 53386 --R 21 --backend numba --threads 6

Give --N for an exact count, --height for overfill/settle/cut filling, or
both for an exact count with a maximum allowed height. Height is not an
independent prescribed outcome when N, diameter, and radius are fixed.

Default --insertion pour releases one sphere at a time from one fixed point.
--inlet X Y Z sets that point in the input length unit. Repeat --inlet for
several fixed points; each feeds independently, and spheres released at the
same check are inserted together. The automatic point is central and its
height is chosen once. Each insertion waits until its point is clear. --pour-interval is a minimum interval, not a guaranteed rate.
By default other grains may still be moving; --wait-until-settled waits for
all existing grains before each release. Small seeded horizontal release
velocity variation (--inlet-jitter) breaks ideal axial symmetry; release
POSITIONS remain identical. Use --insertion cloud for the former batch mode.

A flushed <output-stem>.log reports released/target, settled, moving and
retained counts, cumulative and stage numerical times, wall runtime, steps,
timestep, height, force residual, load and overlaps. Settled means continuous
speed/spin/force/torque tolerances for quiet_time; impacts can unsettle grains.
The history CSV includes these fields; the releases CSV records every exact
insertion position, velocity and time. --plate-opacity changes only the video.

Large runs: --snapshots DIR writes PNG/VTK/JSON milestones directly to disk,
at --snapshot-every released pebbles and stage boundaries. No full trajectory
is retained for snapshots. Snapshot VTK uses input length units and bottom
origin. Native snapshots have elastic contact overlap; only the final verified
snapshot and final DAT/CSV/VTK are cleared. Pillow is sufficient for snapshots.
CLI history is streamed and decimated with --history-interval (default 1).

--checkpoint FILE saves full particle, friction, RNG and protocol state every
--checkpoint-interval wall seconds (default 300) and at stage boundaries.
Ctrl+C requests a safe pause when checkpoints are enabled. --resume FILE
restores the saved arguments; output/recording flags can be overridden, as can
the maximum settling/holding times. Pause before shutting down the computer.
Checkpoint files are compatible with this script version only.

Numba advances small blocks in compiled code and reuses work arrays. --threads
enables a race-free pair/gather contact kernel for N>=2048 (default 1 thread).
Benchmark 1/2/4/6 on your own CPU; more threads need not be faster. The current
backend uses CPU cores, not the GPU. --benchmark uses synthetic contacts and
prints a rough core-time extrapolation; it does not simulate a completed bed.

Video remains optional. --video-max-frames (default 600) bounds recording RAM
through online thinning, including frames forced by releases. Playback sampling
therefore becomes coarser in long runs. At N=53386, stored coordinates are
bounded to about 386 MB at this default, plus solver/rendering/metadata memory.

The DEM uses Hooke normal spring/dashpot contacts, Coulomb-limited tangential
history, sphere rotation, a fixed floor/wall, and an optional smooth weighted
plate. It gravity-settles, ramps/holds/unloads the plate, retracts it, and
settles again. --load-fraction 0 disables the plate.

Internal units are d_p=m=g=1. Stiffness, damping, and time parameters describe
an artificial settling protocol, not experimentally calibrated material/time.
Floating-point/backend differences can lead to different settled arrangements.
Native soft-contact states have small elastic overlaps. The final geometric
cleanup moves centers without changing diameters and is not another mechanical
equilibration. Final exported coordinates must strictly satisfy distance >=
diameter + clearance, plus the same wall/floor clearance (and top if supplied).

--clearance uses the input length unit; its default is 1e-5 times diameter.
DAT is three-column x y z unless --format serpent is selected; CSV and VTK
include radius. Output files share the output stem. JSON includes actual N,
parameters, convergence, geometry checks, and the output coordinate transform.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

__version__ = "1.4.0"

def norm(a):
    return np.linalg.norm(a, axis=-1)

def tangential_force(history, normal, velocity, fn, kn, gamma, mu, dt):
    """Return force ON i, and updated tangential spring displacement.

    normal points toward i from its partner/wall; velocity is i's contact
    velocity relative to that partner/wall. Inputs only contain active contacts.
    """
    old_norm = norm(history)
    xi = history - np.sum(history * normal, axis=1)[:, None] * normal
    projected_norm = norm(xi)
    scale = np.divide(old_norm, projected_norm, out=np.zeros_like(old_norm),
                      where=projected_norm > 1e-30)
    xi *= scale[:, None]
    vt = velocity - np.sum(velocity * normal, axis=1)[:, None] * normal
    xi += dt * vt
    kt = 2.0 * kn / 7.0
    gamma_t = 0.5 * gamma
    ft = -kt * xi - gamma_t * vt
    magnitude = norm(ft)
    cap = mu * fn
    sliding = magnitude > cap
    factor = np.divide(cap[sliding], magnitude[sliding])
    ft[sliding] *= factor[:, None]
    # Reconcile spring history with the capped total force, including damping.
    xi[sliding] = -(ft[sliding] + gamma_t * vt[sliding]) / kt
    return ft, xi

def random_cloud(count, cylinder_radius, rng):
    """Nonoverlapping dilute seed; hash cells avoid all-pairs insertion."""
    accessible_radius = cylinder_radius - 0.5
    cloud_height = max(3.0, count / (6 * max(accessible_radius**2, 0.25) * 0.20))
    cells = {}
    positions = []
    offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    attempts = 0
    while len(positions) < count:
        attempts += 1
        if attempts > max(10000, 300 * count):
            raise RuntimeError("Could not seed the dilute cloud; use a wider cylinder.")
        rho = accessible_radius * math.sqrt(rng.random())
        theta = 2 * math.pi * rng.random()
        p = np.array([rho * math.cos(theta), rho * math.sin(theta),
                      0.51 + rng.random() * cloud_height])
        cell = tuple(np.floor(p).astype(int))
        clear = True
        for offset in offsets:
            neighbor = tuple(cell[k] + offset[k] for k in range(3))
            for j in cells.get(neighbor, ()):
                if np.dot(p - positions[j], p - positions[j]) < 1.01**2:
                    clear = False
                    break
            if not clear:
                break
        if clear:
            cells.setdefault(cell, []).append(len(positions))
            positions.append(p)
    return np.asarray(positions)

class DEM:
    """Vectorized contact forces with a displacement-triggered Verlet list."""

    def __init__(self, positions, R, kn=20000., mu=0.4, mu_wall=0.4,
                 restitution=0.35, drag=0.3, dt_scale=1.0):
        self.x = np.array(positions, dtype=float, copy=True)
        self.v = np.zeros_like(self.x)
        self.w = np.zeros_like(self.x)
        self.R = R
        self.kn = kn
        self.mu = mu
        self.mu_wall = mu_wall
        self.drag = drag
        self.dt = 0.08 * math.sqrt(0.5 / kn) * dt_scale
        # This is the dashpot parameterization from the unclipped linear
        # oscillator. Clipping tensile forces slightly changes measured e.
        log_e = math.log(restitution)
        zeta = -log_e / math.sqrt(math.pi**2 + log_e**2)
        self.gamma_pair = 2 * zeta * math.sqrt(0.5 * kn)
        self.gamma_wall = 2 * zeta * math.sqrt(kn)
        self.skin = 0.15
        self.pairs = np.empty((0, 2), dtype=np.int64)
        self.keys = np.empty(0, dtype=np.int64)
        self.xi = np.empty((0, 3))
        self.wall_xi = np.zeros((len(self.x), 2, 3))
        self.reference = None
        self.elapsed = 0.0
        self.total_steps = 0
        self.target_count = len(self.x)
        self.released_total = len(self.x)
        self.quiet_elapsed = np.zeros(len(self.x))
        self.last_quiet_sample = 0.
        self.settled_count = 0
        self.inlet_position = None
        self.rebuild()

    def rebuild(self):
        n = len(self.x)
        pairs = cKDTree(self.x).query_pairs(1 + self.skin, output_type="ndarray")
        if len(pairs):
            pairs.sort(axis=1)
            keys = pairs[:, 0] * n + pairs[:, 1]
            order = np.argsort(keys)
            pairs, keys = pairs[order], keys[order]
        else:
            keys = np.empty(0, dtype=np.int64)
        xi = np.zeros((len(pairs), 3))
        if len(keys) and len(self.keys):
            loc = np.searchsorted(self.keys, keys)
            bounded = np.minimum(loc, len(self.keys) - 1)
            matched = (loc < len(self.keys)) & (self.keys[bounded] == keys)
            xi[matched] = self.xi[loc[matched]]
        self.pairs, self.keys, self.xi = pairs, keys, xi
        self.reference = self.x.copy()

    def forces(self, history_dt):
        if np.max(norm(self.x - self.reference), initial=0) >= self.skin / 2:
            self.rebuild()
        f = np.zeros_like(self.x)
        f[:, 2] = -1.0
        torque = np.zeros_like(self.x)
        max_overlap = 0.0
        if len(self.pairs):
            i, j = self.pairs.T
            delta_x = self.x[i] - self.x[j]
            distance = norm(delta_x)
            if np.any(distance < 1e-10):
                raise RuntimeError("Coincident particle centers: integration became unstable.")
            active = distance < 1.0
            self.xi[~active] = 0.0
            ids = np.flatnonzero(active)
            if len(ids):
                i, j = i[ids], j[ids]
                d = distance[ids]
                overlap = 1.0 - d
                normal = delta_x[ids] / d[:, None]
                lever = 0.5 * d  # midpoint of the overlap region
                relative = (self.v[i] - self.v[j]
                            - lever[:, None] * np.cross(self.w[i] + self.w[j], normal))
                vn = np.sum(relative * normal, axis=1)
                fn = np.maximum(0., self.kn * overlap - self.gamma_pair * vn)
                ft, xi = tangential_force(self.xi[ids], normal, relative, fn,
                                          self.kn, self.gamma_pair, self.mu, history_dt)
                self.xi[ids] = xi
                contact = fn[:, None] * normal + ft
                np.add.at(f, i, contact)
                np.add.at(f, j, -contact)
                moment = -lever[:, None] * np.cross(normal, ft)
                np.add.at(torque, i, moment)
                np.add.at(torque, j, moment)
                max_overlap = float(np.max(overlap))

        rho = norm(self.x[:, :2])
        for wall in (0, 1):
            overlap = (0.5 - self.x[:, 2] if wall == 0
                       else rho + 0.5 - self.R)
            active = overlap > 0
            self.wall_xi[~active, wall] = 0.
            ids = np.flatnonzero(active)
            if not len(ids):
                continue
            normal = np.zeros((len(ids), 3))
            if wall == 0:
                normal[:, 2] = 1.
            else:
                normal[:, :2] = -self.x[ids, :2] / rho[ids, None]
            lever = 0.5 - 0.5 * overlap[ids]
            relative = self.v[ids] - lever[:, None] * np.cross(self.w[ids], normal)
            vn = np.sum(relative * normal, axis=1)
            fn = np.maximum(0., self.kn * overlap[ids] - self.gamma_wall * vn)
            ft, xi = tangential_force(self.wall_xi[ids, wall], normal, relative,
                                      fn, self.kn, self.gamma_wall,
                                      self.mu_wall, history_dt)
            self.wall_xi[ids, wall] = xi
            f[ids] += fn[:, None] * normal + ft
            torque[ids] -= lever[:, None] * np.cross(normal, ft)
            max_overlap = max(max_overlap, float(np.max(overlap[ids])))
        return f, torque, max_overlap


    def retain(self, keep):
        """Keep state and wall history, remapping particle contact histories."""
        mapping = np.full(len(self.x), -1, dtype=np.int64)
        mapping[keep] = np.arange(np.count_nonzero(keep))
        pair_keep = keep[self.pairs[:, 0]] & keep[self.pairs[:, 1]]
        self.pairs = mapping[self.pairs[pair_keep]]
        self.xi = self.xi[pair_keep]
        self.x, self.v, self.w = self.x[keep], self.v[keep], self.w[keep]
        self.wall_xi = self.wall_xi[keep]
        self.quiet_elapsed = self.quiet_elapsed[keep]
        self.settled_count = 0
        self.keys = self.pairs[:, 0] * len(self.x) + self.pairs[:, 1]
        self.rebuild()

    def append_particle(self, position, velocity):
        """Grow all particle arrays and preserve old tangential histories."""
        position, velocity = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("A release needs finite 3D position and velocity vectors.")
        self.append_particles(position[None, :], velocity[None, :])

    def append_particles(self, positions, velocities):
        """Insert a batch of spheres with a single neighbor-list rebuild."""
        positions, velocities = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
        if (positions.ndim != 2 or positions.shape[1] != 3 or velocities.shape != positions.shape
                or not np.all(np.isfinite([positions, velocities]))):
            raise ValueError("A release needs finite 3D position and velocity vectors.")
        count = len(positions)
        self.x = np.vstack((self.x, positions))
        self.v = np.vstack((self.v, velocities))
        self.w = np.vstack((self.w, np.zeros((count, 3))))
        self.wall_xi = np.concatenate((self.wall_xi, np.zeros((count, 2, 3))), axis=0)
        self.quiet_elapsed = np.append(self.quiet_elapsed, np.zeros(count))
        # Pair keys encode i*N+j. Re-encode old pairs for the new N before
        # rebuilding, so insertion never discards existing friction histories.
        self.keys = self.pairs[:, 0]*len(self.x)+self.pairs[:, 1]
        self.released_total += count
        self.rebuild()



def _tangent_values(hx, hy, hz, nx, ny, nz, vx, vy, vz, fn, kn, gamma, mu, dt):
    previous = math.sqrt(hx*hx+hy*hy+hz*hz)
    dot = hx*nx+hy*ny+hz*nz
    ax, ay, az = hx-dot*nx, hy-dot*ny, hz-dot*nz
    projected = math.sqrt(ax*ax+ay*ay+az*az)
    scale = previous/projected if projected > 1e-30 else 0.
    vn = vx*nx+vy*ny+vz*nz
    tx, ty, tz = vx-vn*nx, vy-vn*ny, vz-vn*nz
    ax, ay, az = ax*scale+dt*tx, ay*scale+dt*ty, az*scale+dt*tz
    kt, gt = 2.*kn/7., .5*gamma
    fx, fy, fz = -kt*ax-gt*tx, -kt*ay-gt*ty, -kt*az-gt*tz
    magnitude = math.sqrt(fx*fx+fy*fy+fz*fz)
    cap = mu*fn
    if magnitude > cap:
        factor = cap/magnitude
        fx, fy, fz = fx*factor, fy*factor, fz*factor
        ax, ay, az = -(fx+gt*tx)/kt, -(fy+gt*ty)/kt, -(fz+gt*tz)/kt
    return fx, fy, fz, ax, ay, az


def _force_into(x, v, w, pairs, xi, wall_xi, R, kn, mu, mu_wall,
                gamma_pair, gamma_wall, dt, plate, plate_z, plate_v, f, torque):
    """Same force law as the vectorized reference; optionally JIT compiled."""
    count = len(x)
    maximum, reaction, plate_overlap, plate_contacts = 0., 0., 0., 0
    for i in range(count):
        f[i, 0], f[i, 1] = 0., 0.
        f[i, 2] = -1.
        torque[i, 0], torque[i, 1], torque[i, 2] = 0., 0., 0.
    for p in range(len(pairs)):
        i, j = pairs[p, 0], pairs[p, 1]
        dx, dy, dz = x[i, 0]-x[j, 0], x[i, 1]-x[j, 1], x[i, 2]-x[j, 2]
        distance = math.sqrt(dx*dx+dy*dy+dz*dz)
        if distance >= 1.:
            xi[p, 0], xi[p, 1], xi[p, 2] = 0., 0., 0.
            continue
        if distance < 1e-12:
            maximum = 1.
            continue
        nx, ny, nz = dx/distance, dy/distance, dz/distance
        lever, overlap = .5*distance, 1.-distance
        wx, wy, wz = w[i, 0]+w[j, 0], w[i, 1]+w[j, 1], w[i, 2]+w[j, 2]
        rx = v[i, 0]-v[j, 0]-lever*(wy*nz-wz*ny)
        ry = v[i, 1]-v[j, 1]-lever*(wz*nx-wx*nz)
        rz = v[i, 2]-v[j, 2]-lever*(wx*ny-wy*nx)
        fn = max(0., kn*overlap-gamma_pair*(rx*nx+ry*ny+rz*nz))
        tx, ty, tz, hx, hy, hz = _tangent_values(xi[p,0],xi[p,1],xi[p,2], nx, ny, nz, rx, ry, rz,
                                                fn, kn, gamma_pair, mu, dt)
        xi[p,0],xi[p,1],xi[p,2]=hx,hy,hz
        fx, fy, fz = nx*fn+tx, ny*fn+ty, nz*fn+tz
        f[i, 0] += fx; f[i, 1] += fy; f[i, 2] += fz
        f[j, 0] -= fx; f[j, 1] -= fy; f[j, 2] -= fz
        mx, my, mz = -lever*(ny*tz-nz*ty), -lever*(nz*tx-nx*tz), -lever*(nx*ty-ny*tx)
        torque[i, 0] += mx; torque[i, 1] += my; torque[i, 2] += mz
        torque[j, 0] += mx; torque[j, 1] += my; torque[j, 2] += mz
        maximum = max(maximum, overlap)
    for i in range(count):
        rho = math.sqrt(x[i, 0]*x[i, 0]+x[i, 1]*x[i, 1])
        for wall in range(2):
            overlap = .5-x[i, 2] if wall == 0 else rho+.5-R
            if overlap <= 0.:
                wall_xi[i, wall, 0] = 0.; wall_xi[i, wall, 1] = 0.; wall_xi[i, wall, 2] = 0.
                continue
            if wall == 0:
                nx, ny, nz = 0., 0., 1.
            else:
                nx, ny, nz = -x[i, 0]/rho, -x[i, 1]/rho, 0.
            lever = .5-.5*overlap
            rx = v[i, 0]-lever*(w[i, 1]*nz-w[i, 2]*ny)
            ry = v[i, 1]-lever*(w[i, 2]*nx-w[i, 0]*nz)
            rz = v[i, 2]-lever*(w[i, 0]*ny-w[i, 1]*nx)
            fn = max(0., kn*overlap-gamma_wall*(rx*nx+ry*ny+rz*nz))
            tx, ty, tz, hx, hy, hz = _tangent_values(wall_xi[i,wall,0],wall_xi[i,wall,1],wall_xi[i,wall,2],
                                                    nx, ny, nz, rx, ry, rz,fn, kn, gamma_wall, mu_wall, dt)
            wall_xi[i,wall,0],wall_xi[i,wall,1],wall_xi[i,wall,2]=hx,hy,hz
            f[i, 0] += nx*fn+tx; f[i, 1] += ny*fn+ty; f[i, 2] += nz*fn+tz
            torque[i, 0] -= lever*(ny*tz-nz*ty)
            torque[i, 1] -= lever*(nz*tx-nx*tz)
            torque[i, 2] -= lever*(nx*ty-ny*tx)
            maximum = max(maximum, overlap)
        if plate:
            overlap = x[i, 2]+.5-plate_z
            if overlap > 0.:
                fn = max(0., kn*overlap-gamma_wall*(plate_v-v[i, 2]))
                f[i, 2] -= fn
                reaction += fn; plate_contacts += 1
                plate_overlap = max(plate_overlap, overlap)
                maximum = max(maximum, overlap)
    return maximum, reaction, plate_contacts, plate_overlap


def _force_loop(x, v, w, pairs, xi, wall_xi, R, kn, mu, mu_wall,
                gamma_pair, gamma_wall, dt, plate, plate_z, plate_v):
    f, torque = np.empty_like(x), np.empty_like(x)
    stats = _force_into(x, v, w, pairs, xi, wall_xi, R, kn, mu, mu_wall,
                        gamma_pair, gamma_wall, dt, plate, plate_z, plate_v, f, torque)
    return f, torque, stats[0], stats[1], stats[2], stats[3]


def _needs_rebuild(x, reference, limit):
    for i in range(len(x)):
        dx, dy, dz = x[i, 0]-reference[i, 0], x[i, 1]-reference[i, 1], x[i, 2]-reference[i, 2]
        if dx*dx+dy*dy+dz*dz >= limit:
            return True
    return False


def _advance_block(x, v, w, pairs, xi, wall_xi, reference, limit,
                   R, kn, mu, mu_wall, gamma_pair, gamma_wall, dt, damping,
                   plate_mode, plate_z, plate_v, plate_mass, plate_drag, plate_speed,
                   start_load, end_load, duration, stage_step, steps, elapsed, f, torque):
    """Advance without Python calls or per-step array allocation. Stop before a rebuild."""
    reaction, contacts, plate_overlap, maximum, load = 0., 0, 0., 0., start_load
    completed = 0
    for k in range(steps):
        if _needs_rebuild(x, reference, limit):
            break
        if plate_mode == 2:
            plate_v = plate_speed
        if plate_mode:
            t = (stage_step+k)*dt
            ramp = .5-.5*math.cos(math.pi*min(1., t/duration))
            load = start_load+(end_load-start_load)*ramp
        maximum, reaction, contacts, plate_overlap = _force_into(
            x, v, w, pairs, xi, wall_xi, R, kn, mu, mu_wall,
            gamma_pair, gamma_wall, dt, plate_mode != 0, plate_z, plate_v, f, torque)
        for i in range(len(x)):
            for axis in range(3):
                v[i, axis] = (v[i, axis]+dt*f[i, axis])*damping
                w[i, axis] = (w[i, axis]+(10.*dt)*torque[i, axis])*damping
                x[i, axis] += dt*v[i, axis]
        if plate_mode == 1:
            candidate = (plate_v+dt*(reaction-load)/plate_mass)*math.exp(-plate_drag*dt)
            plate_v = min(plate_speed, max(-plate_speed, candidate))
            plate_z += dt*plate_v
        elif plate_mode == 2:
            plate_z += dt*plate_v
        elapsed += dt
        completed += 1
    return completed, elapsed, plate_z, plate_v, reaction, contacts, plate_overlap, maximum, load


_NUMBA_KERNEL = None
_ADVANCE_KERNEL = None
_REBUILD_KERNEL = None
_PARALLEL_FORCE = None
_PUSH_KERNEL = None
prange = range


def _pair_contributions(x,v,w,pairs,xi,kn,gamma,mu,dt,out):
    for p in prange(len(pairs)):
        for axis in range(7): out[p,axis]=0.
        i,j=pairs[p,0],pairs[p,1]
        dx,dy,dz=x[i,0]-x[j,0],x[i,1]-x[j,1],x[i,2]-x[j,2]
        distance=math.sqrt(dx*dx+dy*dy+dz*dz)
        if distance>=1.:
            xi[p,0],xi[p,1],xi[p,2]=0.,0.,0.
            continue
        if distance<1e-12:
            out[p,6]=1.
            continue
        nx,ny,nz=dx/distance,dy/distance,dz/distance
        lever,overlap=.5*distance,1.-distance
        wx,wy,wz=w[i,0]+w[j,0],w[i,1]+w[j,1],w[i,2]+w[j,2]
        rx=v[i,0]-v[j,0]-lever*(wy*nz-wz*ny)
        ry=v[i,1]-v[j,1]-lever*(wz*nx-wx*nz)
        rz=v[i,2]-v[j,2]-lever*(wx*ny-wy*nx)
        fn=max(0.,kn*overlap-gamma*(rx*nx+ry*ny+rz*nz))
        tx,ty,tz,hx,hy,hz=_tangent_values(xi[p,0],xi[p,1],xi[p,2],nx,ny,nz,rx,ry,rz,fn,kn,gamma,mu,dt)
        xi[p,0],xi[p,1],xi[p,2]=hx,hy,hz
        out[p,0],out[p,1],out[p,2]=nx*fn+tx,ny*fn+ty,nz*fn+tz
        out[p,3],out[p,4],out[p,5]=-lever*(ny*tz-nz*ty),-lever*(nz*tx-nx*tz),-lever*(nx*ty-ny*tx)
        out[p,6]=overlap


def _particle_contributions(x,v,w,wall_xi,ptr,indices,signs,pair_values,R,kn,gamma,mu,dt,
                            plate,plate_z,plate_v,f,torque,stats):
    for i in prange(len(x)):
        f[i,0],f[i,1],f[i,2]=0.,0.,-1.
        torque[i,0],torque[i,1],torque[i,2]=0.,0.,0.
        for entry in range(ptr[i],ptr[i+1]):
            p,sign=indices[entry],signs[entry]
            for axis in range(3):
                f[i,axis]+=sign*pair_values[p,axis]
                torque[i,axis]+=pair_values[p,axis+3]
        stats[i,0],stats[i,1],stats[i,2]=0.,0.,0.
        rho=math.sqrt(x[i,0]*x[i,0]+x[i,1]*x[i,1])
        for wall in range(2):
            overlap=.5-x[i,2] if wall==0 else rho+.5-R
            if overlap<=0.:
                wall_xi[i,wall,0],wall_xi[i,wall,1],wall_xi[i,wall,2]=0.,0.,0.
                continue
            if wall==0: nx,ny,nz=0.,0.,1.
            else: nx,ny,nz=-x[i,0]/rho,-x[i,1]/rho,0.
            lever=.5-.5*overlap
            rx=v[i,0]-lever*(w[i,1]*nz-w[i,2]*ny)
            ry=v[i,1]-lever*(w[i,2]*nx-w[i,0]*nz)
            rz=v[i,2]-lever*(w[i,0]*ny-w[i,1]*nx)
            fn=max(0.,kn*overlap-gamma*(rx*nx+ry*ny+rz*nz))
            tx,ty,tz,hx,hy,hz=_tangent_values(wall_xi[i,wall,0],wall_xi[i,wall,1],wall_xi[i,wall,2],
                                              nx,ny,nz,rx,ry,rz,fn,kn,gamma,mu,dt)
            wall_xi[i,wall,0],wall_xi[i,wall,1],wall_xi[i,wall,2]=hx,hy,hz
            f[i,0]+=nx*fn+tx;f[i,1]+=ny*fn+ty;f[i,2]+=nz*fn+tz
            torque[i,0]-=lever*(ny*tz-nz*ty)
            torque[i,1]-=lever*(nz*tx-nx*tz)
            torque[i,2]-=lever*(nx*ty-ny*tx)
            stats[i,0]=max(stats[i,0],overlap)
        if plate:
            overlap=x[i,2]+.5-plate_z
            if overlap>0.:
                fn=max(0.,kn*overlap-gamma*(plate_v-v[i,2]))
                f[i,2]-=fn;stats[i,1]=overlap;stats[i,2]=fn


def _parallel_force_into(x,v,w,pairs,xi,wall_xi,R,kn,mu,mu_wall,gamma_pair,gamma_wall,dt,
                          plate,plate_z,plate_v,f,torque,ptr,indices,signs,pair_values,stats):
    _pair_contributions(x,v,w,pairs,xi,kn,gamma_pair,mu,dt,pair_values)
    _particle_contributions(x,v,w,wall_xi,ptr,indices,signs,pair_values,R,kn,gamma_wall,mu_wall,dt,
                            plate,plate_z,plate_v,f,torque,stats)
    maximum,reaction,contacts,plate_overlap=0.,0.,0,0.
    for p in range(len(pairs)): maximum=max(maximum,pair_values[p,6])
    for i in range(len(x)):
        maximum=max(maximum,stats[i,0],stats[i,1])
        plate_overlap=max(plate_overlap,stats[i,1])
        reaction+=stats[i,2]
        if stats[i,1]>0.:contacts+=1
    return maximum,reaction,contacts,plate_overlap


def _push_particles(x,v,w,f,torque,dt,damping):
    for i in range(len(x)):
        for axis in range(3):
            v[i,axis]=(v[i,axis]+dt*f[i,axis])*damping
            w[i,axis]=(w[i,axis]+(10.*dt)*torque[i,axis])*damping
            x[i,axis]+=dt*v[i,axis]


class PackingDEM(DEM):
    """Optional Numba acceleration and a smooth, vertically translating plate."""
    def __init__(self, positions, R, backend="auto", threads=1, **kwargs):
        if int(threads)!=threads or threads<1:
            raise ValueError("threads must be a positive integer.")
        if threads>1 and backend=="numpy":
            raise ValueError("Multiple contact threads require --backend numba.")
        self.parallel,self.parallel_min_particles,self.threads=False,2048,int(threads)
        super().__init__(positions, R, **kwargs)
        self.backend, self.kernel = "numpy", None
        self.plate_enabled = False
        self.plate_z, self.plate_v, self.plate_reaction = 1e6, 0., 0.
        self.plate_contacts, self.plate_overlap = 0, 0.
        if backend != "numpy":
            try:
                from numba import njit,set_num_threads
                global _NUMBA_KERNEL, _ADVANCE_KERNEL, _REBUILD_KERNEL, _tangent_values, _force_into, _needs_rebuild
                global _PARALLEL_FORCE,_PUSH_KERNEL,prange,_pair_contributions,_particle_contributions
                set_num_threads(self.threads)
                if _NUMBA_KERNEL is None:
                    _tangent_values = njit(cache=False,inline="always")(_tangent_values)
                    _force_into = njit(cache=False)(_force_into)
                    _needs_rebuild = njit(cache=False)(_needs_rebuild)
                    _NUMBA_KERNEL = njit(cache=False)(_force_loop)
                    _ADVANCE_KERNEL = njit(cache=False)(_advance_block)
                    _REBUILD_KERNEL = _needs_rebuild
                    _PUSH_KERNEL=njit(cache=False)(_push_particles)
                if threads>1 and _PARALLEL_FORCE is None:
                    from numba import prange
                    _pair_contributions=njit(cache=False,parallel=True)(_pair_contributions)
                    _particle_contributions=njit(cache=False,parallel=True)(_particle_contributions)
                    _PARALLEL_FORCE=njit(cache=False)(_parallel_force_into)
                self.kernel = _NUMBA_KERNEL
                self.parallel=threads>1
                if self.parallel:self.rebuild()
                self.forces(0.)  # Compile before the first timed stage.
                self.backend = "numba"
            except Exception as error:
                self.kernel = None
                if backend == "numba" or threads>1:
                    raise RuntimeError(f"Numba backend unavailable: {error}. Install with: python -m pip install numba") from error

    def forces(self, history_dt):
        if self.kernel is not None:
            if _REBUILD_KERNEL(self.x, self.reference, self.skin**2/4):
                self.rebuild()
            if self.parallel and len(self.x)>=self.parallel_min_particles:
                self._buffers()
                stats=_PARALLEL_FORCE(self.x,self.v,self.w,self.pairs,self.xi,self.wall_xi,self.R,self.kn,self.mu,self.mu_wall,
                    self.gamma_pair,self.gamma_wall,history_dt,self.plate_enabled,self.plate_z,self.plate_v,
                    self.force_buffer,self.torque_buffer,self.adj_ptr,self.adj_indices,self.adj_signs,self.pair_values,self.particle_stats)
                overlap,self.plate_reaction,self.plate_contacts,self.plate_overlap=stats
                return self.force_buffer,self.torque_buffer,overlap
            result = self.kernel(self.x, self.v, self.w, self.pairs, self.xi,
                                 self.wall_xi, self.R, self.kn, self.mu, self.mu_wall,
                                 self.gamma_pair, self.gamma_wall, history_dt,
                                 self.plate_enabled, self.plate_z, self.plate_v)
            f, torque, overlap, self.plate_reaction, self.plate_contacts, self.plate_overlap = result
            return f, torque, overlap
        f, torque, overlap = super().forces(history_dt)
        self.plate_reaction, self.plate_contacts, self.plate_overlap = 0., 0, 0.
        if self.plate_enabled:
            delta = self.x[:, 2]+.5-self.plate_z
            ids = np.flatnonzero(delta > 0.)
            if len(ids):
                fn = np.maximum(0., self.kn*delta[ids]-self.gamma_wall*(self.plate_v-self.v[ids, 2]))
                f[ids, 2] -= fn
                self.plate_reaction = float(fn.sum())
                self.plate_contacts = len(ids)
                self.plate_overlap = float(delta[ids].max())
                overlap = max(overlap, self.plate_overlap)
        return f, torque, overlap

    def _buffers(self):
        if not hasattr(self,"force_buffer") or self.force_buffer.shape!=self.x.shape:
            self.force_buffer,self.torque_buffer=np.empty_like(self.x),np.empty_like(self.x)

    def rebuild(self):
        super().rebuild()
        if self.parallel and len(self.x)>=self.parallel_min_particles:
            self._build_adjacency()

    def _build_adjacency(self):
        endpoints=self.pairs.ravel()
        order=np.argsort(endpoints,kind="stable")
        self.adj_ptr=np.r_[0,np.cumsum(np.bincount(endpoints,minlength=len(self.x)))].astype(np.int64)
        self.adj_indices=(order//2).astype(np.int64)
        self.adj_signs=np.where(order%2==0,1.,-1.)
        self.pair_values=np.empty((len(self.pairs),7))
        self.particle_stats=np.empty((len(self.x),3))

    def advance(self, steps, settings, *, plate_mode="off", start_load=0., end_load=0.,
                duration=1., stage_step=0):
        mode = {"off": 0, "dynamic": 1, "retract": 2}[plate_mode]
        damping = math.exp(-self.drag*self.dt)
        load = start_load
        if self.kernel is not None and self.parallel and len(self.x)>=self.parallel_min_particles:
            for k in range(steps):
                if mode:
                    ramp=.5-.5*math.cos(math.pi*min(1.,(stage_step+k)*self.dt/duration))
                    load=start_load+(end_load-start_load)*ramp
                if mode==2:self.plate_v=settings["plate_speed"]
                force,torque,_=self.forces(self.dt)
                _PUSH_KERNEL(self.x,self.v,self.w,force,torque,self.dt,damping)
                if mode==1:
                    candidate=(self.plate_v+self.dt*(self.plate_reaction-load)/settings["plate_mass"])*math.exp(-settings["plate_drag"]*self.dt)
                    self.plate_v=min(settings["plate_speed"],max(-settings["plate_speed"],candidate))
                    self.plate_z+=self.dt*self.plate_v
                elif mode==2:self.plate_z+=self.dt*self.plate_v
                self.elapsed+=self.dt;self.total_steps+=1
        elif self.kernel is not None:
            if not hasattr(self, "force_buffer") or self.force_buffer.shape != self.x.shape:
                self.force_buffer, self.torque_buffer = np.empty_like(self.x), np.empty_like(self.x)
            done = 0
            while done < steps:
                result = _ADVANCE_KERNEL(
                    self.x, self.v, self.w, self.pairs, self.xi, self.wall_xi,
                    self.reference, self.skin**2/4, self.R, self.kn, self.mu, self.mu_wall,
                    self.gamma_pair, self.gamma_wall, self.dt, damping, mode,
                    self.plate_z, self.plate_v, settings["plate_mass"], settings["plate_drag"],
                    settings["plate_speed"], start_load, end_load, duration,
                    stage_step+done, steps-done, self.elapsed, self.force_buffer, self.torque_buffer)
                n, self.elapsed, self.plate_z, self.plate_v, self.plate_reaction, self.plate_contacts, self.plate_overlap, _, load = result
                done += n
                if done < steps:
                    self.rebuild()
            self.total_steps += done
        else:
            for k in range(steps):
                if mode:
                    ramp = .5-.5*math.cos(math.pi*min(1., (stage_step+k)*self.dt/duration))
                    load = start_load+(end_load-start_load)*ramp
                if mode == 2: self.plate_v = settings["plate_speed"]
                force, torque, _ = self.forces(self.dt)
                self.v = (self.v+self.dt*force)*damping
                self.w = (self.w+10.*self.dt*torque)*damping
                self.x += self.dt*self.v
                if mode == 1:
                    self.plate_v = float(np.clip(
                        (self.plate_v+self.dt*(self.plate_reaction-load)/settings["plate_mass"])
                        *math.exp(-settings["plate_drag"]*self.dt),
                        -settings["plate_speed"], settings["plate_speed"]))
                    self.plate_z += self.dt*self.plate_v
                elif mode == 2:
                    self.plate_z += self.dt*self.plate_v
                self.elapsed += self.dt; self.total_steps += 1
        return load


def interior_porosity(x, R, trim=1.):
    bottom = float(x[:, 2].min()-.5)
    lo, hi = bottom+trim, float(x[:, 2].max()+.5)-trim
    if hi <= lo:
        return None
    a, b = np.maximum(-.5, lo-x[:, 2]), np.minimum(.5, hi-x[:, 2])
    m = b > a
    solid = math.pi*np.sum(.25*(b[m]-a[m])-(b[m]**3-a[m]**3)/3)
    return float(1.-solid/(math.pi*R*R*(hi-lo)))


class PackingPaused(Exception):
    pass


class ProgressLog:
    """A flushed text log that survives an interrupted run."""
    def __init__(self, path, echo=True, append=False, elapsed_before=0.):
        self.path, self.echo = Path(path), echo
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a" if append else "w", encoding="utf-8")
        self.started = time.perf_counter()-elapsed_before

    def __enter__(self):
        self.write(f"pebble_dem {__version__} | started {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
        self.write("Simulation time t* is in sqrt(d_p/g); wall time is elapsed clock time.")
        return self

    def write(self, message):
        if self.echo:
            print(message, flush=True)
        self.file.write(str(message)+"\n"); self.file.flush()

    def __exit__(self, kind, error, traceback):
        if error is not None and not isinstance(error, PackingPaused):
            self.file.write(f"ERROR: {error}\n"); self.file.flush()
        self.file.close()


class HistoryStream:
    """Write sparse history immediately; keep no history rows in memory."""
    def __init__(self, path, interval=1., append=False):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("History interval must be positive and finite.")
        self.path, self.interval = Path(path), interval
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.needs_header = not (append and self.path.exists() and self.path.stat().st_size)
        self.file = self.path.open("a" if append else "w", newline="", encoding="utf-8")
        self.writer, self.next_time, self.stage = None, 0., None

    def append(self, row):
        force = (row["event"] == "release" or row["stage"] != self.stage
                 or row["quiet_time"] >= row["quiet_time_required"] or row["stage_progress"] == 1.)
        if row["time"] < self.next_time and not force:
            return
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            if self.needs_header: self.writer.writeheader()
        self.writer.writerow(row); self.file.flush()
        self.next_time, self.stage = row["time"]+self.interval, row["stage"]

    def __bool__(self):
        return False  # Already written; write_final must not rewrite the file.

    def __enter__(self): return self

    def __exit__(self, *args): self.file.close()


def checkpoint_metadata(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
    if meta.get("checkpoint_schema") != 1 or meta.get("script_version") != __version__:
        raise ValueError("Unsupported checkpoint format or script version.")
    return meta


def trim_history_for_resume(path, elapsed, total_steps):
    """Discard rows written after the last durable checkpoint, after an abrupt stop."""
    path=Path(path)
    if not path.exists():return
    with tempfile.TemporaryDirectory(prefix=".resume_history_",dir=path.parent) as directory:
        staged=Path(directory)/"history.csv"
        with path.open(newline="",encoding="utf-8") as source,staged.open("w",newline="",encoding="utf-8") as target:
            reader=csv.DictReader(source)
            if reader.fieldnames is None:return
            writer=csv.DictWriter(target,fieldnames=reader.fieldnames);writer.writeheader()
            for row in reader:
                if not row.get("time") or not row.get("total_steps"):break
                if float(row["time"])>elapsed or int(row["total_steps"])>total_steps:break
                writer.writerow(row)
        staged.replace(path)


class CheckpointManager:
    """Atomic full-state restarts, including tangential history and RNG state."""
    array_names = ("x", "v", "w", "pairs", "keys", "xi", "wall_xi", "reference", "quiet_elapsed")
    scalar_names = ("R", "kn", "mu", "mu_wall", "drag", "dt", "gamma_pair", "gamma_wall", "skin",
                    "elapsed", "total_steps", "target_count", "released_total", "last_quiet_sample", "settled_count",
                    "plate_enabled", "plate_z", "plate_v", "plate_reaction", "plate_contacts", "plate_overlap")

    def __init__(self, path, interval=300., *, resume=None, command=None, logger=None, recorder=None, history=None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Checkpoint interval must be positive wall-clock seconds.")
        self.path, self.interval, self.resume_path = Path(path), interval, resume
        self.command, self.logger, self.recorder, self.history = command or {}, logger, recorder, history
        self.last_save = time.perf_counter()
        self.context, self.loop, self.stop_requested = {}, None, False

    def periodic(self, sim, loop):
        self.loop = loop
        if self.stop_requested or time.perf_counter()-self.last_save >= self.interval:
            self.save(sim, loop)
        if self.stop_requested:
            message=f"Paused safely. Resume with: python pebble_dem.py --resume {self.path}"
            if self.logger:self.logger.write(message)
            raise PackingPaused(message)

    def boundary(self, sim):
        self.loop = None
        self.save(sim, None)

    def save(self, sim, loop):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        context = {key:value for key,value in self.context.items() if key != "rng"}
        context["rng_state"] = self.context["rng"].bit_generator.state
        arrays = {name:getattr(sim,name) for name in self.array_names}
        meta = dict(checkpoint_schema=1, script_version=__version__, command=self.command,
                    context=context, loop=loop, scalars={name:getattr(sim,name) for name in self.scalar_names},
                    inlet=None if sim.inlet_position is None else sim.inlet_position.tolist(),
                    elapsed_wall_seconds=(time.perf_counter()-self.logger.started) if self.logger else 0.)
        if isinstance(self.history, HistoryStream):
            meta["history_stream"] = dict(next_time=self.history.next_time, stage=self.history.stage)
        if self.recorder is not None:
            video = self.recorder.video if isinstance(self.recorder, RecordingBundle) else self.recorder
            if isinstance(video, PackingRecorder):
                frames = []
                for i, frame in enumerate(video.frames):
                    arrays[f"video_{i}"] = frame["positions"]
                    item = {k:v for k,v in frame.items() if k != "positions"}
                    if item.get("inlet") is not None: item["inlet"] = np.asarray(item["inlet"]).tolist()
                    frames.append(item)
                meta["video"] = dict(frames=frames, next_time=video.next_time, raw_count=video.raw_count,
                                     stride=video.stride, max_frames=video.max_frames, interval=video.interval)
            snapshots = self.recorder.snapshots if isinstance(self.recorder, RecordingBundle) else None
            if snapshots is not None:
                first = snapshots.first_item
                if first is not None:
                    arrays["snapshot_first"] = first["positions"]
                    item = {k:v for k,v in first.items() if k != "positions"}
                    if item.get("inlet") is not None: item["inlet"] = np.asarray(item["inlet"]).tolist()
                else: item = None
                meta["snapshots"] = dict(next_count=snapshots.next_count, first=item)
        arrays["metadata"] = np.array(json.dumps(meta, default=lambda v:v.item() if isinstance(v,np.generic) else str(v), allow_nan=False))
        with tempfile.TemporaryDirectory(prefix=".dem_checkpoint_", dir=self.path.parent) as directory:
            staged = Path(directory)/"state.npz"
            np.savez_compressed(staged, **arrays)
            staged.replace(self.path)
        self.last_save = time.perf_counter()
        if self.logger:
            phase = loop["name"] if loop is not None else "stage boundary"
            self.logger.write(f"[checkpoint] {self.path} | phase={phase} N={len(sim.x)} t*={sim.elapsed:.2f}")

    def restore(self, backend, threads=1):
        meta = checkpoint_metadata(self.resume_path)
        with np.load(self.resume_path, allow_pickle=False) as data:
            s = meta["scalars"]
            sim = PackingDEM(data["x"], s["R"], backend=backend, threads=threads,kn=s["kn"], mu=s["mu"], mu_wall=s["mu_wall"], drag=s["drag"])
            for name in self.array_names: setattr(sim, name, data[name].copy())
            for name,value in s.items(): setattr(sim,name,value)
            if sim.parallel and len(sim.x)>=sim.parallel_min_particles:sim._build_adjacency()
            sim.inlet_position = None if meta["inlet"] is None else np.array(meta["inlet"],float)
            if self.recorder is not None:
                video = self.recorder.video if isinstance(self.recorder,RecordingBundle) else self.recorder
                if isinstance(video,PackingRecorder) and "video" in meta:
                    item = meta["video"]
                    video.frames = [dict(frame, positions=data[f"video_{i}"].copy()) for i,frame in enumerate(item["frames"])]
                    for field in ("next_time", "raw_count", "stride", "interval"): setattr(video,field,item[field])
                    while len(video.frames) > video.max_frames: video._thin()
                snapshots = self.recorder.snapshots if isinstance(self.recorder,RecordingBundle) else None
                if snapshots is not None and "snapshots" in meta:
                    item=meta["snapshots"]; snapshots.next_count=item["next_count"]
                    if item["first"] is not None:
                        snapshots.first_item=dict(item["first"],positions=data["snapshot_first"].copy())
        if isinstance(self.history,HistoryStream) and "history_stream" in meta:
            for field,value in meta["history_stream"].items(): setattr(self.history,field,value)
        self.context,self.loop=meta["context"],meta["loop"]
        return sim,meta


def announce(settings, message):
    if settings.get("logger") is not None:
        settings["logger"].write(message)
    elif settings["verbose"]:
        print(message, flush=True)


def sample_metrics(sim, stage, stage_time, load, settings, force, torque, overlap, event="sample"):
    speed, spin = norm(sim.v), .5*norm(sim.w)
    forces, moments = norm(force), 2.*norm(torque)
    stable = (np.maximum(speed, spin) < settings["speed_tol"]) & (np.maximum(forces, moments) < settings["force_tol"])
    elapsed = max(0., sim.elapsed-sim.last_quiet_sample)
    sim.quiet_elapsed = np.where(stable, sim.quiet_elapsed+elapsed, 0.)
    sim.last_quiet_sample = sim.elapsed
    sim.settled_count = int(np.count_nonzero(sim.quiet_elapsed >= settings["quiet_time"]))
    relative = abs(sim.plate_reaction-load)/max(load, 1.) if sim.plate_enabled else 0.
    return dict(time=sim.elapsed, stage=stage, stage_time=stage_time, event=event,
                total_steps=sim.total_steps, timestep=sim.dt,
                wall_elapsed_seconds=time.perf_counter()-settings["started"],
                particle_count=len(sim.x), released_count=sim.released_total,
                target_count=sim.target_count, release_progress=sim.released_total/max(1, sim.target_count),
                settled_count=sim.settled_count, moving_count=len(sim.x)-sim.settled_count,
                height=float(sim.x[:, 2].max()+.5), interior_porosity=interior_porosity(sim.x, sim.R),
                target_load=load, reaction_load=sim.plate_reaction, load_fraction=load/len(sim.x),
                plate_height=sim.plate_z if sim.plate_enabled else None,
                plate_velocity=sim.plate_v if sim.plate_enabled else 0.,
                max_speed=float(speed.max()), max_surface_spin_speed=float(spin.max()),
                force_residual=float(max(forces.max(), moments.max())),
                plate_relative_residual=relative, max_overlap=overlap,
                quiet_time=0., quiet_time_required=settings["quiet_time"],
                stage_progress=None, feed_status="not_feeding")


def log_progress(settings, row):
    wall = row["wall_elapsed_seconds"]
    clock = f"{int(wall)//3600:02d}:{int(wall)//60%60:02d}:{int(wall)%60:02d}"
    progress = ""
    if row["stage_progress"] is not None:
        progress = f" stage={100*row['stage_progress']:.1f}%"
    if row["quiet_time"]:
        progress += f" quiet={row['quiet_time']:.2f}/{row['quiet_time_required']:.2f}"
    feed = f" feed={row['feed_status']}" if row["feed_status"] != "not_feeding" else ""
    announce(settings,
             f"[{row['stage']}] released={row['released_count']}/{row['target_count']} "
             f"({100*row['release_progress']:.1f}%) settled={row['settled_count']} "
             f"moving={row['moving_count']} retained={row['particle_count']} | "
             f"t_total*={row['time']:.2f} stage_t*={row['stage_time']:.2f} "
             f"wall={clock} step={row['total_steps']} dt*={row['timestep']:.3g} | "
             f"H/d={row['height']:.4f} vmax*={row['max_speed']:.2e} "
             f"force_res={row['force_residual']:.2e} load/W={row['load_fraction']:.3f} "
             f"overlap/d={row['max_overlap']:.2e}{progress}{feed}")


def block_size(sim, step, remaining, recorder):
    count = min(200-step % 200, remaining)
    if recorder is not None and math.isfinite(recorder.next_time):
        until = max(1, math.ceil((recorder.next_time-sim.elapsed)/sim.dt))
        count = min(count, until)
    return count


def run_stage(sim, name, duration, settings, history, *, start_load=0., end_load=None,
              settle=False, plate_mode="off"):
    if end_load is None:
        end_load = start_load
    sim.plate_enabled = plate_mode != "off"
    recorder = settings.get("recorder")
    checkpoint = settings.get("checkpoint")
    resumed = checkpoint.loop if checkpoint is not None and checkpoint.loop and checkpoint.loop["name"] == name else None
    if recorder is not None and resumed is None:
        recorder.capture(sim, name, start_load/len(sim.x), force=True)
    steps = math.ceil(duration/sim.dt)
    quiet, last_check, next_report = 0., 0, 0.
    started = time.perf_counter()
    row = None
    step = 0
    past_wall = 0.
    if resumed is not None:
        step, quiet, last_check, next_report, row, past_wall = (resumed[k] for k in ("step", "quiet", "last_check", "next_report", "row", "wall_seconds"))
    while step < steps:
        count = block_size(sim, step, steps-step, recorder)
        load = sim.advance(count, settings, plate_mode=plate_mode, start_load=start_load,
                           end_load=end_load, duration=duration, stage_step=step)
        step += count
        t = (step-1)*sim.dt
        if recorder is not None and sim.elapsed >= recorder.next_time:
            recorder.capture(sim, name, load/len(sim.x))
        if step % 200 and step != steps:
            continue
        force, torque, overlap = sim.forces(0.)
        if not np.all(np.isfinite(sim.x)) or overlap > .04:
            raise RuntimeError(f"Unstable DEM stage {name}; decrease --dt-scale (overlap/d={overlap:g}).")
        row = sample_metrics(sim, name, step*sim.dt, load, settings, force, torque, overlap)
        small = max(row["max_speed"], row["max_surface_spin_speed"]) < settings["speed_tol"] and row["force_residual"] < settings["force_tol"]
        if sim.plate_enabled:
            small = small and abs(sim.plate_v) < 1e-4 and row["plate_relative_residual"] < .001
        quiet = quiet+(step-last_check)*sim.dt if small else 0.
        last_check = step
        row["quiet_time"] = quiet
        row["stage_progress"] = min(1., step*sim.dt/duration) if not settle and plate_mode != "retract" else None
        history.append(row)
        done = (settle and quiet >= settings["quiet_time"]) or (plate_mode == "retract" and sim.plate_z > row["height"]+.1)
        if t >= next_report or step == steps or done:
            log_progress(settings, row)
            next_report = t+settings["log_interval"]
        if checkpoint is not None and not done:
            checkpoint.periodic(sim, dict(name=name, step=step, quiet=quiet, last_check=last_check,
                                          next_report=next_report, row=row, wall_seconds=past_wall+time.perf_counter()-started))
        if settle and quiet >= settings["quiet_time"]:
            break
        if plate_mode == "retract" and sim.plate_z > row["height"]+.1:
            break
    result = dict(stage=name, converged=(quiet >= settings["quiet_time"]) if settle else None,
                  simulated_duration=step*sim.dt, wall_seconds=past_wall+time.perf_counter()-started,
                  final=row)
    if recorder is not None:
        recorder.capture(sim, name, load/len(sim.x), force=True)
    if settle and not result["converged"]:
        raise RuntimeError(f"DEM stage {name} did not converge in {duration:g} time units. "
                           "Increase --max-settle-time / --max-hold-time; no final bed was exported.")
    if plate_mode == "retract" and sim.plate_z <= row["height"]+.1:
        raise RuntimeError("The plate did not retract clear of the bed.")
    return result


def pour_particles(sim, target_count, position, rng, interval, jitter, velocity_z,
                   wait_until_settled, duration, inlet_timeout, settings, history, releases=None):
    """Release individual spheres at one or several exact fixed points, each only when clear.

    position is one (3,) point or a (K,3) array; every point keeps its own feed
    interval and blockage timer, and pebbles released together share one rebuild.
    """
    recorder = settings.get("recorder")
    position = np.asarray(position, dtype=float)
    sim.inlet_position = position.copy()
    inlets = np.atleast_2d(position)
    sim.target_count = target_count
    checkpoint = settings.get("checkpoint")
    resumed = checkpoint.loop if checkpoint is not None and checkpoint.loop and checkpoint.loop["name"] == "pouring" else None
    releases = [] if releases is None else releases
    next_release, blocked_since = [sim.elapsed+interval]*len(inlets), [None]*len(inlets)
    start_time, started, next_report = sim.elapsed, time.perf_counter(), 0.
    steps = math.ceil(duration/sim.dt)
    insertion_distance = 1.0001
    step = 0
    past_wall = 0.
    if resumed is not None:
        step, next_release, blocked_since, start_time, next_report, row, past_wall = (resumed[k] for k in
            ("step", "next_release", "blocked_since", "start_time", "next_report", "row", "wall_seconds"))
    else:
        for k in range(len(sim.x)):
            releases.append(dict(index=k+1,inlet=k+1,time=sim.elapsed,x_over_dp=float(inlets[k,0]),y_over_dp=float(inlets[k,1]),
                                 z_over_dp=float(inlets[k,2]),vx_over_sqrt_gdp=float(sim.v[k,0]),
                                 vy_over_sqrt_gdp=float(sim.v[k,1]),vz_over_sqrt_gdp=float(sim.v[k,2]),minimum_insertion_distance=None))
        if recorder is not None: recorder.capture(sim,"pouring",force=True)
        force,torque,overlap=sim.forces(0.)
        row=sample_metrics(sim,"pouring",0.,0.,settings,force,torque,overlap,"release")
        row["feed_status"]="released_particle";row["stage_progress"]=len(sim.x)/target_count
        history.append(row);log_progress(settings,row)
    while step < steps and len(sim.x) < target_count:
        count = block_size(sim, step, steps-step, recorder)
        sim.advance(count, settings)
        step += count
        if recorder is not None and sim.elapsed >= recorder.next_time:
            recorder.capture(sim, "pouring")
        if step % 200 and step != steps:
            continue
        force, torque, overlap = sim.forces(0.)
        if not np.all(np.isfinite(sim.x)) or overlap > .04:
            raise RuntimeError("Unstable pouring integration; decrease --dt-scale.")
        row = sample_metrics(sim, "pouring", sim.elapsed-start_time, 0., settings, force, torque, overlap)
        nearest = [float(norm(sim.x-point).min()) for point in inlets]
        clear = [distance >= insertion_distance for distance in nearest]
        due = [sim.elapsed >= ready_time for ready_time in next_release]
        ready = not wait_until_settled or sim.settled_count == len(sim.x)
        chosen = [k for k in range(len(inlets)) if due[k] and clear[k] and ready][:target_count-len(sim.x)]
        for k in range(len(inlets)):
            if k in chosen:
                continue
            if due[k] and not clear[k]:
                if blocked_since[k] is None: blocked_since[k] = sim.elapsed
                if sim.elapsed-blocked_since[k] > inlet_timeout:
                    which = "The fixed inlet" if len(inlets) == 1 else f"Fixed inlet {k+1} {(inlets[k]).tolist()} (diameters)"
                    raise RuntimeError(f"{which} stayed blocked for {inlet_timeout:g} time units after "
                                       f"{len(sim.x)}/{target_count} releases. Increase --inlet Z; its position is never moved automatically.")
            else:
                blocked_since[k] = None
        released = bool(chosen)
        if released:
            velocities = np.array([[*rng.normal(0., jitter, 2), velocity_z] for _ in chosen])
            settled_before, first = sim.settled_count, len(sim.x)
            sim.append_particles(inlets[chosen], velocities)
            for offset, (k, velocity) in enumerate(zip(chosen, velocities)):
                releases.append(dict(index=first+offset+1, inlet=k+1, time=sim.elapsed,
                                     x_over_dp=float(inlets[k,0]), y_over_dp=float(inlets[k,1]), z_over_dp=float(inlets[k,2]),
                                     vx_over_sqrt_gdp=float(velocity[0]), vy_over_sqrt_gdp=float(velocity[1]),
                                     vz_over_sqrt_gdp=float(velocity[2]), minimum_insertion_distance=nearest[k]))
                next_release[k], blocked_since[k] = sim.elapsed+interval, None
            # New spheres are contact-free, so old forces are unchanged.
            row = sample_metrics(sim, "pouring", sim.elapsed-start_time, 0., settings,
                                 np.vstack((force, np.tile([0., 0., -1.], (len(chosen), 1)))),
                                 np.vstack((torque, np.zeros((len(chosen), 3)))), overlap, "release")
            row["settled_count"] = settled_before
            row["feed_status"] = "released_particle"
            if recorder is not None:
                recorder.capture(sim, "pouring", force=True)
        else:
            row["feed_status"] = ("waiting_for_inlet_clearance" if any(d and not c for d, c in zip(due, clear)) else
                                  "waiting_for_settling" if any(due) else "waiting_for_interval")
        row["stage_progress"] = len(sim.x)/target_count
        history.append(row)
        if released or row["stage_time"] >= next_report or step == steps:
            log_progress(settings, row)
            next_report = row["stage_time"]+settings["log_interval"]
        if checkpoint is not None and len(sim.x) < target_count:
            checkpoint.periodic(sim,dict(name="pouring",step=step,next_release=list(next_release),blocked_since=list(blocked_since),
                                        start_time=start_time,next_report=next_report,row=row,
                                        wall_seconds=past_wall+time.perf_counter()-started))
    if len(sim.x) != target_count:
        raise RuntimeError(f"Only {len(sim.x)}/{target_count} particles were released before --max-pour-time.")
    result = dict(stage="pouring", converged=None, simulated_duration=sim.elapsed-start_time,
                  wall_seconds=past_wall+time.perf_counter()-started, final=row)
    return result, releases


def enforce_clearance(initial, R, clearance, height=None):
    """Geometry correction with arrays sized from the actual particle count."""
    initial = np.asarray(initial, dtype=float)
    if initial.ndim != 2 or initial.shape[1] != 3 or not len(initial):
        raise ValueError("Expected a nonempty (N,3) array of centers.")
    count = len(initial)
    if not np.all(np.isfinite(initial)):
        raise ValueError("Non-finite particle coordinates.")
    if count > 1 and cKDTree(initial).query(initial, k=2)[0][:, 1].min() < 1e-12:
        raise ValueError("Coincident centers cannot be corrected; check the input coordinates.")
    guard = max(1e-7, 512*np.finfo(float).eps*max(1., R, np.abs(initial).max()))
    target = clearance+guard
    if R <= .5+target or (height is not None and height <= 1.+2*target):
        raise ValueError("The cylinder is too small for the diameter and requested boundary clearance.")
    x, log = initial.copy(), []
    started = time.perf_counter()
    for attempt in range(6):
        pairs = cKDTree(x).query_pairs(1.2+target, output_type="ndarray")
        i, j = pairs.T

        def objective(flat):
            q = flat.reshape(count, 3)
            delta = q[i]-q[j]
            distance = norm(delta)
            deficit = np.maximum(1.+target-distance, 0.)
            pair_gradient = -deficit[:, None]*delta/np.maximum(distance[:, None], 1e-15)
            gradient = np.zeros_like(q)
            for axis in range(3):
                gradient[:, axis] = (np.bincount(i, weights=pair_gradient[:, axis], minlength=count)
                                     - np.bincount(j, weights=pair_gradient[:, axis], minlength=count))
            rho = norm(q[:, :2])
            side = np.maximum(rho+.5+target-R, 0.)
            floor = np.maximum(.5+target-q[:, 2], 0.)
            gradient[:, :2] += side[:, None]*q[:, :2]/np.maximum(rho[:, None], 1e-15)
            gradient[:, 2] -= floor
            value = .5*(deficit@deficit+side@side+floor@floor)
            if height is not None:
                top = np.maximum(q[:, 2]+.5+target-height, 0.)
                gradient[:, 2] += top
                value += .5*(top@top)
            return value, gradient.ravel()

        result = minimize(objective, x.ravel(), jac=True, method="L-BFGS-B",
                          options=dict(maxiter=6000, maxls=40, maxcor=20,
                                       gtol=1e-12, ftol=1e-22))
        x = result.x.reshape(count, 3)
        pair_gap = float(cKDTree(x).query(x, k=2)[0][:, 1].min()-1.) if count > 1 else None
        wall_gap = float(R-.5-norm(x[:, :2]).max())
        floor_gap = float(x[:, 2].min()-.5)
        top_gap = float(height-.5-x[:, 2].max()) if height is not None else None
        gaps = [wall_gap, floor_gap]+([pair_gap] if pair_gap is not None else [])+([top_gap] if top_gap is not None else [])
        log.append(dict(attempt=attempt+1, iterations=int(result.nit), objective=float(result.fun),
                        minimum_pair_gap=pair_gap, wall_gap=wall_gap, floor_gap=floor_gap,
                        top_gap=top_gap, optimizer_message=str(result.message)))
        if min(gaps) >= target-1e-8:
            displacement = norm(x-initial)
            return x, dict(requested_clearance_over_dp=clearance,
                           solver_guard_over_dp=guard, elapsed_seconds=time.perf_counter()-started,
                           maximum_center_shift_over_dp=float(displacement.max()),
                           rms_center_shift_over_dp=float(np.sqrt(np.mean(displacement**2))),
                           iterations=log,
                           note="Geometric cleanup without gravity; not a new mechanical equilibrium.")
    raise RuntimeError(f"Could not enforce clearance for {count} particles in the requested cylinder. "
                       f"Try a larger cylinder/height or smaller gap. Last check: {log[-1]}")


def validate_geometry(centers, diameter, R, clearance, *, floor=0., height=None, expected_count=None):
    """Strict checks; no acceptance tolerance is subtracted from the gap."""
    x = np.asarray(centers)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("Invalid exported particle coordinates.")
    if expected_count is not None and len(x) != expected_count:
        raise RuntimeError(f"Particle count changed: expected {expected_count}, found {len(x)}.")
    minimum = float(cKDTree(x).query(x, k=2)[0][:, 1].min()) if len(x) > 1 else None
    wall_gap = float(R-np.max(norm(x[:, :2])+diameter/2))
    floor_gap = float(np.min(x[:, 2]-diameter/2-floor))
    top_gap = float(floor+height-np.max(x[:, 2]+diameter/2)) if height is not None else None
    if ((minimum is not None and minimum < diameter+clearance)
            or wall_gap < clearance or floor_gap < clearance
            or (top_gap is not None and top_gap < clearance)):
        raise RuntimeError(f"Final clearance check failed: minimum distance={minimum}, "
                           f"required={diameter+clearance:.17g}, wall={wall_gap:.17g}, "
                           f"floor={floor_gap:.17g}, top={top_gap}. No final bed was exported.")
    return dict(particle_count=len(x), minimum_center_distance=minimum,
                required_center_distance=diameter+clearance,
                minimum_particle_gap=(minimum-diameter) if minimum is not None else None,
                minimum_wall_gap=wall_gap, minimum_floor_gap=floor_gap,
                minimum_top_gap=top_gap, strict_clearance_passed=True)


def generate_pebbles(pebble_diameter, core_radius, core_height=None, n_pebbles=None, *,
                     clearance=None, load_fraction=.2, seed=42, backend="auto", kn=50000.,
                     mu=.4, mu_wall=.4, restitution=.35, drag=.3, dt_scale=1.,
                     plate_drag=20., plate_speed=.03, ramp_time=10.,
                     max_settle_time=120., max_hold_time=100., speed_tol=1e-3,
                     force_tol=1e-2, quiet_time=.5, verbose=True, return_report=False,
                     recorder=None, insertion="pour", inlet=None, pour_interval=.5,
                     inlet_jitter=.03, inlet_vz=0., wait_until_settled=False,
                     max_pour_time=None, inlet_timeout=30., logger=None, log_interval=5., history=None, checkpoint=None, threads=1):
    """Return (N,3) centers in the input length unit, with no fixed N.

    With n_pebbles: retain exactly that many spheres. With height only:
    overfill/settle/cut/resettle, then optionally load/unload; count is an output.
    With both: height is a maximum and an incompatible combination fails.
    Optional return_report=True returns (centers, diagnostics).
    """
    model_parameters = {key:value for key,value in locals().items()
                        if key not in {"verbose", "return_report", "recorder", "logger", "history", "checkpoint"}}
    if inlet is not None: model_parameters["inlet"] = np.asarray(inlet,float).tolist()
    D, radius = float(pebble_diameter), float(core_radius)
    gap = 1e-5*D if clearance is None else float(clearance)
    H = None if core_height is None else float(core_height)
    numbers = [D, radius, gap, load_fraction, kn, mu, mu_wall, restitution, drag,
               dt_scale, plate_drag, plate_speed, ramp_time, max_settle_time,
               max_hold_time, speed_tol, force_tol, quiet_time, pour_interval,
               inlet_jitter, inlet_vz, inlet_timeout, log_interval]
    if not np.all(np.isfinite(numbers)) or (H is not None and not math.isfinite(H)):
        raise ValueError("All geometry and model parameters must be finite.")
    if D <= 0 or gap < 0 or radius <= D/2+gap or (H is not None and H <= D+2*gap):
        raise ValueError("Invalid diameter, radius, height, or clearance.")
    if n_pebbles is None and H is None:
        raise ValueError("Give --N/--n-pebbles or --height, or both.")
    if n_pebbles is not None and (int(n_pebbles) != n_pebbles or n_pebbles < 1):
        raise ValueError("The particle count must be a positive integer.")
    if (load_fraction < 0 or min(kn, dt_scale, plate_drag, plate_speed, ramp_time,
                                max_settle_time, max_hold_time, speed_tol, force_tol, quiet_time) <= 0
            or min(mu, mu_wall, drag) < 0 or not 0 < restitution <= 1 or dt_scale > 1):
        raise ValueError("Invalid DEM load, stiffness, friction, restitution, timestep, or settling parameters.")
    if backend not in {"auto", "numpy", "numba"}:
        raise ValueError("backend must be auto, numpy, or numba.")
    if insertion not in {"pour", "cloud"} or min(pour_interval, inlet_timeout, log_interval) <= 0 or inlet_jitter < 0 or inlet_vz > 0:
        raise ValueError("Invalid insertion mode, feed interval, inlet velocity/scatter, timeout, or log interval.")
    if max_pour_time is not None and (not math.isfinite(max_pour_time) or max_pour_time <= 0):
        raise ValueError("--max-pour-time must be positive and finite.")
    R, height, delta = radius/D, None if H is None else H/D, gap/D
    inlet_requested = None if inlet is None else np.asarray(inlet, dtype=float)/D
    if inlet_requested is not None:
        points = np.atleast_2d(inlet_requested)
        if (inlet_requested.ndim > 2 or points.shape[1] != 3 or not len(points) or not np.all(np.isfinite(points))
                or np.any(norm(points[:, :2])+.5+delta > R) or np.any(points[:, 2] <= .5+delta)):
            raise ValueError("Every fixed inlet must be above the floor and inside the cylinder with particle clearance.")
        if len(points) > 1 and cKDTree(points).query(points, k=2)[0][:, 1].min() < 1.0001:
            raise ValueError("Fixed inlets must be at least 1.0001 pebble diameters apart.")
    count = int(n_pebbles) if n_pebbles is not None else max(1, math.ceil(.72*6*R*R*height))
    history, stages = ([] if history is None else history), []
    started = logger.started if logger is not None else time.perf_counter()
    settings = dict(plate_drag=plate_drag, plate_speed=plate_speed, speed_tol=speed_tol,
                    force_tol=force_tol, quiet_time=quiet_time, verbose=verbose, recorder=recorder,
                    logger=logger, started=started, log_interval=log_interval, checkpoint=checkpoint)
    elapsed_offset, steps_offset, releases = 0., 0, []
    restored = None
    start_attempt = 0
    if checkpoint is not None and checkpoint.resume_path is not None:
        sim,meta=checkpoint.restore(backend,threads)
        restored=meta["context"]
        allowed={"backend", "threads", "max_settle_time", "max_hold_time", "max_pour_time", "inlet_timeout", "log_interval"}
        for key,value in model_parameters.items():
            if key not in allowed and value != restored["parameters"].get(key):
                raise ValueError(f"Cannot change {key} when resuming this checkpoint.")
        start_attempt, count = restored["filling_attempt"], restored["count"]
        stages,releases=restored["stages"],restored["releases"]
        announce(settings,f"[resume] Restored N={len(sim.x)}/{sim.target_count}, t*={sim.elapsed:.2f}, step={sim.total_steps}; contact and random histories restored.")
    context={}

    def completed_stage(name, duration, **kwargs):
        if name in context["completed"]:
            return
        result=run_stage(sim,name,duration,settings,history,**kwargs)
        stages.append(result);context["completed"].append(name)
        if checkpoint is not None: checkpoint.boundary(sim)

    for filling_attempt in range(start_attempt,4):
        if restored is not None:
            context=restored
            rng=np.random.default_rng();rng.bit_generator.state=context["rng_state"]
            position=sim.inlet_position
            settings["plate_mass"]=context["plate_mass"]
            context["parameters"]=model_parameters
        else:
            if recorder is not None: recorder.reset()
            rng=np.random.default_rng(seed)
            if insertion == "pour":
                release_z=max((height+2.) if height is not None else 0.,1.3*count/(6*R*R*.5)+.5*R+2.)
                position=np.array([0.,0.,release_z]) if inlet_requested is None else inlet_requested.copy()
                initial=np.atleast_2d(position)[:count]
            else:
                position=None;initial=random_cloud(count,R,rng)
            sim=PackingDEM(initial,R,backend=backend,threads=threads,kn=kn,mu=mu,mu_wall=mu_wall,restitution=restitution,drag=drag,dt_scale=dt_scale)
            sim.target_count,sim.elapsed,sim.total_steps=count,elapsed_offset,steps_offset
            sim.last_quiet_sample=sim.elapsed
            settings["plate_mass"]=max(load_fraction*count,1e-12)
            if load_fraction:
                mass=settings["plate_mass"];sim.dt=.08*math.sqrt(min(.5,mass/(1.+mass))/kn)*dt_scale
            releases=[]
            context=dict(parameters=model_parameters,filling_attempt=filling_attempt,count=count,completed=[],stages=stages,
                         releases=releases,plate_mass=settings["plate_mass"],height_cut_done=False,plate_initialized=False)
            if insertion == "pour":
                for k in range(len(sim.x)): sim.v[k]=[*rng.normal(0.,inlet_jitter,2),inlet_vz]
        context["rng"]=rng
        if checkpoint is not None: checkpoint.context=context
        announce(settings, f"Backend={sim.backend}; threads={threads}; target={count}; R/d={R:g}; d={D:g}; "
                           f"insertion={insertion}; load={load_fraction:.1%} of retained bed weight")
        if insertion == "pour" and "pouring" not in context["completed"]:
            announce(settings, f"Fixed inlet{'s' if position.ndim == 2 else ''} (input length units): {(position*D).tolist()}; "
                               f"minimum feed interval={pour_interval:g}; horizontal velocity std={inlet_jitter:g}; "
                               f"wait for settling={wait_until_settled}. Inlet positions are never randomized.")
            duration = max_pour_time if max_pour_time is not None else count*(pour_interval+(max_settle_time if wait_until_settled else 2.))+max_settle_time
            stage, releases = pour_particles(sim, count, position, rng, pour_interval, inlet_jitter,
                                            inlet_vz, wait_until_settled, duration, inlet_timeout, settings, history, releases)
            stages.append(stage);context["completed"].append("pouring")
            if checkpoint is not None: checkpoint.boundary(sim)
        completed_stage("gravity_settle",max_settle_time,settle=True)
        if n_pebbles is not None or context["height_cut_done"] or sim.x[:, 2].max()+.5 >= height:
            break
        count = max(count+1, math.ceil(1.3*count))
        elapsed_offset, steps_offset = sim.elapsed, sim.total_steps
        restored=None
        announce(settings, "Increasing the overfill count and starting a new bed to reach the requested height.")
    else:
        raise RuntimeError("Could not overfill the requested height; specify --N explicitly.")
    if n_pebbles is None and not context["height_cut_done"]:
        keep = sim.x[:, 2]+.5 <= height-delta-1e-6
        if not np.any(keep):
            raise RuntimeError("No complete pebble fits below the requested height.")
        sim.retain(keep)
        count = len(sim.x)
        settings["plate_mass"] = max(load_fraction*count, 1e-12)
        if load_fraction:
            mass = settings["plate_mass"]
            sim.dt = .08*math.sqrt(min(.5, mass/(1.+mass))/kn)*dt_scale
        context.update(height_cut_done=True,count=count,plate_mass=settings["plate_mass"])
    if n_pebbles is None:
        completed_stage("settle_after_height_cut",max_settle_time,settle=True)
    if load_fraction:
        target = load_fraction*len(sim.x)
        if not context["plate_initialized"]:
            sim.plate_z,sim.plate_v=float(sim.x[:,2].max()+.5)+.02,0.
            context["plate_initialized"]=True
        completed_stage("loading_ramp",ramp_time,start_load=0.,end_load=target,plate_mode="dynamic")
        completed_stage("loaded_hold",max_hold_time,start_load=target,settle=True,plate_mode="dynamic")
        completed_stage("unloading_ramp",ramp_time,start_load=target,end_load=0.,plate_mode="dynamic")
        if "retract_time" not in context:
            context["retract_time"]=max(15.,(float(sim.x[:,2].max()+.5)+.15-sim.plate_z)/plate_speed+5.)
        completed_stage("plate_retraction",context["retract_time"],plate_mode="retract")
        sim.plate_enabled, sim.plate_v = False, 0.
        completed_stage("unloaded_settle",max_settle_time,settle=True)
    native = sim.x.copy()
    if n_pebbles is not None and height is not None and native[:, 2].max()+.5 > height:
        raise RuntimeError(f"The settled bed height is {(native[:,2].max()+.5)*D:.8g}, "
                           f"above the requested maximum {H:g}. Increase height or reduce N.")
    announce(settings, f"[clearance] Enforcing the final gap for {len(native)} particles; t_total*={sim.elapsed:.2f}.")
    clean, cleanup = enforce_clearance(native, R, delta, height)
    centers = clean*D
    validation = validate_geometry(centers, D, radius, gap, height=H,
                                  expected_count=None if n_pebbles is None else int(n_pebbles))
    if recorder is not None:
        recorder.add_final(clean, sim.elapsed, delta)
    report = dict(script_version=__version__, backend=sim.backend,
                  parameters=dict(pebble_diameter=D, core_radius=radius, core_height=H,
                                  requested_count=n_pebbles, clearance=gap, load_fraction=load_fraction,
                                  seed=seed, threads=threads,kn=kn, mu=mu, mu_wall=mu_wall, restitution=restitution,
                                  drag=drag, dt_scale=dt_scale, plate_drag=plate_drag,
                                  plate_speed=plate_speed, ramp_time=ramp_time,
                                  max_settle_time=max_settle_time, max_hold_time=max_hold_time,
                                  speed_tol=speed_tol, force_tol=force_tol, quiet_time=quiet_time,
                                  insertion=insertion, inlet=None if sim.inlet_position is None else (sim.inlet_position*D).tolist(),
                                  pour_interval=pour_interval, inlet_jitter=inlet_jitter, inlet_vz=inlet_vz,
                                  wait_until_settled=wait_until_settled, max_pour_time=max_pour_time,
                                  inlet_timeout=inlet_timeout, log_interval=log_interval),
                  actual_count=len(clean), native_height=float(native[:, 2].max()+.5)*D,
                  final_height=float(clean[:, 2].max()+.5)*D,
                  interior_porosity=interior_porosity(clean, R),
                  geometry_cleanup=cleanup, validation=validation, stages=stages,
                  elapsed_seconds=time.perf_counter()-started,
                  total_simulation_time=sim.elapsed, total_steps=sim.total_steps,
                  release_events=releases, history=history, native_centers=native*D,
                  settled_definition="Each retained particle must satisfy speed, surface spin, force and scaled torque tolerances continuously for quiet_time; later impacts can unsettle it.",
                  note="Final geometry is unloaded and clearance-corrected. Native DEM equilibrium precedes geometric cleanup.")
    return (centers, report) if return_report else centers


def _circle_overlap(a, distance, b):
    a, distance, b = np.broadcast_arrays(a, distance, b)
    out = np.zeros(a.shape)
    inside = distance <= np.abs(a-b)
    out[inside] = math.pi*np.minimum(a, b)[inside]**2
    partial = ~inside & (distance < a+b)
    aa, dd, bb = a[partial], distance[partial], b[partial]
    c1 = np.clip((dd*dd+aa*aa-bb*bb)/(2*dd*aa), -1., 1.)
    c2 = np.clip((dd*dd+bb*bb-aa*aa)/(2*dd*bb), -1., 1.)
    product = (-dd+aa+bb)*(dd+aa-bb)*(dd-aa+bb)*(dd+aa+bb)
    out[partial] = aa*aa*np.arccos(c1)+bb*bb*np.arccos(c2)-.5*np.sqrt(np.maximum(product, 0.))
    return out


def radial_porosity(centers, diameter, R, *, trim=1., width=None):
    """Annular volume porosity, clipping spheres at the axial slab boundaries."""
    a = diameter/2
    lo = float(centers[:, 2].min()-a)+trim*diameter
    hi = float(centers[:, 2].max()+a)-trim*diameter
    if hi <= lo:
        raise ValueError("No interior slab remains for porosity; use --end-trim 0 for a shallow bed.")
    width = diameter/20 if width is None else width
    edges = np.linspace(0., R, math.ceil(R/width)+1)
    rho = norm(centers[:, :2])
    low, high = np.maximum(-a, lo-centers[:, 2]), np.minimum(a, hi-centers[:, 2])
    active = high > low
    rho, low, high = rho[active], low[active], high[active]
    solid = math.pi*(a*a*(high-low)-(high**3-low**3)/3)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    half, mid = .5*(high-low), .5*(high+low)
    cumulative = np.zeros(len(edges))
    for k, radius in enumerate(edges[1:], 1):
        contained = rho+a <= radius
        partial = ~contained & (rho-a < radius)
        cumulative[k] = solid[contained].sum()
        if partial.any():
            z = mid[partial, None]+half[partial, None]*nodes
            disks = np.sqrt(np.maximum(a*a-z*z, 0.))
            areas = _circle_overlap(radius, rho[partial, None], disks)
            cumulative[k] += np.sum(half[partial]*(areas@weights))
    shell_volume = math.pi*np.diff(edges**2)*(hi-lo)
    epsilon = 1.-np.diff(cumulative)/shell_volume
    radial = (2./3.)*(edges[1:]**3-edges[:-1]**3)/(edges[1:]**2-edges[:-1]**2)
    mean = float(1.-cumulative[-1]/(math.pi*R*R*(hi-lo)))
    return radial, epsilon, mean, (lo, hi)


def de_klerk(s, epsilon_inf):
    """de Klerk (2003), DOI 10.1002/aic.690490812; 0.637-threshold form."""
    s = np.asarray(s)
    return np.where(s <= .637, 2.14*s*s-2.53*s+1.,
                    epsilon_inf+.29*np.exp(-.6*s)*np.cos(2.3*math.pi*(s-.16))+.15*np.exp(-.9*s))


def de_klerk_asymptote(R_over_d, mean):
    if R_over_d <= .637 or not 0 < mean < 1:
        raise ValueError("Matching de Klerk's mean requires R/d > 0.637 and a mean between 0 and 1.")
    nodes, weights = np.polynomial.legendre.leggauss(128)
    base = 0.
    for lo, hi in [(0., .637), (.637, R_over_d)]:
        s = .5*(hi+lo)+.5*(hi-lo)*nodes
        base += float(.5*(hi-lo)*np.dot(weights, 2.*(R_over_d-s)/R_over_d**2*de_klerk(s, 0.)))
    epsilon_inf = (mean-base)/((R_over_d-.637)/R_over_d)**2
    if not 0 <= epsilon_inf <= 1:
        raise ValueError("That radius and prescribed mean give an unphysical de Klerk asymptote.")
    return epsilon_inf


def output_prefix(output):
    output = Path(output)
    if output.is_dir():
        output = output/"pebbles_dem.dat"
    return output.with_suffix("") if output.suffix.lower() in {".csv", ".dat", ".vtk", ".json"} else output


def write_final(centers, report, output, *, units="input", origin="bottom",
                file_format="xyz", universe="pebble", save_native=False):
    """Validate the serialized coordinates before publishing any final files."""
    parameters = report["parameters"]
    D, R, gap, H = (parameters[k] for k in ["pebble_diameter", "core_radius", "clearance", "core_height"])
    scale = 1./D if units == "diameter" else 1.
    zshift = 0.
    if origin == "center":
        zshift = -.5*(H if H is not None else float(centers[:, 2].max()+D/2))
    elif origin == "centroid":
        zshift = -float(centers[:, 2].mean())
    x = centers.copy()
    x[:, 2] += zshift
    x *= scale
    Dout, Rout, gout, floor = D*scale, R*scale, gap*scale, zshift*scale
    Hout = None if H is None else H*scale
    prefix = output_prefix(output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((x, np.full(len(x), Dout/2)))
    with tempfile.TemporaryDirectory(prefix=".dem_export_", dir=prefix.parent) as directory:
        staged = Path(directory)/prefix.name
        np.savetxt(str(staged)+".csv", data, delimiter=",", header="x,y,z,radius", comments="", fmt="%.17g")
        csv_data = np.loadtxt(str(staged)+".csv", delimiter=",", skiprows=1, ndmin=2)
        verified = validate_geometry(csv_data[:, :3], Dout, Rout, gout, floor=floor,
                                     height=Hout, expected_count=len(centers))
        with Path(str(staged)+".dat").open("w", encoding="utf-8") as f:
            if file_format == "serpent":
                for row in csv_data:
                    f.write(" ".join(f"{value:.17g}" for value in row)+" "+universe+"\n")
            else:
                np.savetxt(f, csv_data[:, :3], fmt="%.17g")
        dat_data = np.loadtxt(str(staged)+".dat", usecols=(0, 1, 2), ndmin=2)
        if not np.array_equal(csv_data[:, :3], dat_data):
            raise RuntimeError("DAT and CSV coordinates differ; no final files were exported.")
        validate_geometry(dat_data, Dout, Rout, gout, floor=floor, height=Hout, expected_count=len(centers))
        with Path(str(staged)+".vtk").open("w", encoding="ascii") as f:
            f.write("# vtk DataFile Version 3.0\nValidated DEM pebble centers\nASCII\nDATASET POLYDATA\n")
            f.write(f"POINTS {len(x)} double\n")
            np.savetxt(f, csv_data[:, :3], fmt="%.17g")
            f.write(f"VERTICES {len(x)} {2*len(x)}\n")
            for i in range(len(x)):
                f.write(f"1 {i}\n")
            f.write(f"POINT_DATA {len(x)}\nSCALARS radius double 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, csv_data[:, 3], fmt="%.17g")
        serializable = {key: value for key, value in report.items() if key not in {"history", "native_centers"}}
        serializable["export"] = dict(units=units, scale_from_input=scale, origin=origin,
                                     floor_z=floor, diameter=Dout, radius=Rout, clearance=gout,
                                     maximum_height=Hout, validation=verified)
        Path(str(staged)+".json").write_text(json.dumps(serializable, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        if report.get("history"):
            with Path(str(staged)+"_history.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(report["history"][0]))
                writer.writeheader(); writer.writerows(report["history"])
        if report.get("release_events"):
            with Path(str(staged)+"_releases.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(report["release_events"][0]))
                writer.writeheader(); writer.writerows(report["release_events"])
        if save_native and report.get("native_centers") is not None:
            native = report["native_centers"].copy()
            native[:, 2] += zshift
            native *= scale
            np.savetxt(str(staged)+"_native_soft_contacts.csv", np.column_stack((native, np.full(len(native), Dout/2))),
                       delimiter=",", header="x,y,z,radius", comments="", fmt="%.17g")
        for path in Path(directory).iterdir():
            path.replace(prefix.parent/path.name)
    return csv_data[:, :3], prefix, verified, Dout, Rout


def read_centers(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return np.asarray(data["centers"], dtype=float)
    with path.open(encoding="utf-8-sig") as f:
        first = f.readline()
    delimiter = "," if "," in first else None
    try:
        [float(t) for t in (first.split(",") if delimiter else first.split())[:3]]
        skip = 0
    except ValueError:
        skip = 1
    return np.loadtxt(path, delimiter=delimiter, skiprows=skip, usecols=(0, 1, 2), ndmin=2)


def self_test(backend="auto", threads=1):
    """Small physical/geometry checks; not a calibration of a real material."""
    x = np.array([[0., 0., .49], [.95, .03, .50], [1.91, .01, .48], [3.515, 0., 1.], [0., 1., 1.46]])
    reference = PackingDEM(x, 4., backend="numpy")
    other = PackingDEM(x, 4., backend=backend, threads=threads)
    if other.parallel:
        other.parallel_min_particles=0;other.rebuild()
    rng = np.random.default_rng(8)
    velocity, spin = rng.normal(0., .05, x.shape), rng.normal(0., .03, x.shape)
    history = rng.normal(0., 1e-5, reference.xi.shape)
    wall_history = rng.normal(0., 1e-5, reference.wall_xi.shape)
    for sim in [reference, other]:
        sim.v[:] = velocity; sim.w[:] = spin
        sim.xi[:] = history; sim.wall_xi[:] = wall_history
        sim.plate_enabled, sim.plate_z, sim.plate_v = True, 1.95, -.02
    a, ta, oa = reference.forces(reference.dt)
    b, tb, ob = other.forces(other.dt)
    if not (np.allclose(a, b, rtol=1e-11, atol=1e-9) and np.allclose(ta, tb, rtol=1e-11, atol=1e-9)
            and np.allclose(reference.xi, other.xi, atol=1e-12)
            and np.allclose(reference.wall_xi, other.wall_xi, atol=1e-12) and abs(oa-ob) < 1e-12):
        raise RuntimeError("Contact force/torque/history parity check failed.")
    single = PackingDEM([[0., 0., 1.]], 3., backend=backend)
    single.plate_enabled, single.plate_z = True, 1.49
    f, t, _ = single.forces(0.)
    if abs(-f[0, 2]-1.-single.plate_reaction) > 1e-10 or np.any(t):
        raise RuntimeError("Smooth-plate action/reaction check failed.")
    for distance in [.99, 1.000005]:
        try:
            validate_geometry(np.array([[0., 0., .6], [distance, 0., .6]]), 1., 3.5, 1e-5)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("An insufficient gap was incorrectly accepted.")
    clean, _ = enforce_clearance(np.array([[0., 0., .499], [.999, 0., .499], [0., 1., .499]]), 3.5, 1e-5)
    validate_geometry(clean, 1., 3.5, 1e-5, expected_count=3)
    growing = PackingDEM(x, 4., backend=backend)
    growing.v[:] = velocity; growing.w[:] = spin
    growing.xi[:] = history; growing.wall_xi[:] = wall_history
    before, torque_before, _ = growing.forces(0.)
    growing.append_particle([0., 0., 4.], [0., 0., 0.])
    after, torque_after, _ = growing.forces(0.)
    if not (np.allclose(before, after[:-1], rtol=1e-11, atol=1e-9)
            and np.allclose(torque_before, torque_after[:-1], rtol=1e-11, atol=1e-9)
            and growing.quiet_elapsed.shape == (len(x)+1,)):
        raise RuntimeError("Particle insertion changed old contact forces/history.")
    return dict(contact_backend=other.backend, force_torque_history_parity=True,
                plate_action_reaction=True, insufficient_gap_rejected=True,
                variable_count_cleanup=True, insertion_preserves_contact_forces=True,
                contact_threads=threads, parallel_contact_parity=other.parallel)


def benchmark_packing(count, radius, *, threads=1, backend="numba", kn=50000., drag=.3,
                      dt_scale=1., interval=.5, velocity_z=0., video_max_frames=600, inlets=1, post_pour_time=0.):
    """Short synthetic contact benchmark; never generates an actual packed bed."""
    if count is None or count<1 or radius<=.501 or inlets<1:
        raise ValueError("Benchmark needs a positive --N and --R/--dp > .501.")
    axis=np.arange(-math.ceil(radius),math.ceil(radius)+1)*.9998
    xx,yy=np.meshgrid(axis,axis)
    xy=np.column_stack((xx.ravel(),yy.ravel()))
    xy=xy[norm(xy)<=radius-.501]
    geometry=np.vstack([np.column_stack((xy,np.full(len(xy),.5+k*.9998)))
                        for k in range(math.ceil(count/len(xy)))])[:count]
    settings=dict(plate_mass=1.,plate_drag=20.,plate_speed=.03)
    sizes=sorted({min(count,1000),max(1,count//2),count})
    rows=[]
    print("Synthetic dense-contact benchmark; compilation is excluded. No packing trajectory is generated.",flush=True)
    for n in sizes:
        samples=[]
        for trial in range(2):
            sim=PackingDEM(geometry[:n],radius,backend=backend,threads=threads,kn=kn,drag=drag,dt_scale=dt_scale)
            sim.advance(1,settings)
            started=time.perf_counter();sim.advance(100,settings)
            samples.append((time.perf_counter()-started)/100)
        rebuilds=[]
        for trial in range(3):
            started=time.perf_counter();sim.rebuild();rebuilds.append(time.perf_counter()-started)
        row=dict(particles=n,contact_candidates=len(sim.pairs),step_seconds=float(np.median(samples)),
                 rebuild_seconds=float(np.median(rebuilds)),
                 active_contact_threads=threads if sim.parallel and n>=sim.parallel_min_particles else 1)
        rows.append(row)
        print(f"N={n:7d} | threads={row['active_contact_threads']} | {1000*row['step_seconds']:.4f} ms/step"
              f" | neighbor rebuild {1000*row['rebuild_seconds']:.1f} ms",flush=True)
    downward=-velocity_z
    def travel(t):
        if drag==0.:return downward*t+.5*t*t
        terminal=1./drag
        return terminal*t+(downward-terminal)*(-math.expm1(-drag*t))/drag
    low,high=0.,1.
    while travel(high)<1.0001:high*=2
    for _ in range(60):
        middle=(low+high)/2
        if travel(middle)<1.0001:low=middle
        else:high=middle
    release_steps=200*math.ceil(max(interval,high)/(200*sim.dt))
    # The fastest falling pebble triggers a rebuild each time it travels skin/2.
    fall_speed=max(downward,1./drag) if drag>0. else None
    rebuild_every=max(1.,.5*sim.skin/(fall_speed*sim.dt)) if fall_speed else None
    points=np.array([0]+[r["particles"] for r in rows],float)
    def pour_seconds(costs):
        seconds=np.array([costs[0]/rows[0]["particles"]]+list(costs))
        return float(np.sum(.5*(seconds[1:]+seconds[:-1])*np.diff(points))*release_steps/inlets)
    estimate=pour_seconds([r["step_seconds"] for r in rows])
    pour=pour_seconds([r["step_seconds"]+(r["rebuild_seconds"]/rebuild_every if rebuild_every else 0.) for r in rows])
    post=post_pour_time/sim.dt*rows[-1]["step_seconds"]
    result=dict(backend=sim.backend,threads=threads,inlets=inlets,measurements=rows,estimated_pour_core_days=estimate/86400.,
                nominal_release_steps=release_steps,nominal_total_pour_steps=math.ceil(max(0,count-inlets)/inlets)*release_steps,
                rebuild_interval_steps=rebuild_every,
                estimated_pour_hours_force_only=estimate/3600.,estimated_pour_hours_with_rebuilds=pour/3600.,
                post_pour_stage_hours_upper_bound=post/3600.,estimated_total_hours=(pour+post)/3600.,
                bounded_video_position_MB=(video_max_frames+1)*count*12/1e6,
                scope="Synthetic cubic-lattice contacts. The pour integrates the measured step cost over the growing bed for the given "
                      "inlet count, interval and release velocity, with one neighbor rebuild per skin/2 of travel by the fastest falling "
                      "pebble. The post-pour term is an upper bound from the settling/hold caps at full N. Excludes final clearance "
                      "cleanup, snapshots, checkpoints, history sampling and inlet blockage; a full 8-inlet test pour at N=2500 "
                      "took 1.23x this estimate. Not a completion-time guarantee.")
    print(json.dumps(result,indent=2),flush=True)
    print(f"ESTIMATE threads={threads} inlets={inlets}: pour {pour/3600:.2f} h (force only {estimate/3600:.2f} h) "
          f"+ post-pour <= {post/3600:.2f} h = {(pour+post)/3600:.2f} h",flush=True)
    return result


class PackingRecorder:
    """Bounded recording, with online thinning when the frame budget is reached."""
    def __init__(self, interval=.1, max_frames=600):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Video interval must be positive and finite.")
        if int(max_frames) != max_frames or max_frames < 32:
            raise ValueError("Video max frames must be an integer >= 32.")
        self.interval, self.max_frames = interval, int(max_frames)
        self.reset()

    def reset(self):
        self.frames, self.next_time, self.raw_count, self.stride = [], 0., 0, 1

    def _thin(self):
        if len(self.frames) < self.max_frames:
            return
        removable = [i for i, f in enumerate(self.frames) if i and not f.get("boundary", False)]
        remove = set(removable[::2])
        if not remove:
            remove = {1}
        self.frames = [f for i, f in enumerate(self.frames) if i not in remove]
        self.stride *= 2

    def capture(self, sim, stage, load_fraction=0., force=False):
        if not force and sim.elapsed < self.next_time:
            return
        self.raw_count += 1
        first_batch = 1 if sim.inlet_position is None else len(np.atleast_2d(sim.inlet_position))
        boundary = force and (stage != "pouring" or len(sim.x) <= first_batch or len(sim.x) == sim.target_count)
        if sim.elapsed >= self.next_time:
            self.next_time = (math.floor(sim.elapsed/self.interval)+1)*self.interval
        if not boundary and self.raw_count % self.stride:
            return
        self._thin()
        item = dict(time=float(sim.elapsed), stage=stage,
                    positions=sim.x.astype(np.float32, copy=True),
                    plate_z=float(sim.plate_z) if sim.plate_enabled else None,
                    load_fraction=float(load_fraction), released=sim.released_total,
                    target=sim.target_count, settled=sim.settled_count,
                    inlet=None if sim.inlet_position is None else sim.inlet_position.copy(), boundary=boundary)
        if (self.frames and self.frames[-1]["time"] == item["time"]
                and self.frames[-1]["stage"] == stage):
            self.frames[-1] = item
        else:
            self.frames.append(item)

    def add_final(self, positions, elapsed, clearance):
        self._thin()
        previous = self.frames[-1] if self.frames else {}
        self.frames.append(dict(time=float(elapsed), stage="clearance_verified",
                                positions=np.array(positions, dtype=float, copy=True),
                                plate_z=None, load_fraction=0., clearance=clearance,
                                released=previous.get("released", len(positions)),
                                target=previous.get("target", len(positions)), settled=len(positions), inlet=None, boundary=True))


class SnapshotRecorder:
    """PNG and VTK milestones, written immediately without retaining trajectories."""
    next_time = math.inf

    def __init__(self, directory, radius, diameter, every=1000, plate_opacity=.22, stage_names=None, logger=None):
        if int(every) != every or every < 1:
            raise ValueError("Snapshot interval must be a positive particle count.")
        try:
            import PIL
        except ImportError as error:
            raise RuntimeError("PNG snapshots require: python -m pip install Pillow") from error
        self.directory, self.R, self.D = Path(directory), radius, diameter
        self.directory.mkdir(parents=True, exist_ok=True)
        self.every, self.opacity, self.stage_names, self.logger = int(every), plate_opacity, stage_names, logger
        self.reset()

    def reset(self):
        self.next_count, self.first_item = 1, None

    def capture(self, sim, stage, load_fraction=0., force=False):
        if stage == "pouring":
            if len(sim.x) < self.next_count and len(sim.x) != sim.target_count:
                return
            self.next_count = (len(sim.x)//self.every+1)*self.every
        elif not force:
            return
        item = dict(time=sim.elapsed, stage=stage, positions=sim.x.copy(),
                    plate_z=sim.plate_z if sim.plate_enabled else None, load_fraction=load_fraction,
                    released=sim.released_total, target=sim.target_count, settled=sim.settled_count,
                    inlet=None if sim.inlet_position is None else sim.inlet_position.copy())
        if self.first_item is None:
            self.first_item = dict(item, positions=item["positions"][:1].copy())
        self._write(item, sim.total_steps, verified=False)

    def add_final(self, positions, elapsed, clearance):
        previous = self.first_item or {}
        item = dict(time=elapsed, stage="clearance_verified", positions=positions,
                    plate_z=None, load_fraction=0., released=previous.get("target", len(positions)),
                    target=previous.get("target", len(positions)), settled=len(positions), inlet=None, clearance=clearance)
        self._write(item, -1, verified=True)

    def _write(self, item, step, verified):
        state = "verified" if verified else "native"
        stem = f"{item['stage']}_n{len(item['positions']):06d}_s{max(0,step):012d}_{state}"
        renderer = PackingVideoRenderer(self.R, [self.first_item or item, item], plate_opacity=self.opacity,
                                        stage_names=self.stage_names)
        if item["stage"] not in {"pouring", "gravity_settle"}:
            renderer.camera_height = max(float(item["positions"][:,2].max()+.5)*1.2, 1.5)
        picture = renderer.render(dict(item,snapshot=True))
        with tempfile.TemporaryDirectory(prefix=".snapshot_", dir=self.directory) as tmp:
            tmp = Path(tmp)
            picture.save(tmp/(stem+".png"))
            x = item["positions"]*self.D
            with (tmp/(stem+".vtk")).open("w", encoding="ascii") as f:
                f.write("# vtk DataFile Version 3.0\n"+state+" DEM snapshot; input length units, bottom origin\nASCII\nDATASET POLYDATA\n")
                f.write(f"POINTS {len(x)} double\n"); np.savetxt(f,x,fmt="%.17g")
                f.write(f"VERTICES {len(x)} {2*len(x)}\n")
                for i in range(len(x)): f.write(f"1 {i}\n")
                f.write(f"POINT_DATA {len(x)}\nSCALARS radius double 1\nLOOKUP_TABLE default\n")
                np.savetxt(f,np.full(len(x),self.D/2),fmt="%.17g")
            metadata = {k:v for k,v in item.items() if k not in {"positions", "inlet"}}
            metadata.update(particle_count=len(x), total_steps=step, input_diameter=self.D,
                            coordinate_units="input length units; bottom origin", clearance_verified=verified)
            (tmp/(stem+".json")).write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
            for path in tmp.iterdir(): path.replace(self.directory/path.name)
        if self.logger is not None:
            self.logger.write(f"[snapshot] {self.directory/stem}.png + .vtk | N={len(x)} t*={item['time']:.2f} {state}")


class RecordingBundle:
    def __init__(self, video=None, snapshots=None):
        self.video, self.snapshots = video, snapshots

    @property
    def next_time(self): return self.video.next_time if self.video is not None else math.inf

    def reset(self):
        for item in (self.video, self.snapshots):
            if item is not None: item.reset()

    def capture(self, *args, **kwargs):
        for item in (self.video, self.snapshots):
            if item is not None: item.capture(*args, **kwargs)

    def add_final(self, *args):
        for item in (self.video, self.snapshots):
            if item is not None: item.add_final(*args)


def video_dependencies():
    """Resolve optional video packages before spending time on the DEM run."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("Video requires Pillow: python -m pip install Pillow imageio-ffmpeg") from error
    try:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("MP4 encoding requires: python -m pip install imageio-ffmpeg")
    result = subprocess.run([executable, "-hide_banner", "-encoders"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode or "libx264" not in result.stdout:
        raise RuntimeError("This FFmpeg lacks the libx264 encoder. Install the imageio-ffmpeg wheel or an FFmpeg build with libx264.")
    return executable


_STAGE_LABELS = {
    "pouring": "Pouring pebbles",
    "gravity_settle": "Gravity settling",
    "settle_after_height_cut": "Settling after height cut",
    "loading_ramp": "Loading top plate",
    "loaded_hold": "Holding top load",
    "unloading_ramp": "Unloading top plate",
    "plate_retraction": "Removing top plate",
    "unloaded_settle": "Unloaded settling",
    "clearance_verified": "Clearance verified",
}


class PackingVideoRenderer:
    """Orthographic sphere surfaces with a depth buffer; no OpenGL required."""
    def __init__(self, radius, frames, width=1280, height=720, plate_opacity=.22, stage_names=None):
        from PIL import ImageFont
        self.R, self.width, self.height = radius, width, height
        self.plate_opacity = plate_opacity
        self.vw, self.vh = int(width*.71), int(height*.77)
        self.camera_height = max(float(frames[0]["positions"][:, 2].max()+.5)*1.10, 2.)
        self.wall_height = self.camera_height
        azimuth, elevation = math.radians(-55), math.radians(22)
        self.view = np.array([math.cos(elevation)*math.cos(azimuth), math.cos(elevation)*math.sin(azimuth), math.sin(elevation)])
        self.right = np.array([-math.sin(azimuth), math.cos(azimuth), 0.])
        self.up = np.cross(self.view, self.right)
        self.elevation = elevation
        self.stages = stage_names or list(dict.fromkeys(f["stage"] for f in frames))
        self.fonts = {}
        for key, size in [("title", 31), ("subtitle", 17), ("stage", 18), ("small", 14), ("value", 27)]:
            size = max(10, round(size*width/1280))
            for name in ["DejaVuSans.ttf", "arial.ttf"]:
                try:
                    self.fonts[key] = ImageFont.truetype(name, size)
                    break
                except OSError:
                    pass
            if key not in self.fonts:
                try:
                    self.fonts[key] = ImageFont.load_default(size=size)
                except TypeError:
                    self.fonts[key] = ImageFont.load_default()

    def render(self, item):
        from PIL import Image, ImageDraw
        W, H, R = self.width, self.height, self.R
        x = item["positions"]
        if item["stage"] not in {"gravity_settle", "pouring"}:
            target_height = max(float(x[:, 2].max()+.5)*1.20, 1.5)
            if item["plate_z"] is not None:
                target_height = max(target_height, item["plate_z"]+.6)
            self.camera_height += .09*(target_height-self.camera_height)
        ch = self.camera_height
        target = np.array([0., 0., ch*.45])
        scale = min(.88*self.vw/(2*R+1.), .88*self.vh/(ch*math.cos(self.elevation)+2*R*math.sin(self.elevation)+1.))

        def project(points):
            q = np.asarray(points)-target
            return np.column_stack((self.vw/2+scale*(q@self.right), self.vh*.51-scale*(q@self.up)))

        viewport = Image.new("RGB", (self.vw, self.vh), "#f9fbfc")
        drawing = ImageDraw.Draw(viewport)
        theta = np.linspace(0., 2*math.pi, 180)
        ring = np.column_stack((R*np.cos(theta), R*np.sin(theta), np.zeros_like(theta)))
        drawing.polygon([tuple(p) for p in project(ring)], fill="#e4edf0")
        for z in [0., self.wall_height]:
            ring[:, 2] = z
            drawing.line([tuple(p) for p in project(ring)], fill="#c3d1d7", width=1)
        azimuth = math.atan2(self.view[1], self.view[0])
        for angle in [azimuth-math.pi/2, azimuth+math.pi/2]:
            edge = [[R*math.cos(angle), R*math.sin(angle), 0.], [R*math.cos(angle), R*math.sin(angle), self.wall_height]]
            drawing.line([tuple(p) for p in project(edge)], fill="#c3d1d7", width=1)
        pixels = np.asarray(viewport).astype(np.float32)/255.
        depth_buffer = np.full((self.vh, self.vw), -np.inf, dtype=np.float32)
        projected = project(x)
        depths = (x-target)@self.view
        light = np.array([-.45, .6, .85]); light /= np.linalg.norm(light)
        halfway = light+np.array([0., 0., 1.]); halfway /= np.linalg.norm(halfway)
        for i, (center, depth) in enumerate(zip(projected, depths)):
            cx, cy = center
            left, right = max(0, math.floor(cx-.5*scale-1)), min(self.vw, math.ceil(cx+.5*scale+1))
            top, bottom = max(0, math.floor(cy-.5*scale-1)), min(self.vh, math.ceil(cy+.5*scale+1))
            if left >= right or top >= bottom:
                continue
            xx = (np.arange(left, right)[None, :]+.5-cx)/scale
            yy = -(np.arange(top, bottom)[:, None]+.5-cy)/scale
            rr = xx*xx+yy*yy
            zz = np.sqrt(np.maximum(.25-rr, 0.))
            front = depth+zz
            visible = (rr <= .25) & (front > depth_buffer[top:bottom, left:right])
            if not visible.any():
                continue
            diffuse = np.maximum(0., 2*(xx*light[0]+yy*light[1]+zz*light[2]))
            specular = np.maximum(0., 2*(xx*halfway[0]+yy*halfway[1]+zz*halfway[2]))**40
            base = np.array([.85, .48, .22]) if i % 17 == 0 else np.array([.20, .59, .64])
            color = base*(.32+.68*diffuse[:, :, None])+.24*specular[:, :, None]
            pixels[top:bottom, left:right][visible] = np.clip(color[visible], 0., 1.)
            depth_buffer[top:bottom, left:right][visible] = front[visible]
        viewport = Image.fromarray(np.round(pixels*255).astype(np.uint8))
        drawing = ImageDraw.Draw(viewport)
        if item["plate_z"] is not None:
            overlay = Image.new("RGBA", viewport.size, (0, 0, 0, 0))
            plate_drawing = ImageDraw.Draw(overlay)
            ring[:, 2] = item["plate_z"]
            polygon = [tuple(p) for p in project(ring)]
            plate_drawing.polygon(polygon, fill=(185, 162, 135, round(255*self.plate_opacity)))
            plate_drawing.line(polygon+[polygon[0]], fill=(123, 103, 81, 210), width=2)
            if item["load_fraction"] > 1e-5:
                line = project([[0., 0., item["plate_z"]+.85], [0., 0., item["plate_z"]+.18]])
                a, b = tuple(line[0]), tuple(line[1])
                plate_drawing.line([a, b], fill=(167, 94, 43, 255), width=3)
                plate_drawing.line([(b[0]-6, b[1]-8), b, (b[0]+6, b[1]-8)], fill=(167, 94, 43, 255), width=3)
            viewport = Image.alpha_composite(viewport.convert("RGBA"), overlay).convert("RGB")
        if item["stage"] == "pouring" and item.get("inlet") is not None:
            drawing = ImageDraw.Draw(viewport)
            points = np.atleast_2d(item["inlet"])
            for inlet in points:
                nozzle = np.column_stack((inlet[0]+.65*np.cos(theta), inlet[1]+.65*np.sin(theta), np.full_like(theta, inlet[2]+.1)))
                drawing.line([tuple(p) for p in project(nozzle)], fill="#965c35", width=2)
            marker = project([points[0]])[0]
            label = "Fixed release point" if len(points) == 1 else f"{len(points)} fixed release points"
            drawing.text((marker[0]+8, marker[1]-26), label, font=self.fonts["small"], fill="#965c35")
        image = Image.new("RGB", (W, H), "#f9fbfc")
        image.paste(viewport, (0, round(H*.15)))
        d = ImageDraw.Draw(image)
        ink, muted, teal = "#173d49", "#617983", "#087f8c"
        d.text((W*.035, H*.030), "DEM packing of a pebble bed", font=self.fonts["title"], fill=ink)
        d.text((W*.036, H*.095), f"Released {item.get('released',len(x))}/{item.get('target',len(x))}  |  settled {item.get('settled',0)}  |  retained {len(x)}  |  R/d = {R:g}", font=self.fonts["subtitle"], fill=muted)
        sx = round(W*.745)
        d.line([(W*.72, H*.19), (W*.72, H*.88)], fill="#d4e0e5", width=1)
        d.text((sx, H*.205), "PACKING STAGES", font=self.fonts["small"], fill=muted)
        active = self.stages.index(item["stage"])
        for k, name in enumerate(self.stages):
            y = H*(.255+k*.045)
            color = teal if k == active else (ink if k < active else "#a4b3bb")
            if k == active:
                d.rounded_rectangle((sx-8, y-5, W*.978, y+H*.035), radius=6, fill="#e5f0f1")
            d.text((sx, y), _STAGE_LABELS.get(name, name.replace("_", " ")), font=self.fonts["stage"], fill=color)
        y = H*.665
        for label, value in [("Height / diameter", f"{x[:,2].max()+.5:.4f}"),
                             ("Applied top load / bed weight", f"{100*item['load_fraction']:.1f}%"),
                             ("Numerical time t*", f"{item['time']:.2f}")]:
            d.text((sx, y), label, font=self.fonts["small"], fill=muted)
            d.text((sx, y+H*.027), value, font=self.fonts["value"], fill=ink)
            y += H*.083
        footer = "Recorded DEM positions; playback is accelerated.  t* = t sqrt(g/d_p)."
        if item.get("snapshot"):
            footer = "Native DEM snapshot; final exported geometry is clearance-corrected.  t* = t sqrt(g/d_p)."
        if item["plate_z"] is not None and self.plate_opacity < 1:
            footer = "DEM positions; plate drawn translucent for visibility.  t* = t sqrt(g/d_p)."
        if item["stage"] == "clearance_verified":
            footer = f"Final exported geometry verified: center distance >= 1 + {item['clearance']:g}; wall and floor gaps verified."
        lines, line = [], ""
        for word in footer.split():
            trial = (line+" "+word).strip()
            if line and d.textbbox((0, 0), trial, font=self.fonts["small"])[2] > W*.93:
                lines.append(line); line = word
            else:
                line = trial
        lines.append(line)
        for k, line in enumerate(lines):
            d.text((W*.035, H*.94+k*max(12, H*.023)), line, font=self.fonts["small"],
                   fill=teal if item["stage"] == "clearance_verified" else muted)
        return image


def write_packing_video(recorder, radius, path, *, fps=30, width=1280, height=720, verbose=True,
                        plate_opacity=.22, logger=None):
    """Encode real DEM snapshots and a final verified-geometry hold as H.264."""
    executable = video_dependencies()
    path = Path(path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("Use an .mp4 filename for --video.")
    if fps <= 0 or width < 640 or height < 480 or width % 2 or height % 2:
        raise ValueError("Video requires positive FPS and even dimensions of at least 640x480.")
    if not math.isfinite(plate_opacity) or not 0 <= plate_opacity <= 1:
        raise ValueError("Plate opacity must be between 0 and 1.")
    if not recorder.frames:
        raise ValueError("No DEM trajectory was recorded.")
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = PackingVideoRenderer(radius, recorder.frames, width, height, plate_opacity)
    total = len(recorder.frames)+2*fps
    with tempfile.TemporaryDirectory(prefix=".dem_video_", dir=path.parent) as directory:
        staged = Path(directory)/"packing.mp4"
        with tempfile.TemporaryFile() as log:
            command = [executable, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
                       "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", str(staged)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                for i in range(total):
                    item = recorder.frames[min(i, len(recorder.frames)-1)]
                    process.stdin.write(renderer.render(item).tobytes())
                    if i % max(1, total//10) == 0 or i == total-1:
                        message = f"[video] rendering {i+1}/{total} frames ({100*(i+1)/total:.1f}%)"
                        if logger is not None: logger.write(message)
                        elif verbose: print(message, flush=True)
                process.stdin.close()
                status = process.wait(timeout=60)
                if status:
                    log.seek(0)
                    raise RuntimeError("MP4 encoding failed: "+log.read().decode(errors="replace")[-2000:])
            except BaseException:
                process.kill(); process.wait()
                raise
        staged.replace(path)
    return dict(path=str(path), frames=total, fps=fps, duration_seconds=total/fps,
                width=width, height=height, numerical_snapshot_interval=recorder.interval,
                plate_opacity=plate_opacity,
                frame_budget=recorder.max_frames,online_thinning_stride=recorder.stride,
                minimum_frame_particle_count=min(len(f["positions"]) for f in recorder.frames),
                maximum_frame_particle_count=max(len(f["positions"]) for f in recorder.frames),
                trajectory="Recorded native DEM positions; final hold uses clearance-verified geometry.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diameter", "--dp", type=float, default=1., help="Pebble diameter; all input lengths use the same unit.")
    radius = p.add_mutually_exclusive_group()
    radius.add_argument("--core-radius", "--R", type=float, help="Cylinder radius.")
    radius.add_argument("--core-diameter", type=float, help="Cylinder diameter, as an alternative.")
    p.add_argument("--n-pebbles", "--N", type=int, help="Exact number of pebbles; no fixed default.")
    p.add_argument("--height", "--H", type=float, help="Maximum core height. Without N, use overfill/settle/cut filling.")
    p.add_argument("--load-fraction", type=float, default=.2, help="Top force / total bed weight; 0 skips the plate.")
    p.add_argument("--clearance", type=float, help="Minimum surface gap in INPUT length units; default 1e-5 times diameter.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--insertion", choices=["pour", "cloud"], default="pour", help="pour: release individual pebbles from one fixed point; cloud: settle an initial random cloud.")
    p.add_argument("--inlet", type=float, nargs=3, action="append", metavar=("X", "Y", "Z"), help="Exact fixed release point in input length units; repeat for several points, each feeding independently when clear. Default: one central point at an automatically chosen fixed height.")
    p.add_argument("--pour-interval", type=float, default=.5, help="Minimum numerical time between releases; insertion also waits until the point is clear.")
    p.add_argument("--inlet-jitter", type=float, default=.03, help="Std of seeded horizontal release velocity in sqrt(g*d_p); positions remain identical. Set 0 to remove variation.")
    p.add_argument("--inlet-vz", type=float, default=0., help="Initial downward/zero release velocity in sqrt(g*d_p).")
    p.add_argument("--wait-until-settled", action="store_true", help="Wait for all current pebbles to settle before the next release.")
    p.add_argument("--max-pour-time", type=float, help="Numerical time limit for pouring; default scales with the requested count.")
    p.add_argument("--inlet-timeout", type=float, default=30., help="Maximum continuous time any fixed inlet may remain blocked.")
    p.add_argument("--backend", choices=["auto", "numpy", "numba"], default="auto", help="auto uses optional Numba when available.")
    p.add_argument("--threads", type=int, default=1, help="Numba contact threads; parallel contacts activate at N>=2048. Benchmark 1/2/4/6 to choose the fastest.")
    p.add_argument("--kn", type=float, default=50000.)
    p.add_argument("--mu", type=float, default=.4)
    p.add_argument("--mu-wall", type=float, default=.4)
    p.add_argument("--restitution", type=float, default=.35)
    p.add_argument("--drag", type=float, default=.3)
    p.add_argument("--dt-scale", type=float, default=1.)
    p.add_argument("--plate-drag", type=float, default=20.)
    p.add_argument("--plate-speed", type=float, default=.03)
    p.add_argument("--ramp-time", type=float, default=10.)
    p.add_argument("--max-settle-time", type=float, default=120.)
    p.add_argument("--max-hold-time", type=float, default=100.)
    p.add_argument("--speed-tol", type=float, default=1e-3)
    p.add_argument("--force-tol", type=float, default=1e-2)
    p.add_argument("--quiet-time", type=float, default=.5)
    p.add_argument("--units", choices=["input", "diameter"], default="input")
    p.add_argument("--origin", choices=["bottom", "center", "centroid"], default="bottom")
    p.add_argument("--format", choices=["xyz", "serpent"], default="xyz")
    p.add_argument("--universe", default="pebble")
    p.add_argument("-o", "--output", default="pebbles_dem.dat", help="Output filename or prefix; writes DAT/CSV/VTK/JSON.")
    p.add_argument("--save-native", action="store_true", help="Also save explicitly labeled soft-contact native CSV.")
    p.add_argument("--finalize-only", type=Path, help="Skip DEM; clearance-correct an existing CSV/XYZ file with floor z=0.")
    p.add_argument("--porosity-plot", type=Path, help="Optional radial porosity PNG; requires matplotlib.")
    p.add_argument("--porosity-csv", type=Path, help="Optional radial porosity CSV.")
    p.add_argument("--end-trim", type=float, default=1., help="Exclude this many diameters from both ends for radial porosity.")
    p.add_argument("--de-klerk-mean", type=float, help="Optional de Klerk curve with this prescribed cylindrical mean.")
    p.add_argument("--video", type=Path, help="Record the DEM run and create this MP4; requires Pillow and FFmpeg/imageio-ffmpeg.")
    p.add_argument("--fps", type=int, default=30, help="MP4 playback frames per second.")
    p.add_argument("--video-interval", type=float, default=.1, help="Numerical time between recorded snapshots; smaller gives slower/smoother playback.")
    p.add_argument("--video-max-frames", type=int, default=600, help="Bound trajectory frames in RAM; older frames are thinned to fit this budget (>=32).")
    p.add_argument("--video-width", type=int, default=1280)
    p.add_argument("--video-height", type=int, default=720)
    p.add_argument("--plate-opacity", type=float, default=.22, help="Video plate opacity: 0=outline, 1=opaque. Does not change plate forces.")
    p.add_argument("--snapshots", type=Path, help="Directory for PNG and VTK snapshots; written during the run, without keeping a trajectory.")
    p.add_argument("--snapshot-every", type=int, default=1000, help="Snapshot every this many released pebbles, plus first/last release and stage boundaries.")
    p.add_argument("--history-interval", type=float, default=1., help="Numerical interval for streamed history rows; releases and convergence events are also saved.")
    p.add_argument("--checkpoint", type=Path, help="Atomic restart NPZ; saved periodically and after stages. Ctrl+C requests a safe checkpoint and pause.")
    p.add_argument("--checkpoint-interval", type=float, default=300., help="Elapsed wall-clock seconds between checkpoints.")
    p.add_argument("--resume", type=Path, help="Resume a checkpoint, restoring its original arguments unless explicitly overridden.")
    p.add_argument("--log-file", type=Path, help="Progress text log; default <output-stem>.log, flushed during the run.")
    p.add_argument("--log-interval", type=float, default=5., help="Numerical time between status lines; each particle release is also logged.")
    p.add_argument("--self-test", action="store_true", help="Run force and geometry checks; no geometry arguments needed.")
    p.add_argument("--benchmark", action="store_true", help="Short synthetic contact benchmark at --N/--R; prints timings and rough pouring core-time estimate without making a bed.")
    p.add_argument("--quiet", action="store_true")
    raw_args = list(sys.argv[1:] if argv is None else argv)
    a = p.parse_args(raw_args)
    if a.self_test:
        print(json.dumps(self_test(a.backend,a.threads), indent=2))
        return
    if a.benchmark:
        radius=a.core_radius if a.core_radius is not None else (a.core_diameter/2 if a.core_diameter is not None else None)
        if radius is None or a.diameter<=0:p.error("--benchmark requires --R and a positive --dp")
        post_pour=a.max_settle_time+((2*a.ramp_time+a.max_hold_time+15.+a.max_settle_time) if a.load_fraction else 0.)
        benchmark_packing(a.n_pebbles,radius/a.diameter,threads=a.threads,backend=a.backend,kn=a.kn,drag=a.drag,
                          dt_scale=a.dt_scale,interval=a.pour_interval,velocity_z=a.inlet_vz,video_max_frames=a.video_max_frames,
                          inlets=len(a.inlet) if a.inlet else 1,post_pour_time=post_pour)
        return
    resumed_meta=None
    if a.resume:
        resumed_meta=checkpoint_metadata(a.resume)
        explicit=set()
        for token in raw_args:
            action=p._option_string_actions.get(token.split("=",1)[0])
            if action is not None: explicit.add(action.dest)
        current=vars(a).copy()
        for field,value in resumed_meta["command"].items():
            if field not in explicit and hasattr(a,field):
                action=next((v for v in p._actions if v.dest==field),None)
                setattr(a,field,Path(value) if value is not None and action and action.type is Path else value)
        a.resume=Path(current["resume"])
        if "checkpoint" not in explicit: a.checkpoint=a.resume
    prefix = output_prefix(a.output)
    log_path = a.log_file if a.log_file is not None else Path(str(prefix)+".log")
    continuing_history=bool(a.resume) and output_prefix(resumed_meta["command"].get("output",a.output)).resolve()==prefix.resolve()
    if continuing_history:
        trim_history_for_resume(str(prefix)+"_history.csv",resumed_meta["scalars"]["elapsed"],resumed_meta["scalars"]["total_steps"])
    with ProgressLog(log_path, echo=not a.quiet, append=bool(a.resume),
                     elapsed_before=0. if resumed_meta is None else resumed_meta.get("elapsed_wall_seconds",0.)) as logger, HistoryStream(
            str(prefix)+"_history.csv",a.history_interval,append=continuing_history) as history:
        logger.write("Configuration: "+json.dumps(vars(a), default=str, sort_keys=True))
        run_command(a, p, logger, history)


def run_command(a, p, logger, history):
    if not math.isfinite(a.plate_opacity) or not 0 <= a.plate_opacity <= 1:
        p.error("--plate-opacity must be between 0 and 1")
    R = a.core_radius if a.core_radius is not None else (a.core_diameter/2 if a.core_diameter is not None else None)
    if R is None:
        p.error("give --R/--core-radius or --core-diameter")
    if not np.all(np.isfinite([R,a.diameter])) or a.diameter<=0 or R<=a.diameter/2:
        p.error("give a positive diameter and cylinder radius larger than one sphere radius")
    if a.end_trim < 0 or not math.isfinite(a.end_trim):
        p.error("--end-trim must be finite and nonnegative")
    recorder = None
    if a.video:
        if a.finalize_only:
            p.error("A packing video needs a new DEM run; --finalize-only has no motion history.")
        if (a.video.suffix.lower() != ".mp4" or a.fps <= 0 or a.video_width < 640
                or a.video_height < 480 or a.video_width % 2 or a.video_height % 2):
            p.error("Video requires an .mp4 path, positive FPS, and even dimensions >= 640x480.")
        video_dependencies()
        recorder = RecordingBundle(video=PackingRecorder(a.video_interval, a.video_max_frames))
        logger.write(f"[video] Maximum stored trajectory frames={a.video_max_frames}; recording thins automatically as needed.")
    if a.snapshots:
        stages = (["pouring"] if a.insertion == "pour" else [])+["gravity_settle"]
        if a.height is not None and a.n_pebbles is None: stages.append("settle_after_height_cut")
        if a.load_fraction:
            stages += ["loading_ramp", "loaded_hold", "unloading_ramp", "plate_retraction", "unloaded_settle"]
        stages.append("clearance_verified")
        snapshots = SnapshotRecorder(a.snapshots, R/a.diameter, a.diameter, a.snapshot_every,
                                      a.plate_opacity, stages, logger)
        if recorder is None: recorder = RecordingBundle()
        recorder.snapshots = snapshots
    checkpoint=None
    if a.checkpoint:
        command={key:(str(value.resolve()) if isinstance(value,Path) else value) for key,value in vars(a).items()}
        command["output"]=str(Path(a.output).resolve())
        checkpoint=CheckpointManager(a.checkpoint,a.checkpoint_interval,resume=a.resume,command=command,
                                     logger=logger,recorder=recorder,history=history)
        def pause_requested(*_):
            checkpoint.stop_requested=True
        signal.signal(signal.SIGINT,pause_requested)
    if a.finalize_only:
        D, gap = a.diameter, a.clearance if a.clearance is not None else 1e-5*a.diameter
        if not np.all(np.isfinite([D, R, gap])) or D <= 0 or gap < 0 or R <= D/2+gap:
            p.error("invalid diameter/radius/clearance")
        if a.height is not None and (not math.isfinite(a.height) or a.height <= D+2*gap):
            p.error("invalid maximum height")
        raw = read_centers(a.finalize_only)
        if a.n_pebbles is not None and len(raw) != a.n_pebbles:
            p.error("input particle count differs from --N")
        clean, cleanup = enforce_clearance(raw/D, R/D, gap/D, None if a.height is None else a.height/D)
        centers = clean*D
        report = dict(script_version=__version__, backend="finalize_only", actual_count=len(centers),
                      parameters=dict(pebble_diameter=D, core_radius=R, core_height=a.height,
                                      clearance=gap, requested_count=a.n_pebbles),
                      final_height=float(centers[:, 2].max()+D/2), geometry_cleanup=cleanup,
                      validation=validate_geometry(centers, D, R, gap, height=a.height),
                      stages=[], history=[], native_centers=raw,
                      note="Geometric cleanup of supplied positions; no DEM was run by --finalize-only.")
    else:
        centers, report = generate_pebbles(
            a.diameter, R, a.height, a.n_pebbles, clearance=a.clearance,
            load_fraction=a.load_fraction, seed=a.seed, backend=a.backend,
            kn=a.kn, mu=a.mu, mu_wall=a.mu_wall, restitution=a.restitution, drag=a.drag,
            dt_scale=a.dt_scale, plate_drag=a.plate_drag, plate_speed=a.plate_speed,
            ramp_time=a.ramp_time, max_settle_time=a.max_settle_time, max_hold_time=a.max_hold_time,
            speed_tol=a.speed_tol, force_tol=a.force_tol, quiet_time=a.quiet_time,
            verbose=not a.quiet, return_report=True, recorder=recorder,
            insertion=a.insertion, inlet=None if not a.inlet else (a.inlet[0] if len(a.inlet) == 1 else a.inlet),
            pour_interval=a.pour_interval,
            inlet_jitter=a.inlet_jitter, inlet_vz=a.inlet_vz, wait_until_settled=a.wait_until_settled,
            max_pour_time=a.max_pour_time, inlet_timeout=a.inlet_timeout,
            logger=logger, log_interval=a.log_interval, history=history, checkpoint=checkpoint, threads=a.threads)
        report["history_file"] = str(history.path)
    exported, prefix, verified, D, R = write_final(centers, report, a.output, units=a.units,
                                                   origin=a.origin, file_format=a.format,
                                                   universe=a.universe, save_native=a.save_native)
    minimum = verified["minimum_center_distance"]
    text = f"{minimum:.14g}" if minimum is not None else "n/a (one particle)"
    logger.write(f"VALIDATED: N={len(exported)}, minimum center distance={text}, "
          f"required={verified['required_center_distance']:.14g}")
    logger.write(f"Wrote {prefix}.dat, .csv, .vtk, .json; DAT columns: "
          +( "x y z radius universe" if a.format == "serpent" else "x y z"))
    if a.video:
        parameters = report["parameters"]
        movie = write_packing_video(recorder.video, parameters["core_radius"]/parameters["pebble_diameter"], a.video,
                                    fps=a.fps, width=a.video_width, height=a.video_height,
                                    verbose=not a.quiet, plate_opacity=a.plate_opacity, logger=logger)
        json_path = Path(str(prefix)+".json")
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        metadata["video"] = movie
        with tempfile.TemporaryDirectory(prefix=".video_meta_", dir=json_path.parent) as directory:
            staged = Path(directory)/"metadata.json"
            staged.write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            staged.replace(json_path)
        logger.write(f"Wrote {a.video}: {movie['duration_seconds']:.1f} seconds at {a.fps} FPS.")
    if a.porosity_plot or a.porosity_csv:
        radial, epsilon, mean, window = radial_porosity(exported, D, R, trim=a.end_trim)
        if a.porosity_csv:
            a.porosity_csv.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(a.porosity_csv, np.column_stack((radial, R-radial, epsilon)), delimiter=",",
                       header="r,wall_distance,porosity", comments="", fmt="%.17g")
        if a.porosity_plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 5))
            order = np.argsort(R-radial)
            ax.plot((R-radial[order])/D, epsilon[order], label=f"DEM bed; mean={mean:.4f}", lw=1.8)
            if a.de_klerk_mean is not None:
                asymptote = de_klerk_asymptote(R/D, a.de_klerk_mean)
                s = np.linspace(0., R/D, 1600)
                ax.plot(s, de_klerk(s, asymptote), "--", label=f"de Klerk; prescribed mean={a.de_klerk_mean:.4f}")
            ax.set(xlabel="Distance from wall / pebble diameter", ylabel="Radial porosity",
                   xlim=(0., R/D), ylim=(0., 1.02), title=f"{len(exported)} pebbles; R/d={R/D:g}")
            ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
            a.porosity_plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(a.porosity_plot, dpi=200); plt.close(fig)
        logger.write(f"Radial porosity mean={mean:.6f}; axial window={window} in output coordinates.")
    logger.write(f"RUN COMPLETE | particles={len(exported)} | t_total*={report.get('total_simulation_time', 'not simulated')} "
                 f"| total_steps={report.get('total_steps', 'not simulated')} | wall_seconds={time.perf_counter()-logger.started:.2f}")


if __name__ == "__main__":
    try:
        main()
    except PackingPaused as pause:
        print(str(pause),flush=True)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)

