#!/usr/bin/env python3
"""
pebble_pack.py - random pebble-bed packing by gravity deposition ("drop and roll").

Pebbles are dropped one at a time at a random (x, y) inside a cylinder. Each pebble
falls straight down until it touches the floor or another pebble, then rolls under
gravity (steepest descent on the active contact surfaces: other pebbles, the
cylinder wall, the floor) until it reaches a stable resting position: the floor,
or a position where gravity is balanced by compressive contact forces from 2-3
supports. This is the Visscher-Bolsterli (1972) sequential deposition model.

It gives the same kind of bed as pebbles.dat (flat ordered bottom layer, wall
ordering near the wall, same radial/axial packing oscillations), and pebbles never
overlap (center distance >= pebble diameter).

Pure drop-and-roll gives a bulk packing fraction of ~0.58. A real pour (or a DEM
simulation like the one behind pebbles.dat) settles a bit more, so by default a
short gravity-"tapping" compaction follows the deposition, bringing the bulk value
to ~0.595 (pebbles.dat: 0.594). Tune it with compaction= (number of tapping
cycles). Taller beds compact more per cycle, so calibrate per geometry:
    bed like pebbles.dat (R = 6.5 D, ~11 D tall):  0 -> 0.578, 70 -> 0.596, 120 -> 0.61
    D = 6, R = 39.42, H = 84 cm (filled):         40 -> 0.597, 70 -> 0.617

Usage as a module
-----------------
    from pebble_pack import generate_pebbles
    centers = generate_pebbles(pebble_diameter=6.0,
                               core_radius=6.57*6.0,
                               core_height=84.0,
                               compaction=40,
                               seed=1)           # (N, 3) array, cm
    # fixed number of pebbles instead of filling to the height:
    centers = generate_pebbles(6.0, 6.57*6.0, n_pebbles=1568, seed=1)

Usage from the command line
---------------------------
    python pebble_pack.py --diameter 6 --core-radius 39.42 --height 84 --compaction 40 -o pebbles_new.dat
    python pebble_pack.py --diameter 6 --core-radius 39 --n-pebbles 1568 \
        --units diameter --origin centroid -o like_original.dat
    python pebble_pack.py ... --format serpent --universe fuel_pebble   # Serpent pbed file
    python pebble_pack.py ... --porosity-plot porosity.png --porosity-csv porosity.csv
    python pebble_pack.py --input pebbles_new.dat --diameter 6 --core-radius 130 \
        --porosity-plot porosity.png                 # porosity of an existing bed

Geometry convention: cylinder axis is z, axis at x = y = 0, floor at z = 0
(change with --origin). Pebble centers satisfy  r <= R - D/2  and  D/2 <= z <= H - D/2.
"""

import argparse
import itertools
import math
import sys
import time

import numpy as np

FLOOR, WALL = -1, -2


class DropAndRollPacker:
    """Sequential gravity deposition. Internally everything is in units of D (D = 1)."""

    def __init__(self, core_radius_D, core_height_D=None, seed=None,
                 gap=1e-6, step=0.05):
        if core_radius_D <= 0.5:
            raise ValueError("Core radius must be larger than the pebble radius.")
        self.Rw = core_radius_D - 0.5                 # max radial position of a center
        self.zfloor = 0.5                              # min center height (floor at z = 0)
        self.ztop = None if core_height_D is None else core_height_D - 0.5
        self.dc = 1.0 + gap                            # contact distance (tiny clearance)
        self.step = step                               # rolling step, in diameters
        self.cs = 1.25                                 # neighbour-grid cell size (> dc + step)
        self.rng = np.random.default_rng(seed)
        self.C = np.empty((1024, 3))
        self.n = 0
        self.grid3 = {}
        self.grid2 = {}
        self.zmax = 0.0
        self.touch_tol = 1e-8

    # ------------------------------------------------------------------ storage
    def _add(self, P):
        if self.n == len(self.C):
            self.C = np.vstack([self.C, np.empty_like(self.C)])
        self.C[self.n] = P
        i, j, k = (int(math.floor(v / self.cs)) for v in P)
        self.grid3.setdefault((i, j, k), []).append(self.n)
        self.grid2.setdefault((i, j), []).append(self.n)
        self.zmax = max(self.zmax, P[2])
        self.n += 1

    def _neighbours(self, P):
        i, j, k = (int(math.floor(v / self.cs)) for v in P)
        idx = []
        g = self.grid3
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    lst = g.get((i + di, j + dj, k + dk))
                    if lst:
                        idx.extend(lst)
        return np.asarray(idx, dtype=int)

    def _column(self, P):
        i, j = int(math.floor(P[0] / self.cs)), int(math.floor(P[1] / self.cs))
        idx = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                lst = self.grid2.get((i + di, j + dj))
                if lst:
                    idx.extend(lst)
        return np.asarray(idx, dtype=int)

    # ------------------------------------------------------------- constraints
    def _gaps(self, P, cons):
        """Gap function g (>= 0 means no overlap) for each constraint id."""
        cons = np.asarray(cons, dtype=int)
        out = np.empty(len(cons))
        peb = cons >= 0
        if peb.any():
            out[peb] = np.sqrt(((self.C[cons[peb]] - P) ** 2).sum(axis=1)) - self.dc
        out[cons == FLOOR] = P[2] - self.zfloor
        out[cons == WALL] = self.Rw - math.hypot(P[0], P[1])
        return out

    def _normal(self, P, c):
        if c == FLOOR:
            return np.array([0.0, 0.0, 1.0])
        if c == WALL:
            r = math.hypot(P[0], P[1])
            return np.array([-P[0] / r, -P[1] / r, 0.0])
        v = P - self.C[c]
        return v / np.linalg.norm(v)

    def _project(self, P, S):
        """Bring P back onto all equality constraints in S (Gauss-Newton)."""
        P = P.copy()
        if not S:
            return P
        if len(S) == 1 and S[0] >= 0:                    # exact for one sphere
            v = P - self.C[S[0]]
            return self.C[S[0]] + self.dc * v / math.sqrt(v @ v)
        for _ in range(30):
            g = np.empty(len(S))
            N = np.empty((len(S), 3))
            for m, c in enumerate(S):
                if c == FLOOR:
                    g[m] = P[2] - self.zfloor
                    N[m] = (0.0, 0.0, 1.0)
                elif c == WALL:
                    r = math.hypot(P[0], P[1])
                    g[m] = self.Rw - r
                    N[m] = (-P[0] / r, -P[1] / r, 0.0)
                else:
                    v = P - self.C[c]
                    L = math.sqrt(v @ v)
                    g[m] = L - self.dc
                    N[m] = v / L
            if np.abs(g).max() < 1e-13:
                break
            lam = np.linalg.lstsq(N @ N.T, g, rcond=None)[0]
            P -= N.T @ lam
        return P

    def _descent(self, P, A):
        """Project gravity onto the tangent cone of the active contacts A.
        Returns (direction d, load-bearing contacts S). d = 0 means stable."""
        g = np.array([0.0, 0.0, -1.0])
        N_all = {c: self._normal(P, c) for c in A}
        for size in range(len(A) + 1):
            for S in itertools.combinations(A, size):
                if size == 0:
                    d, mu = g, np.zeros(0)
                else:
                    N = np.array([N_all[c] for c in S])
                    mu = np.linalg.lstsq(N @ N.T, -(N @ g), rcond=None)[0]
                    d = g + N.T @ mu
                if np.any(mu < -1e-10):
                    continue
                if all(N_all[c] @ d >= -1e-10 for c in A if c not in S):
                    return d, [c for c, m in zip(S, mu) if m > 1e-12] or list(S)
        return np.zeros(3), list(A)            # degenerate -> treat as stable

    # --------------------------------------------------------------- dynamics
    def _free_fall(self, P):
        """Drop vertically; return new position and the contact that stops it."""
        idx = self._column(P)
        z_hit, hit = self.zfloor, FLOOR
        if len(idx):
            Cj = self.C[idx]
            dh2 = (Cj[:, 0] - P[0]) ** 2 + (Cj[:, 1] - P[1]) ** 2
            ok = (dh2 < self.dc ** 2) & (Cj[:, 2] < P[2] - 1e-12)
            if np.any(ok):
                zc = Cj[ok, 2] + np.sqrt(self.dc ** 2 - dh2[ok])
                m = np.argmax(zc)
                if zc[m] > z_hit:
                    z_hit, hit = zc[m], idx[ok][m]
        Q = P.copy()
        Q[2] = min(z_hit, P[2])
        return Q, hit

    def drop_one(self, max_iter=20000):
        """Drop one pebble and return its resting position (not yet added)."""
        r = self.Rw * math.sqrt(self.rng.random())
        th = 2 * math.pi * self.rng.random()
        P = np.array([r * math.cos(th), r * math.sin(th), self.zmax + 2.0])
        P, hit = self._free_fall(P)
        if hit == FLOOR:
            return P
        for _ in range(max_iter):
            nb = self._neighbours(P)
            cons = [FLOOR, WALL] + nb.tolist()
            gaps = self._gaps(P, cons)
            A = [c for c, gv in zip(cons, gaps) if gv < self.touch_tol]
            if FLOOR in A:
                return P
            d, S = self._descent(P, A)
            nd = np.linalg.norm(d)
            if nd < 1e-10:
                return P                                  # stable on 2-3 supports
            if not S:                                     # nothing carries load: fall
                P, hit = self._free_fall(P)
                if hit == FLOOR:
                    return P
                continue
            u = d / nd
            others = [c for c in cons if c not in S]
            h = self.step
            moved = False
            while h > 1e-10:
                Pn = self._project(P + h * u, S)
                if Pn[2] >= P[2] - 1e-14:                 # no descent: shorter step
                    h *= 0.5
                    continue
                go = self._gaps(Pn, others)
                bad = [c for c, gv in zip(others, go) if gv < 0]
                if not bad:
                    P, moved = Pn, True
                    break
                already = [c for c in bad if c in A]      # tangent contacts stay on
                if already:
                    S = S + already
                    others = [c for c in others if c not in already]
                    continue
                lo, hi = 0.0, h                           # find the new contact
                for _ in range(32):
                    mid = 0.5 * (lo + hi)
                    Pm = self._project(P + mid * u, S)
                    if np.any(self._gaps(Pm, others) < 0):
                        hi = mid
                    else:
                        lo = mid
                Pn = self._project(P + lo * u, S)
                if Pn[2] < P[2] - 1e-14 and np.all(self._gaps(Pn, others) >= -1e-12):
                    P, moved = Pn, True
                break
            if not moved:
                return P                                   # cannot descend further
        return P

    # ---------------------------------------------------------------- compaction
    def _rebuild(self):
        P = self.C[:self.n].copy()
        self.n, self.grid3, self.grid2, self.zmax = 0, {}, {}, 0.0
        for p in P:
            self._add(p)

    def _relax(self, P, d0, zf, zl, iters):
        """Push overlapping pebbles apart (Jacobi), then clamp to floor/wall/lid."""
        from scipy.spatial import cKDTree
        for _ in range(iters):
            pr = cKDTree(P).query_pairs(d0, output_type="ndarray")
            if len(pr):
                v = P[pr[:, 0]] - P[pr[:, 1]]
                L = np.linalg.norm(v, axis=1)
                L[L == 0] = 1e-12
                dv = v * (0.5 * (d0 - L) / L)[:, None]
                disp = np.zeros_like(P)
                np.add.at(disp, pr[:, 0], dv)
                np.add.at(disp, pr[:, 1], -dv)
                P += disp
            P[:, 2] = np.maximum(P[:, 2], zf)
            if zl is not None:
                P[:, 2] = np.minimum(P[:, 2], zl)
            r = np.hypot(P[:, 0], P[:, 1])
            m = r > self.Rw
            P[m, :2] *= (self.Rw / r[m])[:, None]
        return P

    def compact(self, cycles=120, gravity=0.02, noise=0.05, eps=3e-3, inner=15,
                verbose=True):
        """Gravity 'tapping': repeated small downward moves + random kicks with
        overlap removal, both annealed to zero (gravity ~ sqrt(1-x), kicks ~ (1-x)^2).
        Mimics the extra settling a real pour gets (DEM-like beds), raising the
        bulk packing fraction from ~0.58 to ~0.60-0.61. Pebbles are temporarily
        inflated by eps so the final cleanup can guarantee zero overlap at the
        true diameter."""
        from scipy.spatial import cKDTree
        if self.n < 2 or cycles == 0:
            return
        P = self.C[:self.n].copy()
        d0 = 1.0 + eps
        zf = self.zfloor                                  # walls/floor/lid: true radius
        zl = self.ztop
        for c in range(cycles):
            x = c / cycles
            P[:, 2] -= gravity * math.sqrt(1.0 - x)
            P += noise * (1.0 - x) ** 2 * self.rng.standard_normal(P.shape)
            P = self._relax(P, d0, zf, zl, inner)
        for k in range(500):                              # cleanup, no gravity
            P = self._relax(P, d0, zf, zl, 5)
            if cKDTree(P).query(P, k=2)[0][:, 1].min() >= 1.0 + 1e-9:
                break
        else:
            raise RuntimeError("Compaction cleanup did not remove all overlaps.")
        self.C[:self.n] = P
        self._rebuild()

    # ------------------------------------------------------------------ filling
    def fill(self, n_pebbles=None, max_rejects=200, verbose=True):
        t0, rejects = time.time(), 0
        while True:
            if n_pebbles is not None and self.n >= n_pebbles:
                break
            P = self.drop_one()
            if self.ztop is not None and P[2] > self.ztop + 1e-9:
                if n_pebbles is not None:
                    raise RuntimeError(
                        f"Only {self.n} pebbles fit below the core height; "
                        f"{n_pebbles} requested.")
                rejects += 1
                if rejects >= max_rejects:
                    break
                continue
            rejects = 0
            self._add(P)
            if verbose and self.n % 500 == 0:
                print(f"  {self.n} pebbles, bed top {self.zmax:.2f} D, "
                      f"{time.time() - t0:.1f} s", file=sys.stderr)
        return self.C[:self.n].copy()


def generate_pebbles(pebble_diameter, core_radius, core_height=None, n_pebbles=None,
                     seed=None, compaction=70, verbose=True):
    """Return an (N, 3) array of pebble centers, same length unit as the inputs.

    Cylinder axis along z through x = y = 0; floor at z = 0.
      core_height only : fill the cylinder up to the top (the top acts as a lid;
                         pebbles are added until no more fit).
      n_pebbles only   : pour exactly that many pebbles (free top surface).
      both             : pour n_pebbles, error if they do not fit below the top.
    compaction = number of gravity-tapping cycles after deposition:
        0 -> pure drop-and-roll, bulk packing ~0.58
        70 (default) -> ~0.595 for a bed like pebbles.dat (~11 D tall)
        Taller beds compact more per cycle (84 cm core: 40 -> 0.597, 70 -> 0.617).
    Check the printed bulk packing fraction and adjust."""
    compaction = 70 if compaction is True else int(compaction or 0)
    if core_height is None and n_pebbles is None:
        raise ValueError("Give core_height, n_pebbles, or both.")
    D = float(pebble_diameter)
    packer = DropAndRollPacker(core_radius / D,
                               None if core_height is None else core_height / D,
                               seed=seed)
    log = (lambda s: print(s, file=sys.stderr)) if verbose else (lambda s: None)

    if n_pebbles is not None:
        log("deposition (drop and roll)...")
        packer.fill(n_pebbles=n_pebbles, verbose=verbose)
        if compaction:
            log("compaction...")
            saved, packer.ztop = packer.ztop, None       # free top surface
            packer.compact(cycles=compaction, verbose=verbose)
            packer.ztop = saved
    else:
        log("deposition (drop and roll)...")
        packer.fill(verbose=verbose)
        if compaction:
            for rnd in range(4):                          # compact, top up, repeat
                n_before = packer.n
                log(f"compaction round {rnd + 1}...")
                packer.compact(cycles=compaction if rnd == 0 else max(1, compaction // 7),
                               verbose=verbose)
                packer.fill(verbose=False)
                added = packer.n - n_before
                log(f"  topped up with {added} pebbles")
                if added <= max(2, 0.002 * packer.n):
                    break

    C = packer.C[:packer.n].copy()
    if packer.ztop is not None and C[:, 2].max() > packer.ztop + 1e-9:
        raise RuntimeError("Pebbles exceed the core height.")
    return C * D


def packing_stats(centers, pebble_diameter, core_radius, core_height=None):
    """Packing fraction (overall and bulk) and minimum center distance."""
    from scipy.spatial import cKDTree
    D, R = pebble_diameter, core_radius
    z = centers[:, 2]
    H = core_height if core_height is not None else z.max() + D / 2 - (z.min() - D / 2)
    vp = math.pi * D ** 3 / 6
    pf = len(centers) * vp / (math.pi * R ** 2 * H)
    # bulk: centers > 2.5 D from the wall, floor and top surface
    r = np.hypot(centers[:, 0], centers[:, 1])
    zb, zt = z.min() + 2.5 * D, z.max() - 2.5 * D
    m = (r < R - 2.5 * D) & (z > zb) & (z < zt)
    bulk = m.sum() * vp / (math.pi * (R - 2.5 * D) ** 2 * (zt - zb)) if zt > zb else float("nan")
    dmin = cKDTree(centers).query(centers, k=2)[0][:, 1].min()
    return {"n": len(centers), "packing_fraction": pf, "bulk_packing_fraction": bulk,
            "min_center_distance": dmin}


# ------------------------------------------------------------------ porosity
def _lens_area(rho, s, b):
    """Overlap area of a circle of radius rho at the origin and a circle of radius b
    centered at distance s (broadcast arrays)."""
    rho, s, b = np.broadcast_arrays(rho, s, b)
    out = np.zeros(rho.shape)
    inside = s <= np.abs(rho - b)
    out[inside] = math.pi * np.minimum(rho, b)[inside] ** 2
    m = ~inside & (s < rho + b)
    r_, s_, b_ = rho[m], s[m], b[m]
    c1 = np.clip((s_ ** 2 + r_ ** 2 - b_ ** 2) / (2 * s_ * r_), -1.0, 1.0)
    c2 = np.clip((s_ ** 2 + b_ ** 2 - r_ ** 2) / (2 * s_ * b_), -1.0, 1.0)
    k = (-s_ + r_ + b_) * (s_ + r_ - b_) * (s_ - r_ + b_) * (s_ + r_ + b_)
    out[m] = r_ ** 2 * np.arccos(c1) + b_ ** 2 * np.arccos(c2) - 0.5 * np.sqrt(np.maximum(k, 0.0))
    return out


def radial_porosity(centers, pebble_diameter, core_radius, dr=None, zrange=None, nq=48):
    """Azimuthally averaged porosity (void fraction) in annular shells r..r+dr.

    Exact geometry: the pebble volume inside a cylinder r <= rho is integrated over z
    (Gauss-Legendre, nq points) of the circle-circle overlap area, and shell volumes
    come from differences of consecutive cylinders. Pebbles are clipped to zrange
    (default: the bed extent, lowest pebble bottom to highest pebble top).
    Returns (r_mid, porosity, mean_porosity)."""
    D, R = float(pebble_diameter), float(core_radius)
    a = D / 2
    dr = D / 20 if dr is None else dr
    edges = np.linspace(0.0, R, int(math.ceil(R / dr)) + 1)
    s = np.hypot(centers[:, 0], centers[:, 1])
    zc = centers[:, 2]
    zlo, zhi = (zc.min() - a, zc.max() + a) if zrange is None else zrange
    t0 = np.maximum(-a, zlo - zc)                       # sphere slice range, local z
    t1 = np.minimum(a, zhi - zc)
    keep = t1 > t0
    s, t0, t1 = s[keep], t0[keep], t1[keep]
    vfull = math.pi * (a * a * (t1 - t0) - (t1 ** 3 - t0 ** 3) / 3)
    x, w = np.polynomial.legendre.leggauss(nq)
    half, mid = 0.5 * (t1 - t0), 0.5 * (t1 + t0)
    V = np.zeros(len(edges))
    for k, rho in enumerate(edges[1:], 1):
        full = s + a <= rho
        part = ~full & (s - a < rho)
        Vk = vfull[full].sum()
        if part.any():
            t = mid[part, None] + half[part, None] * x
            b = np.sqrt(np.maximum(a * a - t * t, 0.0))
            Vk += (half[part] * (_lens_area(rho, s[part, None], b) @ w)).sum()
        V[k] = Vk
    shell = math.pi * np.diff(edges ** 2) * (zhi - zlo)
    porosity = 1.0 - np.diff(V) / shell
    mean = 1.0 - V[-1] / (math.pi * R ** 2 * (zhi - zlo))
    return 0.5 * (edges[1:] + edges[:-1]), porosity, mean


def plot_radial_porosity(r, porosity, core_radius, path, mean=None, title=None):
    """Plot porosity vs radial position from -R to R (profile mirrored about the axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.concatenate([-r[::-1], r])
    y = np.concatenate([porosity[::-1], porosity])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, y, lw=0.9, color="tab:blue", label="radial porosity")
    if mean is not None:
        ax.axhline(mean, ls="--", lw=1, color="tab:red", label=f"mean = {mean:.4f}")
    for xw in (-core_radius, core_radius):
        ax.axvline(xw, color="k", lw=1)
    ax.set_xlim(-core_radius * 1.02, core_radius * 1.02)
    ax.set_ylim(0, 1)
    ax.set_xlabel("radial position")
    ax.set_ylabel("porosity (void fraction)")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--diameter", type=float, required=True, help="pebble diameter")
    ap.add_argument("--core-radius", type=float, help="core (cylinder) radius")
    ap.add_argument("--core-diameter", type=float, help="core diameter (alternative)")
    ap.add_argument("--height", type=float, help="core height to fill")
    ap.add_argument("--n-pebbles", type=int, help="number of pebbles to pour")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--compaction", type=int, default=70,
                    help="gravity-tapping cycles after deposition: 0 = pure "
                         "drop-and-roll (bulk ~0.58); 70 (default) ~0.595 for an "
                         "~11 D tall bed like pebbles.dat; taller beds compact more "
                         "(84 cm core: 40 -> 0.597)")
    ap.add_argument("--units", choices=["input", "diameter"], default="input",
                    help="'input': same unit as --diameter (e.g. cm); "
                         "'diameter': normalised by the pebble diameter, like pebbles.dat")
    ap.add_argument("--origin", choices=["bottom", "center", "centroid"], default="bottom",
                    help="z = 0 at the floor, at mid-height of the core, or at the "
                         "pebble centroid (pebbles.dat uses centroid)")
    ap.add_argument("--format", choices=["xyz", "serpent"], default="xyz",
                    help="'xyz': x y z per line; 'serpent': x y z r universe (pbed)")
    ap.add_argument("--universe", default="pebble", help="universe name for --format serpent")
    ap.add_argument("-o", "--output", default="pebbles_generated.dat")
    ap.add_argument("--input", help="analyse an existing x y z file (same unit as "
                                    "--diameter) instead of generating a bed")
    ap.add_argument("--porosity-plot", help="save radial porosity plot (-R..R) to this image")
    ap.add_argument("--porosity-csv", help="save radial porosity profile (r, porosity) to this file")
    ap.add_argument("--porosity-dr", type=float, help="radial shell width (default D/20)")
    ap.add_argument("--zrange", type=float, nargs=2, metavar=("ZLO", "ZHI"),
                    help="axial range for the porosity (default: bed extent)")
    a = ap.parse_args()

    R = a.core_radius if a.core_radius else (a.core_diameter / 2 if a.core_diameter else None)
    if R is None:
        ap.error("give --core-radius or --core-diameter")
    D = a.diameter

    t0 = time.time()
    if a.input:
        P = np.loadtxt(a.input, usecols=(0, 1, 2))
    else:
        P = generate_pebbles(D, R, a.height, a.n_pebbles, seed=a.seed,
                             compaction=a.compaction)
    st = packing_stats(P, D, R, a.height)
    print(f"{st['n']} pebbles in {time.time() - t0:.1f} s | packing fraction "
          f"{st['packing_fraction']:.4f} (bulk {st['bulk_packing_fraction']:.4f}, "
          f"porosity {1 - st['packing_fraction']:.4f}) | "
          f"min center distance / D = {st['min_center_distance'] / D:.6f}", file=sys.stderr)

    if a.porosity_plot or a.porosity_csv:
        r, eps, mean = radial_porosity(P, D, R, dr=a.porosity_dr, zrange=a.zrange)
        print(f"radial porosity: {len(r)} shells, mean porosity {mean:.4f}", file=sys.stderr)
        if a.porosity_csv:
            np.savetxt(a.porosity_csv, np.column_stack([r, eps]), fmt="%.8f",
                       delimiter=",", header="r,porosity", comments="")
            print(f"wrote {a.porosity_csv}", file=sys.stderr)
        if a.porosity_plot:
            plot_radial_porosity(r, eps, R, a.porosity_plot, mean=mean,
                                 title=f"{st['n']} pebbles, D = {D:g}, R = {R:g}")
            print(f"wrote {a.porosity_plot}", file=sys.stderr)
    if a.input:
        return

    if a.origin == "center":
        H = a.height if a.height is not None else P[:, 2].max() + D / 2
        P[:, 2] -= H / 2
    elif a.origin == "centroid":
        P[:, 2] -= P[:, 2].mean()
    scale = 1.0 / D if a.units == "diameter" else 1.0
    P = P * scale
    with open(a.output, "w") as f:
        for x, y, z in P:
            if a.format == "serpent":
                f.write(f"{x:.11f} {y:.11f} {z:.11f} {D / 2 * scale:.11f} {a.universe}\n")
            else:
                f.write(f"{x:.11f}\t{y:.11f}\t{z:.11f}\n")
    print(f"wrote {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
