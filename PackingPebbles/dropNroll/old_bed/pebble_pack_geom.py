#!/usr/bin/env python3
"""
pebble_pack_geom.py - random pebble-bed packing by gravity deposition ("drop and roll")
inside a container of arbitrary shape, read from an OpenMC geometry.

Same algorithm as pebble_pack.py: pebbles are dropped one at a time at a random
(x, y), fall straight down until they touch a surface or another pebble, then roll
under gravity (steepest descent on the active contacts) until they reach a stable
resting position (Visscher-Bolsterli, 1972). A gravity-"tapping" compaction follows
(see pebble_pack.py for calibration of compaction=). Pebbles never overlap and never
cross a container surface. Only the pebble centers are generated.

The container is the region of one OpenMC cell, which must be an intersection of
half-spaces (no unions "|" or complements "~"). Supported surfaces:
    x-plane, y-plane, z-plane, plane, x-cylinder, y-cylinder, z-cylinder, sphere, z-cone
Gravity points along -z. The region must be closed below (pebbles need something to
land on). A z-plane with the region below it ("-top") acts as the fill level (lid):
the bed is filled until no more pebbles fit under it. Without one, give n_pebbles or
fill_level. Examples of containers:
    annular core (central control channel):  -outer_zcyl +hole_zcyl +floor -top
    core with a conical bottom (defuel chute): -zcyl -zcone +outlet_plane -top
    several off-axis control channels:        -outer +chan1 +chan2 +chan3 +floor -top

Usage as a module
-----------------
    from pebble_pack_geom import Container, generate_pebbles_in
    box = Container.from_xml("geometry.xml", cell_id=10)       # or:
    box = Container.from_openmc(-outer & +hole & +floor & -top)  # openmc.Region
    box = Container.cylinder(radius=130, height=200, hole_radius=20)
    centers = generate_pebbles_in(box, pebble_diameter=6.0, compaction=40, seed=1)
    centers = generate_pebbles_in(box, 6.0, n_pebbles=5000, seed=1)

Usage from the command line
---------------------------
    python pebble_pack_geom.py --geometry geometry.xml --cell 10 --diameter 6 \
        --compaction 40 -o pebbles_geom.dat
    python pebble_pack_geom.py --diameter 6 --core-radius 130 --height 200 \
        --hole-radius 20 -o pebbles_hole.dat               # built-in cylinder (+ hole)
    python pebble_pack_geom.py ... --porosity-plot porosity.png --porosity-csv porosity.csv
    python pebble_pack_geom.py --input pebbles_new.dat --diameter 6 --core-radius 130 \
        --porosity-plot porosity.png                       # analyse an existing bed

Coordinates are those of the geometry (same length unit as --diameter, normally cm).
"""

import argparse
import itertools
import math
import re
import sys
import time

import numpy as np

A = 0.5                     # pebble radius in internal units (pebble diameter = 1)
SUPPORTED = ("x-plane", "y-plane", "z-plane", "plane", "x-cylinder", "y-cylinder",
             "z-cylinder", "sphere", "z-cone")


# ------------------------------------------------------------------ boundaries
# Each boundary is one half-space of the container. dist(P) is the signed distance of
# point P to the surface (> 0 on the allowed side), gap(P) = dist - pebble radius
# (>= 0 means the pebble does not cross it), normal(P) is the unit gradient of dist,
# and fall_hit(P) is the height at which a pebble falling straight down from P first
# touches the surface (None if never). dist/normal/gap work on (3,) or (N, 3) arrays.
class _Boundary:
    floor = False           # horizontal floor: a pebble touching it is at rest
    lid = None              # z of a horizontal lid (region below it), else None

    def gap(self, P):
        return self.dist(P) - A

    def bounds(self):
        return {}


class _Plane(_Boundary):
    def __init__(self, a, b, c, d, side):
        n = np.array([a, b, c], dtype=float)
        L = np.linalg.norm(n)
        s = 1.0 if side == "+" else -1.0
        self.n, self.off = s * n / L, s * d / L
        self.floor = self.n[2] > 1 - 1e-12
        if self.n[2] < -1 + 1e-12:
            self.lid = -self.off

    def dist(self, P):
        return P @ self.n - self.off

    def normal(self, P):
        return np.broadcast_to(self.n, np.shape(P)).copy()

    def fall_hit(self, P):
        if self.n[2] <= 1e-12:
            return None
        return P[2] - self.gap(P) / self.n[2]

    def bounds(self):
        out = {}
        for ax in range(3):
            if abs(self.n[ax]) > 1 - 1e-12:
                v = self.off / self.n[ax]
                out[ax] = (v, np.inf) if self.n[ax] > 0 else (-np.inf, v)
        return out


class _Cylinder(_Boundary):
    def __init__(self, axis, c1, c2, r, side):
        self.ax = axis
        self.i, self.j = [k for k in range(3) if k != axis]
        self.c, self.r = (c1, c2), r
        self.s = -1.0 if side == "-" else 1.0          # dist = s (rho - r)

    def _rho(self, P):
        u, v = P[..., self.i] - self.c[0], P[..., self.j] - self.c[1]
        return u, v, np.hypot(u, v)

    def dist(self, P):
        return self.s * (self._rho(P)[2] - self.r)

    def normal(self, P):
        u, v, rho = self._rho(P)
        rho = np.maximum(rho, 1e-300)
        N = np.zeros(np.shape(P))
        N[..., self.i], N[..., self.j] = self.s * u / rho, self.s * v / rho
        return N

    def fall_hit(self, P):
        if self.ax == 2:
            return None
        u, re_ = P[self.i] - self.c[0], self.r + self.s * A   # j is z for x/y-cylinders
        if abs(u) >= re_:
            return None
        w, z0 = math.sqrt(re_ * re_ - u * u), self.c[1]
        if self.s < 0:
            return z0 - w
        return z0 + w if P[2] >= z0 else None

    def bounds(self):
        if self.s > 0:
            return {}
        return {self.i: (self.c[0] - self.r, self.c[0] + self.r),
                self.j: (self.c[1] - self.r, self.c[1] + self.r)}


class _Sphere(_Boundary):
    def __init__(self, x0, y0, z0, r, side):
        self.c, self.r = np.array([x0, y0, z0], dtype=float), r
        self.s = -1.0 if side == "-" else 1.0

    def dist(self, P):
        return self.s * (np.linalg.norm(P - self.c, axis=-1) - self.r)

    def normal(self, P):
        v = P - self.c
        return self.s * v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)

    def fall_hit(self, P):
        re_ = self.r + self.s * A
        d2 = (P[0] - self.c[0]) ** 2 + (P[1] - self.c[1]) ** 2
        if d2 >= re_ * re_:
            return None
        w = math.sqrt(re_ * re_ - d2)
        if self.s < 0:
            return self.c[2] - w
        return self.c[2] + w if P[2] >= self.c[2] else None

    def bounds(self):
        if self.s > 0:
            return {}
        return {ax: (self.c[ax] - self.r, self.c[ax] + self.r) for ax in range(3)}


class _ZCone(_Boundary):
    """OpenMC z-cone (x-x0)^2 + (y-y0)^2 = r2 (z-z0)^2 (both nappes). dist is the
    distance to the cone in the meridian plane (exact away from the apex)."""

    def __init__(self, x0, y0, z0, r2, side):
        if r2 <= 0:
            raise ValueError("z-cone needs r2 > 0")
        self.c, self.t, self.S = (x0, y0, z0), math.sqrt(r2), math.sqrt(1 + r2)
        self.s = -1.0 if side == "-" else 1.0          # dist = s (rho - t|h|) / S

    def dist(self, P):
        rho = np.hypot(P[..., 0] - self.c[0], P[..., 1] - self.c[1])
        return self.s * (rho - self.t * np.abs(P[..., 2] - self.c[2])) / self.S

    def normal(self, P):
        u, v = P[..., 0] - self.c[0], P[..., 1] - self.c[1]
        rho = np.maximum(np.hypot(u, v), 1e-300)
        N = np.empty(np.shape(P))
        N[..., 0] = self.s * u / rho / self.S
        N[..., 1] = self.s * v / rho / self.S
        N[..., 2] = -self.s * self.t * np.sign(P[..., 2] - self.c[2]) / self.S
        return N

    def fall_hit(self, P):
        rho = math.hypot(P[0] - self.c[0], P[1] - self.c[1])
        h = P[2] - self.c[2]
        if self.s < 0:                                   # inside: allowed |h| >= hc
            hc = (rho + A * self.S) / self.t
            return self.c[2] + hc if h >= hc - 1e-12 else None
        hc = (rho - A * self.S) / self.t                 # outside: allowed |h| <= hc
        if hc < 0 or abs(h) > hc + 1e-12:
            return None
        return self.c[2] - hc

    def bounds_z(self, zlo, zhi):
        if self.s > 0 or not (np.isfinite(zlo) and np.isfinite(zhi)):
            return {}
        rmax = self.t * max(abs(zlo - self.c[2]), abs(zhi - self.c[2]))
        return {0: (self.c[0] - rmax, self.c[0] + rmax), 1: (self.c[1] - rmax, self.c[1] + rmax)}


def _make_boundary(kind, coeffs, side, scale=1.0):
    c = [float(v) for v in coeffs]
    if kind in ("x-plane", "y-plane", "z-plane"):
        n = [0.0, 0.0, 0.0]
        n["xyz".index(kind[0])] = 1.0
        return _Plane(*n, c[0] * scale, side)
    if kind == "plane":
        return _Plane(c[0], c[1], c[2], c[3] * scale, side)
    if kind in ("x-cylinder", "y-cylinder", "z-cylinder"):
        return _Cylinder("xyz".index(kind[0]), c[0] * scale, c[1] * scale, c[2] * scale, side)
    if kind == "sphere":
        return _Sphere(*(v * scale for v in c), side)
    if kind == "z-cone":
        return _ZCone(c[0] * scale, c[1] * scale, c[2] * scale, c[3], side)
    raise ValueError(f"surface type '{kind}' is not supported (supported: {', '.join(SUPPORTED)})")


def _parse_region(expr, surfs):
    toks = re.findall(r"\(|\)|\||~|[+-]?\d+", expr)
    if "|" in toks or "~" in toks:
        raise ValueError(f"region '{expr}': unions (|) and complements (~) are not supported; "
                         "the container must be an intersection of half-spaces")
    out = []
    for t in toks:
        if t in ("(", ")"):
            continue
        sid = t.lstrip("+-")
        if sid not in surfs:
            raise ValueError(f"surface {sid} used in region '{expr}' is not defined")
        kind, c = surfs[sid]
        out.append((kind, tuple(float(v) for v in c), "-" if t.startswith("-") else "+"))
    return out


class Container:
    """Pebble container: intersection of surface half-spaces, given as
    (OpenMC surface type, coefficients, '+' or '-') in the same length unit as the
    pebble diameter. bbox = (xmin, xmax, ymin, ymax, zmin, zmax) is only needed when it
    cannot be deduced from the surfaces (e.g. a container closed only by planes)."""

    def __init__(self, halfspaces, bbox=None, name="container"):
        self.halfspaces = [(k, tuple(c), s) for k, c, s in halfspaces]
        self.bbox, self.name = bbox, name
        self.boundaries()                                 # validate

    def boundaries(self, scale=1.0):
        return [_make_boundary(k, c, s, scale) for k, c, s in self.halfspaces]

    def bounds(self, scale=1.0):
        """(lo, hi) arrays of the container bounding box, scaled."""
        B = self.boundaries(scale)
        lo, hi = np.full(3, -np.inf), np.full(3, np.inf)

        def clip(bd):
            for ax, (l, h) in bd.items():
                lo[ax], hi[ax] = max(lo[ax], l), min(hi[ax], h)
        for b in B:
            clip(b.bounds())
        for b in B:
            if isinstance(b, _ZCone):
                clip(b.bounds_z(lo[2], hi[2]))
        if self.bbox is not None:
            bb = np.asarray(self.bbox, dtype=float) * scale
            lo, hi = np.maximum(lo, bb[0::2]), np.minimum(hi, bb[1::2])
        return lo, hi

    def outer_radius(self):
        """Smallest radius of the '-' z-cylinders on the z axis (the core wall), or None."""
        r = [c[2] for k, c, s in self.halfspaces
             if k == "z-cylinder" and s == "-" and abs(c[0]) < 1e-12 and abs(c[1]) < 1e-12]
        return min(r) if r else None

    @classmethod
    def cylinder(cls, radius, height=None, hole_radius=0.0, z_bottom=0.0):
        """Cylinder on the z axis, floor at z_bottom, optional lid and central hole."""
        hs = [("z-cylinder", (0.0, 0.0, radius), "-"), ("z-plane", (z_bottom,), "+")]
        if height is not None:
            hs.append(("z-plane", (z_bottom + height,), "-"))
        if hole_radius:
            hs.append(("z-cylinder", (0.0, 0.0, hole_radius), "+"))
        return cls(hs, name=f"cylinder R={radius:g}" + (f", hole {hole_radius:g}" if hole_radius else ""))

    @classmethod
    def from_xml(cls, path, cell_id, bbox=None):
        """Region of cell cell_id in an OpenMC geometry.xml."""
        import xml.etree.ElementTree as ET
        root = ET.parse(path).getroot()
        surfs = {}
        for s in root.iter("surface"):
            kind = s.get("type") or s.findtext("type")
            coeffs = s.get("coeffs") or s.findtext("coeffs")
            surfs[s.get("id")] = (kind.strip(), coeffs.split())
        cells = {c.get("id"): c for c in root.iter("cell")}
        cell = cells.get(str(cell_id))
        if cell is None:
            raise ValueError(f"cell {cell_id} not found in {path} (cells: {', '.join(cells)})")
        region = cell.get("region") or cell.findtext("region")
        if not region:
            raise ValueError(f"cell {cell_id} has no region")
        return cls(_parse_region(region, surfs), bbox=bbox, name=f"{path} cell {cell_id}")

    @classmethod
    def from_openmc(cls, region, bbox=None):
        """From an openmc.Region (intersection of half-spaces)."""
        import openmc
        hs = []

        def walk(r):
            if isinstance(r, openmc.Halfspace):
                s = r.surface
                hs.append((s.type, tuple(getattr(s, k) for k in s._coeff_keys), r.side))
            elif isinstance(r, openmc.Intersection):
                for q in r:
                    walk(q)
            else:
                raise ValueError("only intersections of half-spaces are supported "
                                 "(no unions or complements)")
        walk(region)
        return cls(hs, bbox=bbox, name=str(region))


# ------------------------------------------------------------------ packer
class DropAndRollPacker:
    """Sequential gravity deposition. Internally everything is in units of D (D = 1).
    Constraint ids: >= 0 is a pebble, -1 - k is boundary k."""

    def __init__(self, container, pebble_diameter, fill_level=None, seed=None,
                 gap=1e-6, step=0.05):
        scale = 1.0 / pebble_diameter
        allb = container.boundaries(scale)
        self.lo, self.hi = container.bounds(scale)
        if not (np.all(np.isfinite(self.lo[:2])) and np.all(np.isfinite(self.hi[:2]))
                and np.isfinite(self.lo[2])):
            raise ValueError("Cannot deduce the container bounding box from its surfaces "
                             "(is the region closed below and on the sides?); "
                             "give bbox=(xmin, xmax, ymin, ymax, zmin, zmax).")
        if np.any(self.hi[:2] - self.lo[:2] < 1.0):
            raise ValueError("The container is narrower than one pebble.")
        lids = [b.lid for b in allb if b.lid is not None]
        if fill_level is not None:
            lids.append(fill_level * scale)
        self.ztop = min(lids) - A if lids else None    # max center height
        self.B = [b for b in allb if b.lid is None]
        self.bids = [-1 - k for k in range(len(self.B))]
        self.dc = 1.0 + gap                            # contact distance (tiny clearance)
        self.step = step                               # rolling step, in diameters
        self.cs = 1.25                                 # neighbour-grid cell size (> dc + step)
        self.rng = np.random.default_rng(seed)
        self.C = np.empty((1024, 3))
        self.n = 0
        self.grid3 = {}
        self.grid2 = {}
        self.zmax = self.lo[2] + A
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
    def _is_floor(self, c):
        return c is not None and c < 0 and self.B[-1 - c].floor

    def _gaps(self, P, cons):
        """Gap function g (>= 0 means no overlap) for each constraint id."""
        cons = np.asarray(cons, dtype=int)
        out = np.empty(len(cons))
        peb = cons >= 0
        if peb.any():
            out[peb] = np.sqrt(((self.C[cons[peb]] - P) ** 2).sum(axis=1)) - self.dc
        for m in np.nonzero(~peb)[0]:
            out[m] = self.B[-1 - cons[m]].gap(P)
        return out

    def _normal(self, P, c):
        if c < 0:
            return self.B[-1 - c].normal(P)
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
                if c < 0:
                    b = self.B[-1 - c]
                    g[m] = b.gap(P)
                    N[m] = b.normal(P)
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

    def _descent(self, P, A_):
        """Project gravity onto the tangent cone of the active contacts A_.
        Returns (direction d, load-bearing contacts S). d = 0 means stable."""
        g = np.array([0.0, 0.0, -1.0])
        N_all = {c: self._normal(P, c) for c in A_}
        for size in range(len(A_) + 1):
            for S in itertools.combinations(A_, size):
                if size == 0:
                    d, mu = g, np.zeros(0)
                else:
                    N = np.array([N_all[c] for c in S])
                    mu = np.linalg.lstsq(N @ N.T, -(N @ g), rcond=None)[0]
                    d = g + N.T @ mu
                if np.any(mu < -1e-10):
                    continue
                if all(N_all[c] @ d >= -1e-10 for c in A_ if c not in S):
                    return d, [c for c, m in zip(S, mu) if m > 1e-12] or list(S)
        return np.zeros(3), list(A_)           # degenerate -> treat as stable

    # --------------------------------------------------------------- dynamics
    def _free_fall(self, P):
        """Drop vertically; return new position and the contact that stops it."""
        z_hit, hit = -math.inf, None
        for c, b in zip(self.bids, self.B):
            zb = b.fall_hit(P)
            if zb is not None and zb <= P[2] + 1e-9 and zb > z_hit:
                z_hit, hit = zb, c
        idx = self._column(P)
        if len(idx):
            Cj = self.C[idx]
            dh2 = (Cj[:, 0] - P[0]) ** 2 + (Cj[:, 1] - P[1]) ** 2
            ok = (dh2 < self.dc ** 2) & (Cj[:, 2] < P[2] - 1e-12)
            if np.any(ok):
                zc = Cj[ok, 2] + np.sqrt(self.dc ** 2 - dh2[ok])
                m = np.argmax(zc)
                if zc[m] > z_hit:
                    z_hit, hit = zc[m], idx[ok][m]
        if hit is None:
            raise RuntimeError("A pebble fell through the bottom of the container: "
                               "the region must be closed below.")
        Q = P.copy()
        Q[2] = min(z_hit, P[2])
        return Q, hit

    def _start(self, max_tries=10000):
        """Random drop point above the bed, inside the container cross-section."""
        z = self.zmax + 2.0
        for _ in range(max_tries):
            P = np.array([self.rng.uniform(self.lo[0] + A, self.hi[0] - A),
                          self.rng.uniform(self.lo[1] + A, self.hi[1] - A), z])
            if all(b.gap(P) >= 0 for b in self.B):
                return P
        return None

    def drop_one(self, max_iter=20000):
        """Drop one pebble and return its resting position (not yet added), or None
        if no drop point above the bed is inside the container."""
        P = self._start()
        if P is None:
            return None
        P, hit = self._free_fall(P)
        if self._is_floor(hit):
            return P
        for _ in range(max_iter):
            nb = self._neighbours(P)
            cons = self.bids + nb.tolist()
            gaps = self._gaps(P, cons)
            A_ = [c for c, gv in zip(cons, gaps) if gv < self.touch_tol]
            if any(self._is_floor(c) for c in A_):
                return P
            d, S = self._descent(P, A_)
            nd = np.linalg.norm(d)
            if nd < 1e-10:
                return P                                  # stable on 2-3 supports
            if not S:                                     # nothing carries load: fall
                P, hit = self._free_fall(P)
                if self._is_floor(hit):
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
                already = [c for c in bad if c in A_]     # tangent contacts stay on
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
        self.n, self.grid3, self.grid2, self.zmax = 0, {}, {}, self.lo[2] + A
        for p in P:
            self._add(p)

    def _violation(self, P, zl=None):
        """Largest boundary (and lid) crossing of the pebbles in P, in D."""
        v = max((max(0.0, -b.gap(P).min()) for b in self.B), default=0.0)
        if zl is not None:
            v = max(v, P[:, 2].max() - zl)
        return v

    def _relax(self, P, d0, zl, iters):
        """Push overlapping pebbles apart (Jacobi), then project back inside the container."""
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
            for b in self.B:
                g = b.gap(P)
                m = g < 0
                if m.any():
                    P[m] -= g[m, None] * b.normal(P[m])
            if zl is not None:
                P[:, 2] = np.minimum(P[:, 2], zl)
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
        zl = self.ztop                                    # surfaces/lid: true radius
        for c in range(cycles):
            x = c / cycles
            P[:, 2] -= gravity * math.sqrt(1.0 - x)
            P += noise * (1.0 - x) ** 2 * self.rng.standard_normal(P.shape)
            P = self._relax(P, d0, zl, inner)
        for k in range(500):                              # cleanup, no gravity
            P = self._relax(P, d0, zl, 5)
            if (cKDTree(P).query(P, k=2)[0][:, 1].min() >= 1.0 + 1e-9
                    and self._violation(P, zl) <= 1e-9):
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
            if P is None:                                 # no room left above the bed
                if n_pebbles is not None:
                    raise RuntimeError(f"Only {self.n} pebbles fit in the container; "
                                       f"{n_pebbles} requested.")
                break
            if self.ztop is not None and P[2] > self.ztop + 1e-9:
                if n_pebbles is not None:
                    raise RuntimeError(
                        f"Only {self.n} pebbles fit below the fill level; "
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


def generate_pebbles_in(container, pebble_diameter, n_pebbles=None, fill_level=None,
                        seed=None, compaction=70, verbose=True):
    """Return an (N, 3) array of pebble centers inside container (a Container), same
    length unit as the inputs.

      container has a lid (z-plane, region below) or fill_level (z) is given:
                         fill up to it (pebbles are added until no more fit).
      n_pebbles          : pour exactly that many pebbles (free top surface);
                         error if they do not fit below the lid.
    compaction = number of gravity-tapping cycles after deposition (see pebble_pack.py).
    Check the printed bulk packing fraction and adjust."""
    compaction = 70 if compaction is True else int(compaction or 0)
    D = float(pebble_diameter)
    packer = DropAndRollPacker(container, D, fill_level=fill_level, seed=seed)
    if n_pebbles is None and packer.ztop is None:
        raise ValueError("The container has no lid (a z-plane with the region below it): "
                         "give n_pebbles or fill_level.")
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
        raise RuntimeError("Pebbles exceed the fill level.")
    if packer._violation(C) > 1e-9:
        raise RuntimeError("Pebbles cross a container surface.")
    return C * D


def generate_pebbles(pebble_diameter, core_radius, core_height=None, n_pebbles=None,
                     seed=None, compaction=70, verbose=True, hole_radius=0.0):
    """Cylinder on the z axis with the floor at z = 0 (same interface as
    pebble_pack.generate_pebbles), plus an optional central hole of hole_radius."""
    if core_height is None and n_pebbles is None:
        raise ValueError("Give core_height, n_pebbles, or both.")
    box = Container.cylinder(core_radius, core_height, hole_radius)
    return generate_pebbles_in(box, pebble_diameter, n_pebbles=n_pebbles, seed=seed,
                               compaction=compaction, verbose=verbose)


def packing_stats(centers, pebble_diameter, container, fill_level=None, n_grid=8e6):
    """Packing fraction (overall, and bulk = centers more than 2.5 D from every surface
    and from the top), minimum center distance and largest surface crossing.
    The container volume (up to the lid/fill level, or to the bed top if there is none)
    is integrated on a midpoint grid of ~n_grid points."""
    from scipy.spatial import cKDTree
    D = float(pebble_diameter)
    C = np.asarray(centers, dtype=float) / D
    allb = container.boundaries(1.0 / D)
    lo, hi = container.bounds(1.0 / D)
    lids = [b.lid for b in allb if b.lid is not None]
    if fill_level is not None:
        lids.append(fill_level / D)
    ztop = min(lids) if lids else C[:, 2].max() + A
    B = [b for b in allb if b.lid is None]
    hi = hi.copy()
    hi[2] = min(hi[2], ztop)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("Cannot deduce the container bounding box; give bbox.")
    h = (np.prod(hi - lo) / n_grid) ** (1 / 3)
    nn = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)
    ax = [lo[k] + (np.arange(nn[k]) + 0.5) * (hi[k] - lo[k]) / nn[k] for k in range(3)]
    dv = np.prod((hi - lo) / nn)
    X, Y = np.meshgrid(ax[0], ax[1], indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    vol = vbulk = 0
    for z in ax[2]:
        P = np.column_stack([X, Y, np.full(X.size, z)])
        d = np.min([b.dist(P) for b in B], axis=0)
        vol += np.count_nonzero(d >= 0)
        if z < ztop - 2.5:
            vbulk += np.count_nonzero(d >= 2.5)
    vol, vbulk = vol * dv, vbulk * dv
    vp = math.pi / 6
    dc = np.min([b.dist(C) for b in B], axis=0)
    nbulk = np.count_nonzero((dc >= 2.5) & (C[:, 2] < ztop - 2.5))
    dmin = cKDTree(C).query(C, k=2)[0][:, 1].min()
    return {"n": len(C), "packing_fraction": len(C) * vp / vol,
            "bulk_packing_fraction": nbulk * vp / vbulk if vbulk > 0 else float("nan"),
            "min_center_distance": dmin * D, "container_volume": vol * D ** 3,
            "max_surface_crossing": max(0.0, A - dc.min()) * D}


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
    """Azimuthally averaged porosity (void fraction) in annular shells r..r+dr around
    the z axis (x = y = 0).

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
    g = ap.add_argument_group("container from an OpenMC geometry")
    g.add_argument("--geometry", help="OpenMC geometry.xml")
    g.add_argument("--cell", help="id of the cell whose region is the container")
    g.add_argument("--bbox", type=float, nargs=6,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   help="bounding box, only if it cannot be deduced from the surfaces")
    g = ap.add_argument_group("built-in cylinder (axis z, floor at z = 0)")
    g.add_argument("--core-radius", type=float, help="core (cylinder) radius")
    g.add_argument("--core-diameter", type=float, help="core diameter (alternative)")
    g.add_argument("--height", type=float, help="core height to fill")
    g.add_argument("--hole-radius", type=float, default=0.0, help="central hole radius")
    ap.add_argument("--fill-level", type=float,
                    help="fill up to this z (adds a lid; for containers without one)")
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
                    help="built-in cylinder only: z = 0 at the floor, at mid-height of "
                         "the core, or at the pebble centroid (pebbles.dat uses centroid)")
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

    D = a.diameter
    R = a.core_radius if a.core_radius else (a.core_diameter / 2 if a.core_diameter else None)
    if a.geometry:
        if a.cell is None:
            ap.error("--geometry needs --cell")
        if a.origin != "bottom":
            ap.error("--origin only applies to the built-in cylinder")
        box = Container.from_xml(a.geometry, a.cell, bbox=a.bbox)
        R = R or box.outer_radius()
    else:
        if R is None:
            ap.error("give --geometry and --cell, or --core-radius / --core-diameter")
        box = Container.cylinder(R, a.height, a.hole_radius)
        if a.bbox:
            box.bbox = a.bbox

    t0 = time.time()
    if a.input:
        P = np.loadtxt(a.input, usecols=(0, 1, 2))
    else:
        P = generate_pebbles_in(box, D, n_pebbles=a.n_pebbles, fill_level=a.fill_level,
                                seed=a.seed, compaction=a.compaction)
    st = packing_stats(P, D, box, a.fill_level)
    print(f"{st['n']} pebbles in {time.time() - t0:.1f} s | packing fraction "
          f"{st['packing_fraction']:.4f} (bulk {st['bulk_packing_fraction']:.4f}, "
          f"porosity {1 - st['packing_fraction']:.4f}) | "
          f"min center distance / D = {st['min_center_distance'] / D:.6f} | "
          f"max surface crossing {st['max_surface_crossing']:.2g}", file=sys.stderr)

    if a.porosity_plot or a.porosity_csv:
        if R is None:
            ap.error("radial porosity needs the core radius: give --core-radius")
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
