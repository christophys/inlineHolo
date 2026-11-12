# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Christoph T. Koch
"""Cooperative stopping of long computations (used by the interactive application)."""


class StopRequested(Exception):
    """Raised from a progress callback to stop a fit, an alignment or a reconstruction.

    FRWR3 catches it and returns the current wave as the result; the Thon-ring fits return
    their best parameters so far; other iterative methods are stopped by the caller, which
    keeps the last wave passed to the callback.
    """


class ScheduleETA:
    """Remaining-time estimate for a planned sequence of loops with different costs.

    ``schedule`` is a list of (key, n_images) in execution order (e.g.
    :meth:`inlineholo.frwr3.FRWR3.planned_schedule`).  The time of a loop is taken to be
    n_images x c_key(n_images): the cost per image is learnt online for every key (loops of the
    same kind, e.g. the j-th loop of every intermediate FRWR stage, whose alignment / distortion
    fit dominates the run time), with a power law in n_images fitted to the stages seen so far,
    so that the later stages with more images are predicted to take longer.  Keys not seen yet
    use related keys (final alignment loops ~ first loop of an intermediate stage, plain
    projections ~ the cheapest loops) or, at the very beginning, fixed ratios to the cost of a
    plain projection.  ``update()`` is called once per finished loop.
    """

    PRIOR = {0: 40.0, 1: 20.0, 3: 6.0}           # cost / projection cost of intermediate loop j

    def __init__(self, schedule, t0=None):
        import time
        self.schedule = list(schedule)
        self.k = 0
        self.t0 = time.time() if t0 is None else t0
        self.t_prev = self.t0
        self.obs = {}

    def replan(self, schedule):
        """Replace the loops still to come (e.g. once FRWR3 has chosen its start planes)."""
        self.schedule = self.schedule[:self.k] + list(schedule)

    @property
    def ready(self):
        """True once a few loops have been timed (the first one includes the set-up)."""
        return self.k >= 3

    def update(self, t=None):
        import time
        t = time.time() if t is None else t
        if 0 < self.k < len(self.schedule):           # the first loop includes the set-up: not used
            key, n = self.schedule[self.k]
            self.obs.setdefault(key, []).append((n, max(t - self.t_prev, 1e-6) / max(n, 1)))
        self.k += 1
        self.t_prev = t

    def _proj_cost(self):
        c = [v for key, o in self.obs.items() for (_, v) in o[-3:]]
        return min(c) if c else None

    def per_image(self, key, n):
        import numpy as np
        o = self.obs.get(key)
        if o:
            ns = np.array([a for a, _ in o], float)
            cs = np.array([b for _, b in o], float)
            if len(set(ns)) >= 2:                      # growth between the two latest stages
                i = max(i for i in range(len(ns)) if ns[i] != ns[-1])
                p = np.log(cs[-1] / cs[i]) / np.log(ns[-1] / ns[i])
                p = float(np.clip(p, 0.0, 1.0))
            else:
                p = 0.5 if key[0] == "inter" else 0.0
            return float(cs[-1] * (n / ns[-1]) ** p)
        if key[0] == "final":
            if key[1] == "align":
                return self.per_image(("inter", 0), n) if ("inter", 0) in self.obs else self._prior(0, n)
            return self._prior(2, n)
        if key[0] == "residue":
            return self.per_image(("final", "proj"), n)
        if key[0] == "inter":
            return self._prior(key[1], n)
        return self._prior(2, n)

    def _prior(self, j, n):
        c = self._proj_cost()
        if c is None:
            return 0.0
        if j == 2 or j not in self.PRIOR:
            inter = [k for k in self.obs if k[0] == "inter"]
            if inter:                                  # cheapest intermediate loop seen
                return min(self.per_image(k, n) for k in inter)
            return c
        return c * self.PRIOR[j]

    def remaining(self):
        return sum(n * self.per_image(key, n) for key, n in self.schedule[self.k:])

    def fraction(self, t=None):
        import time
        el = (time.time() if t is None else t) - self.t0
        rem = self.remaining()
        return el / max(el + rem, 1e-9), rem
