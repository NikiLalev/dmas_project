import math
import random
import numpy as np
from mesa import Agent

# --- utility ---
def _norm(vec):
    """Safe normalization of a 2D vector."""
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else vec


class Pedestrian(Agent):
    """
    Minimal social-force pedestrian:
    - continuous position (x,y), velocity (vx,vy)
    - drives toward exit segment
    - repels from other agents + walls
    """
    def __init__(self, unique_id, model, pos, v0=1.3, tau=0.5, radius=0.3, mass=80.0, knows_exit=False, panic=0.5, herding_radius=3.0, is_leader=False, exit_id=None, visibility_radius=10.0):
        super().__init__(unique_id, model)
        self.x, self.y = pos
        self.vx, self.vy = 0.0, 0.0
        self.v0 = v0        # desired speed (m/s)
        self.tau = tau      # adaptation time (s)
        self.r = radius     # body radius (m)
        self.m = mass       # body mass (kg)

        # Helbing params (fixed at the moment)
        self.A = 2000.0
        self.B = 0.08
        self.k = 1.2e5
        self.kappa = 2.4e5

        # extra attributes
        self.heading = np.random.uniform(0, 2*math.pi)  # random direction degree for exploration
        self.knows_exit = knows_exit
        self.panic = panic         # herding factor p ∈ [0,1]
        self.herding_radius = herding_radius # radius for neighbor influence
        self.is_pedestrian = True     #to distinguish from wall (or other) objects
        self.is_leader = is_leader
        self.exit_id = exit_id       # for leaders: fixed exit index
        self.e0 = np.zeros(2)     # desired direction cache
        self.visibility_radius = visibility_radius # for exit/neighbor visibility

    def can_see(self, x, y):
        return np.linalg.norm(np.array([x - self.x, y - self.y])) <= self.visibility_radius

    def nearest_point_on_exit(self):
        """
        Return the point (gx, gy) on the chosen exit segment.
        - If self.exit_id is defined: use that exit (leaders usually).
        - Otherwise: select the currently closest exit (followers).
        Assumes self.model.exits = [(x_exit, y0, y1), ...]
        """
        exits = self.model.exits

        # helper: compute projection on exit segment and distance
        def proj_and_dist(xe, y0, y1):
            # clamp y within [y0, y1]
            y_clamped = min(max(self.y, y0), y1)
            gx, gy = xe, y_clamped
            d = np.linalg.norm([gx - self.x, gy - self.y])
            return (gx, gy, d)

        # case: agent has a fixed exit assigned (e.g., leader)
        if getattr(self, "exit_id", None) is not None:
            xe, y0, y1 = exits[self.exit_id]
            y_clamped = min(max(self.y, y0), y1)
            return (xe, y_clamped)

        # case: follower without exit_id: choose nearest exit dynamically
        best = None
        best_d = float("inf")
        for (xe, y0, y1) in exits:
            gx, gy, d = proj_and_dist(xe, y0, y1)
            if d < best_d:
                best = (gx, gy)
                best_d = d
        return best

    # ---- directions ----
    def known_exit_dir(self, gx, gy):
        return _norm(np.array([gx - self.x, gy - self.y]))

    def exploration_dir(self, gx, gy):
        if self.can_see(gx, gy):
            self.knows_exit = True
            return self.known_exit_dir(gx, gy)
        else:
            # persistent random walk
            if np.random.rand() < 0.05:
                self.heading = np.random.uniform(0, 2*math.pi)
            return np.array([math.cos(self.heading), math.sin(self.heading)])

    def neighbor_mean_desired_dir(self):
        R = min(self.herding_radius, self.visibility_radius)
        vec_sum = np.zeros(2)
        count = 0
        for n in self.model.space.get_neighbors((self.x, self.y), R, include_center=False):
            if n is self or not getattr(n, "is_pedestrian", False):
                continue
            if not self.can_see(n.x, n.y):
                continue
            e_j0 = getattr(n, "e0", None)
            if e_j0 is None:
                e_j0 = _norm(np.array([n.vx, n.vy]))
            vec_sum += e_j0
            count += 1
        return _norm(vec_sum) if count > 0 else np.zeros(2)
    
    def nearest_leader_dir(self):
        """Return the e0 of the closest visible leader, or None if none are visible."""
        R = min(self.herding_radius, self.visibility_radius)
        closest = None
        min_dist = float("inf")
        for n in self.model.space.get_neighbors((self.x, self.y), R, include_center=False):
            if n is self or not getattr(n, "is_pedestrian", False) or not n.is_leader:
                continue
            if not self.can_see(n.x, n.y):
                continue
            d = np.linalg.norm([n.x - self.x, n.y - self.y])
            if d < min_dist:
                min_dist = d
                e_j0 = getattr(n, "e0", None)
                if e_j0 is None:
                    e_j0 = _norm(np.array([n.vx, n.vy]))
                closest = e_j0
        return closest

    def compute_desired_direction(self):
        gx, gy = self.nearest_point_on_exit()
        if self.knows_exit or self.is_leader:
            e0 = self.known_exit_dir(gx, gy)
        else:
            e_ind = self.exploration_dir(gx, gy)
            leader_dir = self.nearest_leader_dir()
            if leader_dir is not None:
                blend = (1.0 - self.panic) * e_ind + self.panic * leader_dir
            else:
                e_nb = self.neighbor_mean_desired_dir()
                blend = (1.0 - self.panic) * e_ind + self.panic * e_nb
            e0 = _norm(blend) if np.linalg.norm(blend) > 1e-12 else e_ind
        self.e0 = e0
        return e0
    
    def driving_forces(self):
        e0 = self.compute_desired_direction()
        vdx, vdy = self.v0 * e0[0], self.v0 * e0[1]
        fx = self.m * (vdx - self.vx) / self.tau
        fy = self.m * (vdy - self.vy) / self.tau
        return fx, fy