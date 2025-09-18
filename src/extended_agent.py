import math
import numpy as np
from mesa import Agent
from simple_agent import SimplePedestrian
from utils import _norm

class ExtendedPedestrian(SimplePedestrian):
    def __init__(
        self,
        unique_id,
        model,
        pos,
        v0=1.3,
        tau=0.5,
        radius=0.3,
        mass=80.0,
        knows_exit=False,
        herding_radius=3.0,
        is_leader=False,
        exit_id=None,
        visibility_radius=10.0,
        panic=0.5,
        vmax=5.0,
        alpha_imp=0.2,
    ):
        super().__init__(unique_id, model, pos, v0, tau, radius, mass)

        # --- extra attributes ---
        self.heading = np.random.uniform(0, 2 * math.pi)
        self.knows_exit = knows_exit
        self.herding_radius = herding_radius
        self.is_leader = is_leader
        self.exit_id = exit_id
        self.e0 = np.zeros(2)
        self.visibility_radius = visibility_radius

        # speed / impatience dynamics
        self.v0_init = v0
        self.vmax = vmax
        self.impatience = 0.0
        self.alpha_imp = alpha_imp

        # panic dynamics
        self.panic = panic
        self.panic_base = panic

    
    # --- override hook methods from SimplePedestrian ---
    def pre_physics_update(self):
        """Update desired panic, direction, impatience and desired speed before each step."""
        self.update_panic()
        gx, gy = self.nearest_exit_point()
        self.e0 = self.desired_direction(gx, gy)
        self.update_impatience_and_speed()

    def desired_direction(self, gx, gy):
        """Return blended direction based on exit, leaders, neighbors, or exploration."""
        if self.knows_exit or self.is_leader:
            return _norm(np.array([gx - self.x, gy - self.y]))
        e_ind = self.exploration_dir(gx, gy)
        leader_dir = self.nearest_leader_dir()
        if leader_dir is not None:
            blend = (1.0 - self.panic) * e_ind + self.panic * leader_dir
        else:
            e_nb = self.neighbor_mean_desired_dir()
            blend = (1.0 - self.panic) * e_ind + self.panic * e_nb
        return _norm(blend) if np.linalg.norm(blend) > 1e-12 else e_ind

    def visibility_metrics(self):
        """
        Returns:
        R_i      = effective visibility radius to be used in can_see (meters)
        vis_term = term in [0,1] for panic update (0 = clear, 1 = no visibility)

        Convention:
        - If the model exposes visibility_at(x,y), we assume it returns R_i (meters).
        - Otherwise we use self.visibility_radius.
        - vis_ref can be defined in the model; fallback is self.visibility_radius.
        """
        if hasattr(self.model, "visibility_at"):
            R_i = float(self.model.visibility_at(self.x, self.y))  # meters (radius)
        else:
            R_i = float(self.visibility_radius)

        V_ref = float(getattr(self.model, "vis_ref", self.visibility_radius))
        V_ref = max(V_ref, 1e-6)  # avoid division by zero
        vis_term = float(np.clip(1.0 - (R_i / V_ref), 0.0, 1.0))
        return R_i, vis_term

    def can_see(self, x, y):
        """Check if a point (x,y) is within the effective visibility radius."""
        R_i, _ = self.visibility_metrics()
        return np.hypot(x - self.x, y - self.y) <= R_i

    def nearest_exit_point(self):
        """
        Return the point (gx, gy) on the chosen exit segment.
        - If self.exit_id is defined: use that exit (leaders usually).
        - Otherwise: select the currently closest exit (followers).
        Assumes self.model.exits = [(x0, y0, x1, y1), ...]
        """
        exits = self.model.exits

        # helper: projection + distance
        def proj_and_dist(x0, y0, x1, y1):
            gx, gy = self._project_to_line(self.x, self.y, x0, y0, x1, y1)
            d = np.linalg.norm([gx - self.x, gy - self.y])
            return (gx, gy, d)

        # leader: fixed exit_id
        if getattr(self, "exit_id", None) is not None:
            x0, y0, x1, y1 = exits[self.exit_id]
            gx, gy = self._project_to_line(self.x, self.y, x0, y0, x1, y1)
            return (gx, gy)

        # follower: choose nearest exit dynamically
        best = None
        best_d = float("inf")
        for x0, y0, x1, y1 in exits:
            gx, gy, d = proj_and_dist(x0, y0, x1, y1)
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
                self.heading = np.random.uniform(0, 2 * math.pi)
            return np.array([math.cos(self.heading), math.sin(self.heading)])

    def neighbor_mean_desired_dir(self):
        R, _ = self.visibility_metrics()
        R = min(self.herding_radius, R)
        vec_sum = np.zeros(2)
        count = 0
        for n in self.model.space.get_neighbors(
            (self.x, self.y), R, include_center=False
        ):
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
        R, _ = self.visibility_metrics()
        R = min(self.herding_radius, R)
        closest = None
        min_dist = float("inf")
        for n in self.model.space.get_neighbors(
            (self.x, self.y), R, include_center=False
        ):
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

    def update_impatience_and_speed(self):
        """
        Update v0(t) per Helbing's impatience idea.
        p_imp = 1 - (progress along e0) / v0_init.
        Low-pass the impatience to avoid jitter.
        """
        e_dir = self.e0 if np.linalg.norm(self.e0) > 1e-12 else np.array([0.0, 0.0])
        v_vec = np.array([self.vx, self.vy])
        v_parallel = max(0.0, float(np.dot(v_vec, e_dir)))  # progress, not backwards

        # raw impatience: 0 if moving at v0_init, 1 if blocked
        p_imp_raw = 1.0 - (v_parallel / self.v0_init)
        p_imp_raw = float(np.clip(p_imp_raw, 0.0, 1.0))

        # smooth update (exponential moving average)
        self.impatience = (
            1.0 - self.alpha_imp
        ) * self.impatience + self.alpha_imp * p_imp_raw

        # update desired speed between relaxed v0_init and panic vmax
        self.v0 = (1.0 - self.impatience) * self.v0_init + self.impatience * self.vmax

    def update_panic(self):
        """
        Update panic according to:
        p = 1 - (1 - p_base) * (1 - vis_term), where vis_term comes from visibility_metrics() and is in [0,1].
        """
        _, vis_term = self.visibility_metrics()
        self.panic = 1.0 - (1.0 - self.panic_base) * (1.0 - vis_term)

    def _ccw(self, ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

    def _segments_intersect(self, p1, p2, q1, q2):
        (ax, ay), (bx, by) = p1, p2
        (cx, cy), (dx, dy) = q1, q2
        return (self._ccw(ax, ay, cx, cy, dx, dy) != self._ccw(bx, by, cx, cy, dx, dy) and
                self._ccw(ax, ay, bx, by, cx, cy) != self._ccw(ax, ay, bx, by, dx, dy))

    def has_exited(self):
        x_prev = getattr(self, "_last_x", self.x)
        y_prev = getattr(self, "_last_y", self.y)
        p1 = (x_prev, y_prev)
        p2 = (self.x, self.y)

        for x0, y0, x1, y1 in self.model.exits:
            q1 = (x0, y0)
            q2 = (x1, y1)
            if self._segments_intersect(p1, p2, q1, q2):
                return True
        return False
