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
        smoke_recovery_rate=0.1,
        knows_exit=False,
        herding_radius=3.0,
        is_leader=False,
        exit_id=None,
        visibility_radius=10.0,
        panic=0.5,
        vmax=5.0,
        alpha_imp=0.2,
    ):
        super().__init__(unique_id, model, pos, v0, tau, radius, mass, smoke_recovery_rate)

        # --- extra attributes ---
        self.heading = np.random.uniform(0, 2 * math.pi)
        self.knows_exit = knows_exit
        self.herding_radius = herding_radius
        self.is_leader = is_leader
        self.exit_id = exit_id
        self.e0 = np.zeros(2)
        self.visibility_radius = visibility_radius

        # speed / impatience dynamics
        if is_leader:
            self.v0_init = 0.5
            self.vmax = 1.0
        else:
            self.v0_init = v0
            self.vmax = vmax
        self.impatience = 0.0
        self.alpha_imp = alpha_imp

        # panic dynamics
        self.panic = panic
        self.panic_base = panic

        self.leader_radius = 15.0

        self.follow_target_id = None   # currently latched leader id (or None)
        self.follow_timer = 0.0        # seconds left to keep current latch even if leader not seen
        self.follow_min_time = 1.0     # minimum latch duration (seconds)
        self.follow_bias = 0.35
        self.last_follow_dir = np.zeros(2)

    
    # --- override hook methods from SimplePedestrian ---
    def pre_physics_update(self):
        """Update desired panic, direction, impatience and desired speed before each step."""
        self.update_panic()
        gx, gy = self.nearest_exit_point()
        self.e0 = self.desired_direction(gx, gy)
        self.update_impatience_and_speed()

    def desired_direction(self, gx, gy):
        """
        Blend exploration with following a latched leader direction (if any).
        Keep a baseline tendency to follow (follow_bias) even with low panic.
        While latched, if the leader is temporarily unseen, keep using last_follow_dir
        until follow_timer expires.
        """
        dt = float(self.model.dt)

        # Leaders or agents who already know the exit: head straight to exit
        if self.knows_exit or self.is_leader:
            return _norm(np.array([gx - self.x, gy - self.y]))

        # Base exploration
        e_ind = self.exploration_dir(gx, gy)

        # Nearest leader (dir + id)
        leader_dir, leader_id = self.nearest_leader_dir_with_id()

        # Maintain/refresh latch
        leader_dir_for_blend = None
        if self.follow_target_id is not None:
            # countdown latch
            self.follow_timer = max(0.0, self.follow_timer - dt)

            if leader_id == self.follow_target_id and leader_dir is not None:
                # same leader still available -> refresh latch + use its direction
                self.follow_timer = max(self.follow_timer, self.follow_min_time)
                self.last_follow_dir = leader_dir
                leader_dir_for_blend = leader_dir
            elif self.follow_timer > 0.0 and np.linalg.norm(self.last_follow_dir) > 1e-12:
                # leader not visible now, but latch active -> keep last known leader dir
                leader_dir_for_blend = self.last_follow_dir
            else:
                # latch expired
                self.follow_target_id = None
                self.last_follow_dir = np.zeros(2)

        # If no active latch, latch now if a leader is available
        if self.follow_target_id is None and leader_dir is not None:
            self.follow_target_id = leader_id
            self.follow_timer = self.follow_min_time
            self.last_follow_dir = leader_dir
            leader_dir_for_blend = leader_dir

        # Blend
        if leader_dir_for_blend is not None:
            # baseline follow even with low panic
            follow_w = self.follow_bias + (1.0 - self.follow_bias) * float(self.panic)
            return _norm((1.0 - follow_w) * e_ind + follow_w * leader_dir_for_blend)
        else:
            e_nb = self.neighbor_mean_desired_dir()
            return _norm((1.0 - self.panic) * e_ind + self.panic * e_nb)

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
        cap = float(self.visibility_radius)
        if hasattr(self.model, "visibility_at"):
            env = float(self.model.visibility_at(self.x, self.y))  # local visibility
        else:
            env = cap  # no visibility model, assume max visibility

        R_i = min(cap, env)  # effective visibility radius
        vis_term = float(np.clip(1.0 - (R_i / max(cap, 1e-6)), 0.0, 1.0))
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

    def nearest_leader_dir_with_id(self):
        """Return (dir, leader_id) for the nearest leader within leader_radius, ignoring visibility."""
        R = float(getattr(self, "leader_radius", 12.0))
        best_dir, best_id, best_d = None, None, float("inf")

        for n in self.model.space.get_neighbors((self.x, self.y), R, include_center=False):
            if n is self or not getattr(n, "is_pedestrian", False) or not getattr(n, "is_leader", False):
                continue
            if getattr(n, "injured", False):
                continue

            d = math.hypot(n.x - self.x, n.y - self.y)

            e_j0 = getattr(n, "e0", None)
            if e_j0 is None or np.linalg.norm(e_j0) < 1e-9:
                sp = (n.vx**2 + n.vy**2) ** 0.5
                e_j0 = np.array([n.vx / sp, n.vy / sp]) if sp > 1e-6 else np.zeros(2)

            if np.linalg.norm(e_j0) < 1e-12:
                continue

            if d < best_d:
                best_d = d
                best_dir = _norm(e_j0)
                best_id = getattr(n, "unique_id", id(n))

        return best_dir, best_id

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
