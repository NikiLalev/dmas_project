import math
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
    def __init__(self, unique_id, model, pos, v0=1.3, tau=0.5, radius=0.3, mass=80.0, knows_exit=False, herding_radius=3.0, is_leader=False, exit_id=None, visibility_radius=10.0, panic=0.5, vmax=5.0, alpha_imp=0.2):
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
        self.herding_radius = herding_radius # radius for neighbor influence
        self.is_pedestrian = True     #to distinguish from wall (or other) objects
        self.is_leader = is_leader
        self.exit_id = exit_id       # for leaders: fixed exit index
        self.e0 = np.zeros(2)     # desired direction cache
        self.visibility_radius = visibility_radius # for exit/neighbor visibility
        self.injured = False       # becomes True if crushed

        # desired-speed dynamics (Helbing impatience)
        self.v0_init = v0          # remember initial relaxed speed
        self.vmax = vmax  # cap for panic speed (tune)
        self.impatience = 0.0      # low-pass state
        self.alpha_imp = alpha_imp       # smoothing factor (0..1)

        # panic dynamics (herding weight p in [0,1])
        self.panic = panic         # herding factor p ∈ [0,1]
        self.panic_base = self.panic      # starting value (e.g., 0.5)

    def effective_visibility(self):
        """Visibility radius (can be static or read from model's smoke field)."""
        if hasattr(self.model, "visibility_at"):
            return float(self.model.visibility_at(self.x, self.y))
        return self.visibility_radius

    def can_see(self, x, y):
        return np.linalg.norm(np.array([x - self.x, y - self.y])) <= self.effective_visibility()

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
        R = min(self.herding_radius, self.effective_visibility())
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
        R = min(self.herding_radius, self.effective_visibility())
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
        e0 = self.e0 if np.linalg.norm(self.e0) > 1e-12 else np.array([0.0, 0.0])
        vdx, vdy = self.v0 * e0[0], self.v0 * e0[1]
        fx = self.m * (vdx - self.vx) / self.tau
        fy = self.m * (vdy - self.vy) / self.tau
        return fx, fy
    
    def agent_repulsion(self):
        fx, fy = 0.0, 0.0
        R = self.visibility_radius
        for n in self.model.space.get_neighbors((self.x, self.y), R, include_center=False):
            if n is self or not getattr(n, "is_pedestrian", False):
                continue
            dx = self.x - n.x
            dy = self.y - n.y
            d = np.linalg.norm([dx, dy])
            if d < 1e-12:
                continue
            r_ij = self.r + n.r
            nij = np.array([dx / d, dy / d])
            t_ij = np.array([-nij[1], nij[0]])
            delta_v = np.array([n.vx - self.vx, n.vy - self.vy])  # note sign
            delta_v_t = np.dot(delta_v, t_ij)

            # psychological force
            f_ij_psy = self.A * math.exp((r_ij - d) / self.B) * nij
            # body force (only if overlapping)
            f_ij_body = self.k * max(0.0, r_ij - d) * nij
            # sliding friction (only if overlapping)
            f_ij_fric = self.kappa * max(0.0, r_ij - d) * delta_v_t * t_ij

            f_ij = f_ij_psy + f_ij_body + f_ij_fric
            fx += f_ij[0]
            fy += f_ij[1]

        return fx, fy
    
    def wall_repulsion(self):
        fx, fy = 0.0, 0.0
        R = self.visibility_radius
        for wall in self.model.walls:  # wall = (x0, y0, x1, y1)
            x0, y0, x1, y1 = wall
            wx, wy = x1 - x0, y1 - y0
            w_len = math.hypot(wx, wy)
            if w_len < 1e-12:
                continue
            wx_n, wy_n = wx / w_len, wy / w_len

            dx, dy = self.x - x0, self.y - y0
            proj = dx * wx_n + dy * wy_n
            if proj < 0:
                gx, gy = x0, y0
            elif proj > w_len:
                gx, gy = x1, y1
            else:
                gx, gy = x0 + proj * wx_n, y0 + proj * wy_n

            d_vec = np.array([self.x - gx, self.y - gy])
            d = np.linalg.norm(d_vec)
            if d > R or d < 1e-12:
                continue

            n_iw = d_vec / d
            t_iw = np.array([-n_iw[1], n_iw[0]])
            r_iw = self.r
            v_i = np.array([self.vx, self.vy])

            f_iw_psy  = self.A * math.exp((r_iw - d) / self.B) * n_iw
            f_iw_body = self.k * max(0.0, r_iw - d) * n_iw
            f_iw_fric = self.kappa * max(0.0, r_iw - d) * np.dot(v_i, t_iw) * t_iw

            f_iw = f_iw_psy + f_iw_body + f_iw_fric
            fx += f_iw[0]
            fy += f_iw[1]

        return fx, fy
    
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
        self.impatience = (1.0 - self.alpha_imp) * self.impatience + self.alpha_imp * p_imp_raw

        # update desired speed between relaxed v0_init and panic vmax
        self.v0 = (1.0 - self.impatience) * self.v0_init + self.impatience * self.vmax

    def check_injury(self, fx_a, fy_a, fx_w, fy_w):
        """
        Check if pedestrian is injured due to excessive radial pressure.
        fx_a, fy_a = agent repulsion forces
        fx_w, fy_w = wall repulsion forces
        """
        # magnitude of radial forces only
        F_radial = np.linalg.norm([fx_a + fx_w, fy_a + fy_w])
        circumference = 2 * math.pi * self.r
        pressure = F_radial / circumference
        if pressure > 1600.0:  # threshold from Helbing et al.
            self.injured = True
            self.vx = 0.0
            self.vy = 0.0

    def update_panic(self):
        """
        Update panic p in [0,1]
        p = 1 - (1 - p_base)*(1 - vis_term)*(1 - wait_term)
        """
        # visibility term in [0,1]: 0 = clear, 1 = no visibility
        vis = self.effective_visibility()
        vis_ref = max(1e-6, getattr(self.model, "vis_ref", self.visibility_radius))
        vis_term = float(np.clip(1.0 - vis / vis_ref, 0.0, 1.0))

        # waiting term in [0,1]: 0 = moving at v0_init, 1 = stuck
        e_dir = self.e0 if np.linalg.norm(self.e0) > 1e-12 else np.array([0.0, 0.0])
        v_vec = np.array([self.vx, self.vy])
        v_parallel = max(0.0, float(np.dot(v_vec, e_dir)))
        wait_term = float(np.clip(1.0 - (v_parallel / max(self.v0_init, 1e-9)), 0.0, 1.0))

        self.panic = 1.0 - (1.0 - self.panic_base) * (1.0 - vis_term) * (1.0 - wait_term)

    def has_exited(self):
        """
        Check if the pedestrian has crossed any exit segment.
        """
        for (x0, y0, x1, y1) in self.model.exits:
            # vertical exit (x0 == x1)
            if abs(x1 - x0) < 1e-12:
                xe = x0
                if x0 == x1 and min(y0, y1) <= self.y <= max(y0, y1):
                    # exited to the right
                    if self.x >= xe and x1 >= x0:
                        return True
                    # exited to the left
                    if self.x <= xe and x1 <= x0:
                        return True

            # horizontal exit (y0 == y1)
            elif abs(y1 - y0) < 1e-12:
                ye = y0
                if y0 == y1 and min(x0, x1) <= self.x <= max(x0, x1):
                    # exited upwards
                    if self.y >= ye and y1 >= y0:
                        return True
                    # exited downwards
                    if self.y <= ye and y1 <= y0:
                        return True
        return False

    def step(self):
        dt = self.model.dt
        if self.injured:
            return  # injured agents remain static obstacles

        # 1) desired direction
        self.compute_desired_direction()

        # 2) impatience → update v0(t)
        self.update_impatience_and_speed()

        # 3) panic update
        self.update_panic()

        # 4) compute forces separately
        fx_d, fy_d = self.driving_forces()
        fx_a, fy_a = self.agent_repulsion()
        fx_w, fy_w = self.wall_repulsion()

        # check injury from radial pressure
        self.check_injury(fx_a, fy_a, fx_w, fy_w)
        if self.injured:
            return

        # total force = driving + interactions
        fx, fy = fx_d + fx_a + fx_w, fy_d + fy_a + fy_w

        # integrate velocity and position
        ax, ay = fx / self.m, fy / self.m
        self.vx += ax * dt
        self.vy += ay * dt

        # cap instantaneous speed at vmax
        sp = math.hypot(self.vx, self.vy)
        if sp > self.vmax:
            s = self.vmax / sp
            self.vx *= s
            self.vy *= s

        self.x += self.vx * dt
        self.y += self.vy * dt

        # 5) exit check
        if self.has_exited():
            # Logic for removing the pedestrian from the simulation
            pass