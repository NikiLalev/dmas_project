import math
import numpy as np
from mesa import Agent
from utils import _norm

class SimplePedestrian(Agent):
    """
    Simplified pedestrian for Helbing replication.
    Only core social force model - no leaders, panic, etc.
    """
    
    def __init__(self, unique_id, model, pos, v0=1.3, tau=0.5, radius=0.1 , mass=80.0):
        super().__init__(model)
        self.unique_id = unique_id  # agent_id
        self.x, self.y = pos  # (x, y) position
        self._last_x, self._last_y = self.x, self.y
        self.vx, self.vy = 0.0, 0.0  # velocity (m/s)
        self.v0 = v0        # desired speed
        self.tau = tau      # adaptation time - how fast we correct - small means impatient / fast adaptation
        self.r = radius     # body radius
        self.m = mass       # body mass (80 kg typical)
        
        # Helbing parameters (exact values from paper)
        self.A = 2000.0     # interaction strength
        self.B = 0.08       # interaction range
        self.k = 1.2e5      # body force constant
        self.kappa = 2.4e5  # friction constant
        
        self.is_pedestrian = True
        self.injured = False
        
    def nearest_exit_point(self):
        """Find nearest point on any exit."""
        best_point = None
        best_dist = float('inf')
        
        for (x0, y0, x1, y1) in self.model.exits:
            # Project to line segment
            gx, gy = self._project_to_line(self.x, self.y, x0, y0, x1, y1)
            dist = np.linalg.norm([gx - self.x, gy - self.y])
            
            # We iterate over the exits and store the closest point
            if dist < best_dist:
                best_dist = dist
                best_point = (gx, gy)
                
        return best_point
    
    def _project_to_line(self, px, py, x0, y0, x1, y1):
        """
        Project point (px, py) onto line segment from (x0,y0) to (x1,y1).
        
        This finds the closest point on the line segment to the agent.
        Used for both exit navigation and wall avoidance.
        
        Mathematical explanation:
        - We can represent all the points on the line segment as: P(t) = (x0, y0) + t*(dx, dy) where t in [0,1]
        - Finds t that minimizes distance to point
        - Clamps t to [0,1] to stay on segment
        """
        # Pythagorean theorem, for line segment squared length
        dx, dy = x1 - x0, y1 - y0
        length_sq = dx*dx + dy*dy
        
        # Base case: line segment is a point
        if length_sq < 1e-12:
            return x0, y0
        
        # t=0 gives (x0,y0), t=1 gives (x1,y1) | We find the t that minimizes distance to px, py
        t = ((px - x0) * dx + (py - y0) * dy) / length_sq  # Dot product formula
        t = max(0, min(1, t))               # Clamp to segment [0,1]
        
        # return the coordinates of the projected point
        return x0 + t * dx, y0 + t * dy

    def driving_force(self):
        """
        Force pulling agent toward exit at desired speed.
        
        Helbing equation: F_drive = m * (v_desired - v_current) / τ
        """
        e0 = getattr(self, "e0", None)
        if e0 is None or np.linalg.norm(e0) < 1e-12:
            # fallback to nearest exit if no direction set
            gx, gy = self.nearest_exit_point()
            e0 = self.desired_direction(gx, gy)
        vd = self.v0 * e0
        fx = self.m * ((vd[0] - self.vx) / self.tau)
        fy = self.m * ((vd[1] - self.vy) / self.tau)
        return fx, fy
    
    def agent_repulsion(self, R=10.0):
        """Social force from other agents."""
        fx, fy = 0.0, 0.0
        
        for other in self.model.space.get_neighbors((self.x, self.y), R, include_center=False):
            if other is self or not getattr(other, "is_pedestrian", False):
                continue
            
            # this is d_ij in Helbing paper    
            dx, dy = self.x - other.x, self.y - other.y
            d = math.hypot(dx, dy)
            
            if d < 1e-12:
                continue
                
            # Normal and tangent vectors
            nij = np.array([dx/d, dy/d])
            tij = np.array([-nij[1], nij[0]])
            
            # sum of radii
            rij = self.r + other.r
            
            # Tanential velocity
            dvij_t = np.dot(np.array([other.vx - self.vx, other.vy - self.vy]), tij)
            
            # Forces
            # Split the Helbing formula (2) into 3 parts for clarity by multiplying with nij
            f_social = self.A * math.exp((rij - d) / self.B) * nij
            # below, we have max as per the defintion of g() to ensure 0 if no contact aka dist > rij
            f_body   = self.k * max(0.0, rij - d) * nij
            f_fric   = self.kappa * max(0.0, rij - d) * dvij_t * tij
            
            f = f_social + f_body + f_fric
            fx += f[0]; fy += f[1]        
        return fx, fy
    
    def wall_repulsion(self):
        """Social force from walls."""
        fx, fy = 0.0, 0.0
        
        for (x0, y0, x1, y1) in self.model.walls:
            gx, gy = self._project_to_line(self.x, self.y, x0, y0, x1, y1)
            # (dx, dy) is vector from wall to agent
            dx = self.x - gx
            dy = self.y - gy
            # diW - distance from agent to wall
            dist = np.linalg.norm([dx, dy])
            
            if dist < 1e-12:
                continue
            # Unit normal (perpendicular to wall) and tangent vectors (parallel to wall)
            niw = np.array([dx / dist, dy / dist])
            tiw = np.array([-niw[1], niw[0]])
            # Calculate tangential velocity component for f_friction
            vi = np.array([self.vx, self.vy])
            vi_t = np.dot(vi, tiw)
            
            # Three force components as per Helbing formula (3)
            f_social = self.A * math.exp((self.r - dist) / self.B) * niw
            # Again max(0, ...) to model g()
            f_body = self.k * max(0, self.r - dist) * niw
            f_friction = self.kappa * max(0, self.r - dist) * vi_t * tiw
            
            force = f_social + f_body + f_friction
            fx += force[0]
            fy += force[1]
            
        return fx, fy
    
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
    
    
    def calculate_acceleration(self, x, y, vx, vy):
        """
        Calculate acceleration at given state (x, y, vx, vy).
        
        This is separated from step() so RK4 can evaluate forces
        at intermediate positions/velocities.
        
        Args:
            x, y: Position to evaluate forces at
            vx, vy: Velocity to evaluate forces at
            
        Returns:
            (ax, ay): Acceleration components in m/s²
        """
        # Temporarily store current state
        old_x, old_y = self.x, self.y
        old_vx, old_vy = self.vx, self.vy
        
        # Set state to evaluation point
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        
        # Calculate forces at this state
        fx_drive, fy_drive = self.driving_force()
        fx_agent, fy_agent = self.agent_repulsion()
        fx_wall, fy_wall = self.wall_repulsion()
        
        # Total force
        fx = fx_drive + fx_agent + fx_wall
        fy = fy_drive + fy_agent + fy_wall
        
        # Restore original state
        self.x, self.y = old_x, old_y
        self.vx, self.vy = old_vx, old_vy
        
        # Return acceleration F = ma → a = F/m
        return fx / self.m, fy / self.m
    
    # --- hooks for extension ---
    def pre_physics_update(self):
        """Called at start of step before forces are computed (override in subclass)."""
        pass

    def desired_direction(self, gx, gy):
        return _norm(np.array([gx - self.x, gy - self.y]))
    
    def check_injury(self, fx_a, fy_a, fx_w, fy_w):
        """
        Check if pedestrian is injured:
        - due to radial pressure (Helbing et al.)
        - if the pedestrian enters the fire area
        """
        # 1) Injury from radial pressure
        F_radial = np.linalg.norm([fx_a + fx_w, fy_a + fy_w])
        circumference = 2 * math.pi * self.r
        pressure = F_radial / circumference
        if pressure > 1600.0:  # threshold from Helbing et al.
            self.injured = True
            self.vx = 0.0
            self.vy = 0.0
            return  # already injured, stop here

        # 2) Injury from fire contact
        fire = getattr(self.model, "fire", None)
        if fire and fire.is_inside_fire((self.x, self.y), self.r):
            self.injured = True
            self.vx = 0.0
            self.vy = 0.0

    def step_rk4(self):
        """
        One simulation time step using 4th-order Runge–Kutta.
        We check injury once at the *current* state using contact forces,
        then integrate with RK4 (forces must be recomputed at sub-states).
        """
        dt = self.model.dt

        # If already injured from a previous step, do nothing
        if self.injured:
            return

        # Keep last position (used by has_exited line-crossing)
        self._last_x, self._last_y = self.x, self.y

        # Model-specific pre-update (ExtendedPedestrian updates e0, v0, panic, ...)
        self.pre_physics_update()

        # --- Contact forces ONCE at the current state (reuse for injury only) ---
        fx_agent, fy_agent = self.agent_repulsion()
        fx_wall,  fy_wall  = self.wall_repulsion()

        # Injury check (may set self.injured = True)
        self.check_injury(fx_agent, fy_agent, fx_wall, fy_wall)
        if self.injured:
            return

        # --- RK4 integration (forces will be recomputed at sub-states inside calculate_acceleration) ---
        x0, y0 = self.x, self.y
        vx0, vy0 = self.vx, self.vy

        # k1 at (x0, y0, vx0, vy0)
        ax1, ay1 = self.calculate_acceleration(x0, y0, vx0, vy0)
        k1_x,  k1_y  = vx0 * dt, vy0 * dt
        k1_vx, k1_vy = ax1 * dt, ay1 * dt

        # k2 at midpoint using k1
        x_mid1  = x0  + 0.5 * k1_x
        y_mid1  = y0  + 0.5 * k1_y
        vx_mid1 = vx0 + 0.5 * k1_vx
        vy_mid1 = vy0 + 0.5 * k1_vy
        ax2, ay2 = self.calculate_acceleration(x_mid1, y_mid1, vx_mid1, vy_mid1)
        k2_x,  k2_y  = vx_mid1 * dt, vy_mid1 * dt
        k2_vx, k2_vy = ax2 * dt,  ay2 * dt

        # k3 at midpoint using k2
        x_mid2  = x0  + 0.5 * k2_x
        y_mid2  = y0  + 0.5 * k2_y
        vx_mid2 = vx0 + 0.5 * k2_vx
        vy_mid2 = vy0 + 0.5 * k2_vy
        ax3, ay3 = self.calculate_acceleration(x_mid2, y_mid2, vx_mid2, vy_mid2)
        k3_x,  k3_y  = vx_mid2 * dt, vy_mid2 * dt
        k3_vx, k3_vy = ax3 * dt,  ay3 * dt

        # k4 at endpoint using k3
        x_end  = x0  + k3_x
        y_end  = y0  + k3_y
        vx_end = vx0 + k3_vx
        vy_end = vy0 + k3_vy
        ax4, ay4 = self.calculate_acceleration(x_end, y_end, vx_end, vy_end)
        k4_x,  k4_y  = vx_end * dt, vy_end * dt
        k4_vx, k4_vy = ax4 * dt,  ay4 * dt

        # RK4 combination
        self.vx += (k1_vx + 2.0*k2_vx + 2.0*k3_vx + k4_vx) / 6.0
        self.vy += (k1_vy + 2.0*k2_vy + 2.0*k3_vy + k4_vy) / 6.0
        self.x  += (k1_x  + 2.0*k2_x  + 2.0*k3_x  + k4_x ) / 6.0
        self.y  += (k1_y  + 2.0*k2_y  + 2.0*k3_y  + k4_y ) / 6.0

        # Speed cap
        speed = math.hypot(self.vx, self.vy)
        if speed > 10.0:
            s = 10.0 / speed
            self.vx *= s
            self.vy *= s

        # Exit check (remove agent if crossed an exit segment)
        if self.has_exited():
            self.model.space.remove_agent(self)
            self.model.agents.remove(self)

    def step_euler(self):
        """
        Simple and fast Euler integration
        #     1. Calculate all forces (driving + repulsion)
        #     2. Update velocity: v_new = v_old + (F/m) * dt
        #     3. Limit speed to realistic maximum (10 m/s)
        #     4. Update position: x_new = x_old + v * dt
        #     5. Check for evacuation completion
        """
        dt = self.model.dt

        # Already injured from previous steps → skip everything
        if self.injured:
            return

        self._last_x, self._last_y = self.x, self.y
        self.pre_physics_update()

        # Contact forces (used also for injury check)
        fx_agent, fy_agent = self.agent_repulsion()
        fx_wall,  fy_wall  = self.wall_repulsion()

        # Injury check (may set self.injured = True)
        self.check_injury(fx_agent, fy_agent, fx_wall, fy_wall)

        # If injured during this step, stop here
        if self.injured:
            return

        # Add driving force and integrate
        fx_drive, fy_drive = self.driving_force()
        fx = fx_drive + fx_agent + fx_wall
        fy = fy_drive + fy_agent + fy_wall

        self.vx += (fx / self.m) * dt
        self.vy += (fy / self.m) * dt

        # Speed cap
        speed = math.hypot(self.vx, self.vy)
        if speed > 10.0:
            self.vx = self.vx / speed * 10.0
            self.vy = self.vy / speed * 10.0

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Exit check
        if self.has_exited():
            self.model.space.remove_agent(self)
            self.model.agents.remove(self)
    
    def step(self):
        """
        Choose integration method based on model settings
        """
        # Use RK4 by default, fallback to Euler if needed
        integration_method = getattr(self.model, 'integration_method', 'euler')
        
        if integration_method == 'rk4':
            self.step_rk4()
        else:
            self.step_euler()
