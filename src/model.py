import numpy as np
import math
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from networkx.classes import neighbors

from simple_agent import SimplePedestrian
from extended_agent import ExtendedPedestrian
from fire import DynamicFire


class EvacuationModel(Model):
    """
    Simple evacuation model replicating Helbing's basic scenario.
    Rectangular room with central wall and exit.
    """

    def __init__(self,
                 n_agents=200,
                 width=20.0,
                 height=15.0,
                 exit_width=1.2,
                 num_exits=1,
                 num_leaders=1,
                 dt=0.01,
                 integration_method='rk4',
                 vis_ref=10.0,
                 smoke_exposure_threshold=15.0,
                 seed=None):
        super().__init__(seed=seed)

        self.n_agents = n_agents
        self.width = width
        self.height = height
        self.exit_width = exit_width
        self.num_exits = max(1, min(3, int(num_exits)))
        self.num_leaders = max(0, min(self.n_agents, int(num_leaders)))
        self.dt = dt
        self.integration_method = integration_method
        self.smoke_exposure_threshold = float(smoke_exposure_threshold)
        self.vis_ref = vis_ref
        self.steps = 0
        self.running = True

        # Create space
        self.space = ContinuousSpace(width, height, False)

        # Define walls and exits (similar to Helbing's setup)
        self._create_geometry()

        # Create agents
        self._create_extended_agents()

        # Create fire
        self._place_fire()

        # Data collection - could add more metrics
        self.datacollector = DataCollector(
            model_reporters={
                "Agents": lambda m: len(m.agents),
                "Average_Speed": lambda m: np.mean([np.sqrt(a.vx ** 2 + a.vy ** 2)
                                                    for a in m.agents]) if m.agents else 0,
                "Exit_Flow": lambda m: m.n_agents - len(m.agents),
                "Average_Density": self._calculate_density,
            },
            agent_reporters={
                "x": "x",
                "y": "y",
                "vx": "vx",
                "vy": "vy",
                "Speed": lambda a: np.sqrt(a.vx ** 2 + a.vy ** 2),
            }
        )

    def _create_geometry(self):
        """Create random exits on the room perimeter and walls excluding those exits."""
        W, H = self.width, self.height
        w = self.exit_width

        exits = []  # list of (x0,y0,x1,y1)
        attempts = 0
        max_attempts = 200
        min_gap = 0.4  # min gap between exits on same wall

        def make_exit_segment(side, c0, c1):
            if side == "bottom":  # y=0, x in [c0,c1]
                return (c0, 0.0, c1, 0.0)
            if side == "top":     # y=H, x in [c0,c1]
                return (c0, H, c1, H)
            if side == "left":    # x=0, y in [c0,c1]
                return (0.0, c0, 0.0, c1)
            if side == "right":   # x=W, y in [c0,c1]
                return (W, c0, W, c1)
            raise ValueError

        sides = ["bottom", "top", "left", "right"]
        safe_margin = 0.75  

        chosen = []  # (side, start, end)

        while len(chosen) < self.num_exits and attempts < max_attempts:
            attempts += 1
            side = self.random.choice(sides)
            if side in ("bottom", "top"):
                L = W
            else:
                L = H

            if L <= w + 2 * safe_margin:
                continue  # not enough space

            start = self.random.uniform(safe_margin, L - safe_margin - w)
            end = start + w

            # no-overlap check
            ok = True
            for s2, a2, b2 in chosen:
                if s2 != side:
                    continue

                if not (end + min_gap <= a2 or b2 + min_gap <= start):
                    ok = False
                    break
            if not ok:
                continue

            chosen.append((side, start, end))

        # fallback
        for side, a, b in chosen:
            exits.append(make_exit_segment(side, a, b))

        self.exits = exits

        # 2) Walls - the room perimeter minus the exits
        def subtract_intervals(L, intervals):
            """Return the complementary intervals in [0,L] after removing 'intervals'."""
            intervals = sorted(intervals)
            pieces = []
            cur = 0.0
            for s, e in intervals:
                s = max(0.0, s); e = min(L, e)
                if s > cur:
                    pieces.append((cur, s))
                cur = max(cur, e)
            if cur < L:
                pieces.append((cur, L))
            return pieces

        map_int = {"bottom": [], "top": [], "left": [], "right": []}
        for (x0, y0, x1, y1) in self.exits:
            if y0 == 0.0 and y1 == 0.0:                  # bottom
                map_int["bottom"].append((min(x0, x1), max(x0, x1)))
            elif y0 == H and y1 == H:                    # top
                map_int["top"].append((min(x0, x1), max(x0, x1)))
            elif x0 == 0.0 and x1 == 0.0:                # left
                map_int["left"].append((min(y0, y1), max(y0, y1)))
            elif x0 == W and x1 == W:                    # right
                map_int["right"].append((min(y0, y1), max(y0, y1)))

        walls = []

        # bottom: y=0, x in [0,W]\exits
        for s, e in subtract_intervals(W, map_int["bottom"]):
            if e - s > 1e-9:
                walls.append((s, 0.0, e, 0.0))

        # top: y=H
        for s, e in subtract_intervals(W, map_int["top"]):
            if e - s > 1e-9:
                walls.append((s, H, e, H))

        # left: x=0, y in [0,H]
        for s, e in subtract_intervals(H, map_int["left"]):
            if e - s > 1e-9:
                walls.append((0.0, s, 0.0, e))

        # right: x=W
        for s, e in subtract_intervals(H, map_int["right"]):
            if e - s > 1e-9:
                walls.append((W, s, W, e))

        self.walls = walls

    def _create_simple_agents(self):
        """Create and place agents randomly in left part of room."""
        max_attempts_per_agent = 300
        placement_margin = 0.02
        max_other_r = 0.35

        for i in range(self.n_agents):
            radius = 0.25 + self.random.uniform(0, 0.1)
            attempts = 0
            placed = False

            while attempts < max_attempts_per_agent and not placed:
                attempts += 1
                # Place agents randomly in left 75% of room
                x = self.random.uniform(2.0, (self.width * 0.75) - 2.0)
                y = self.random.uniform(2.0, self.height - 2.0)

                query_radius = radius + max_other_r + placement_margin

                if len(self.space.agents) > 0:
                    neighbors = self.space.get_neighbors((x, y), radius=query_radius, include_center=False)
                else:
                    neighbors = []

                conflict = False
                for nb in neighbors:
                    # skip non-pedestrian objects if they don't have r attribute
                    nb_r = float(getattr(nb, "r", 0.0))
                    dx = x - nb.x
                    dy = y - nb.y
                    dist = math.hypot(dx, dy)
                    if dist < (radius + nb_r + placement_margin):
                        conflict = True
                        break

                if not conflict:
                    # Initial desired speed from normal distribution with mean 1.3 and std 0.2 in range [0.5, 2.0]
                    v0 = self.random.normalvariate(1.3, 0.2)
                    v0 = max(0.5, min(2.0, v0))

                    agent = SimplePedestrian(
                        unique_id=i,
                        model=self,
                        pos=(x, y),
                        v0=v0,
                        tau=0.5,
                        radius=radius,
                        mass=80.0
                    )

                    self.space.place_agent(agent, (x, y))
                    self.agents.add(agent)
                    placed = True

            if not placed:
                # Initial desired speed from normal distribution with mean 1.3 and std 0.2 in range [0.5, 2.0]
                v0 = self.random.normalvariate(1.3, 0.2)
                v0 = max(0.5, min(2.0, v0))

                agent = SimplePedestrian(
                    unique_id=i,
                    model=self,
                    pos=(x, y),
                    v0=v0,
                    tau=0.5,
                    radius=radius,
                    mass=80.0
                )

                self.space.place_agent(agent, (x, y))
                self.agents.add(agent)

    def _create_extended_agents(self):
        """
        Create and place agents randomly in the left part of the room, 
        using ExtendedPedestrian with randomized parameters.
        """
        max_attempts_per_agent = 300
        placement_margin = 0.02
        max_other_r = 0.35

        leader_ids = set(self.random.sample(range(self.n_agents), self.num_leaders)) if self.n_agents > 0 else set()

        # number of available exits
        n_exits = len(getattr(self, "exits", []))
        leader_assigned = 0

        for i in range(self.n_agents):
            # base body radius with small random variability and mass randomization
            mass = self.random.uniform(60.0, 100.0)
            radius = 0.25 + self.random.uniform(0, 0.1)
            attempts = 0
            placed = False

            # --- ExtendedPedestrian parameters (randomized where it makes sense) ---

            # desired initial speed (truncated normal distribution)
            v0 = self.random.normalvariate(1.3, 0.2)
            v0 = max(0.5, min(2.0, v0))

            # leader or follower
            is_leader = i in leader_ids
            knows_exit = bool(is_leader)  # leaders know an exit from the beginning

            # leaders fix an exit, followers will discover dynamically
            if is_leader and n_exits > 0:
                exit_id = leader_assigned % n_exits
                leader_assigned += 1
            else:
                exit_id = None

            # how strongly the agent follows nearby pedestrians
            herding_radius = self.random.uniform(2.5, 6.0)

            # personal visibility radius cap (environment may reduce it)
            visibility_radius = self.random.uniform(3.0, 6.0)

            # baseline panic predisposition
            panic_base = self.random.uniform(0.0, 1.0)

            # maximum speed when in panic
            vmax = self.random.uniform(3.5, 5.0)

            smoke_recovery_rate = self.random.uniform(0.05, 0.2)

            # Try multiple times to place without overlapping other agents
            while attempts < max_attempts_per_agent and not placed:
                attempts += 1

                # random position in the left 3/4 of the room
                margin = 0.5
                x = self.random.uniform(margin, self.width - margin)
                y = self.random.uniform(margin, self.height - margin)

                query_radius = radius + max_other_r + placement_margin
                if len(getattr(self.space, "agents", [])) > 0:
                    nbs = self.space.get_neighbors((x, y), radius=query_radius, include_center=False)
                else:
                    nbs = []

                # check collisions with existing agents
                conflict = False
                for nb in nbs:
                    nb_r = float(getattr(nb, "r", 0.0))
                    dx = x - nb.x
                    dy = y - nb.y
                    dist = math.hypot(dx, dy)
                    if dist < (radius + nb_r + placement_margin):
                        conflict = True
                        break

                # if no conflicts, place agent
                if not conflict:
                    agent = ExtendedPedestrian(
                        unique_id=i,
                        model=self,
                        pos=(x, y),
                        v0=v0,
                        tau=0.5,
                        radius=radius,
                        mass=mass,
                        smoke_recovery_rate=smoke_recovery_rate,
                        knows_exit=knows_exit,
                        herding_radius=herding_radius,
                        is_leader=is_leader,
                        exit_id=exit_id,
                        visibility_radius=visibility_radius,
                        panic=panic_base,
                        vmax=vmax,
                    )
                    self.space.place_agent(agent, (x, y))
                    self.agents.add(agent)
                    placed = True

            # fallback: if max attempts failed, still place the agent
            if not placed:
                agent = ExtendedPedestrian(
                    unique_id=i,
                    model=self,
                    pos=(x, y),
                    v0=v0,
                    tau=0.5,
                    radius=radius,
                    mass=mass,
                    knows_exit=knows_exit,
                    herding_radius=herding_radius,
                    is_leader=is_leader,
                    exit_id=exit_id,
                    visibility_radius=visibility_radius,
                    panic=panic_base,
                    vmax=vmax,
                )
                self.space.place_agent(agent, (x, y))
                self.agents.add(agent)

    def _place_fire(self):
        """Create and place fire randomly in room."""
        # Place agents randomly in left 75% of room
        x = self.random.uniform(2.0, (self.width * 0.75) - 1.0)
        y = self.random.uniform(1.0, self.height - 1.0)

        fire = DynamicFire(
            model=self,
            pos=(x, y)
        )

        self.space.place_agent(fire, (x, y))

        # reference to the object
        self.fire = fire

    def _calculate_density(self):
        """Calculate local density around exit."""
        if not self.agents:
            return 0

        exit_area = 4.0  # Area around exit to measure density
        exit_x = self.width * 0.75
        exit_y = self.height / 2

        agents_near_exit = 0
        for agent in self.agents:
            dist = np.sqrt((agent.x - exit_x) ** 2 + (agent.y - exit_y) ** 2)
            if dist < exit_area:
                agents_near_exit += 1

        return agents_near_exit / (np.pi * exit_area ** 2)

    def visibility_at(self, x, y):
        """Calculate visibility at position (x, y) considering fire and smoke."""
        base = self.vis_ref
        fire = getattr(self, "fire", None)
        if fire is None:
            return base

        dx, dy = x - fire.x, y - fire.y
        d = (dx * dx + dy * dy) ** 0.5

        if d <= fire.r_smoke:
            smoke_min = 2.0
            return smoke_min + (base - smoke_min) * (d / max(fire.r_smoke, 1e-6))
        return base

    def step(self):
        """Advance the model by one step."""

        if hasattr(self, "fire") and self.fire is not None:
            self.fire.step()

        self.agents.shuffle_do("step")

        self.datacollector.collect(self)

        if len(self.agents) == 0:
            self.running = False

        self.steps += 1

    def get_agent_positions(self):
        """Get current agent positions for visualization."""
        return [(agent.x, agent.y) for agent in self.agents]

    def get_agent_velocities(self):
        """Get current agent velocities for visualization."""
        return [(agent.vx, agent.vy) for agent in self.agents]
