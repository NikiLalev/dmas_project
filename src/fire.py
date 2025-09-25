import math
import numpy as np
from mesa.discrete_space import FixedAgent

class StaticFire(FixedAgent):
    """
    Static fire agent:
    - static position (x, y), zero velocity (vx, vy)
    - radius of the fire (radius)
    - acts as an obstacle
    """
    def __init__(self, model, pos, radius=0.5):

        super().__init__(model)
        self.x, self.y = pos
        self.vx, self.vy = 0.0, 0.0

        self.traversable = False    # cannot be crossed
        self.r = radius        # radius (m)
        self.color = "red"          # color for visualisation

    def get_position(self):
        return (self.x, self.y)
    
    def step(self):
        pass


class DynamicFire(FixedAgent):
    """
    Dynamic fire agent:
    - static position (x, y), zero velocity (vx, vy)
    - fire radius
    - creates a surrounding smoke area that is traversable
    """
    def __init__(self, model, pos, initial_fire_radius=0.5, initial_smoke_radius=0.6, 
                 smoke_growth_rate=0.01, smoke_density=0.7):

        super().__init__(model)
        self.x, self.y = pos
        self.vx, self.vy = 0.0, 0.0

        self.r = initial_fire_radius      # fire radius (m)
        self.traversable = False               # cannot be crossed
        self.color = "red"                     # color for visualisation

        self.r_smoke = initial_smoke_radius    # initial smoke radius
        self.smoke_growth_rate = smoke_growth_rate
        self.smoke_traversable = True          # can be crossed
        self.smoke_density = smoke_density     # for agent visability
        self.smoke_color = "gray"              # color for visualisation

    def get_position(self):
        return (self.x, self.y)
    
    def step(self): 
        self.r_smoke += self.smoke_growth_rate
        print(f"Step {self.model.steps}: Fire core = {self.r:.2f}, Smoke = {self.r_smoke:.2f}")

    def is_inside_fire(self, pos):
        """
        Check if a given position is inside the fire area.
        """
        dx = pos[0] - self.x
        dy = pos[1] - self.y
        return math.sqrt(dx*dx + dy*dy) <= self.r

    def is_inside_smoke(self, pos):
        """
        Check if a given position is inside the smoke area but NOT inside the fire.
        """
        dx = pos[0] - self.x
        dy = pos[1] - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        return self.r < dist <= self.r