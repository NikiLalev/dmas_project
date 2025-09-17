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