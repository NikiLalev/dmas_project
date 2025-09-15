import numpy as np
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from simple_agent import SimplePedestrian
from fire import StaticFire

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
                 dt=0.01,
                 integration_method='rk4',
                 seed=None):
        super().__init__(seed=seed)
        
        self.n_agents = n_agents
        self.width = width
        self.height = height
        self.exit_width = exit_width
        self.dt = dt
        self.integration_method = integration_method
        self.steps = 0
        self.running = True
        
        # Create space
        self.space = ContinuousSpace(width, height, False)
        
        # Define walls and exits (similar to Helbing's setup)
        self._create_geometry()
        
        # Create agents
        self._create_agents()

        # Create fire
        self._place_fire()
        
        # Data collection - could add more metrics
        self.datacollector = DataCollector(
            model_reporters={
                "Agents": lambda m: len(m.agents),
                "Average_Speed": lambda m: np.mean([np.sqrt(a.vx**2 + a.vy**2) 
                                                   for a in m.agents]) if m.agents else 0,
                "Exit_Flow": lambda m: m.n_agents - len(m.agents),
                "Average_Density": self._calculate_density,
            },
            agent_reporters={
                "x": "x",
                "y": "y", 
                "vx": "vx",
                "vy": "vy",
                "Speed": lambda a: np.sqrt(a.vx**2 + a.vy**2),
            }
        )
        
    def _create_geometry(self):
        """Create walls and exits similar to Helbing's setup."""
        # Room boundaries
        self.walls = [
            (0, 0, self.width, 0),      # Bottom wall
            (0, self.height, self.width, self.height),  # Top wall
            (0, 0, 0, self.height),     # Left wall
            (self.width, 0, self.width, self.height),   # Right wall
        ]
        
        # Central wall with exit (like in Helbing paper)
        wall_x = self.width * 0.75  # Wall position
        exit_start = (self.height - self.exit_width) / 2
        exit_end = exit_start + self.exit_width
        
        # Add wall segments above and below exit
        self.walls.extend([
            (wall_x, 0, wall_x, exit_start),           # Wall below exit
            (wall_x, exit_end, wall_x, self.height),   # Wall above exit
        ])
        
        # Define exit
        self.exits = [
            (wall_x, exit_start, wall_x, exit_end)
        ]
        
    def _create_agents(self):
        """Create and place agents randomly in left part of room."""
        for i in range(self.n_agents):
            # Place agents randomly in left 75% of room
            x = self.random.uniform(2.0, (self.width * 0.75) - 2.0)
            y = self.random.uniform(2.0, self.height - 2.0)

            # Initial desired speed from normal distribution with mean 1.3 and std 0.2 in range [0.5, 2.0]
            v0 = self.rng.normal(1.3, 0.2)
            v0 = max(0.5, min(2.0, v0))
            
            agent = SimplePedestrian(
                unique_id=i,
                model=self,
                pos=(x, y),
                v0=v0,
                tau=0.5,
                radius=0.25 + self.random.uniform(0, 0.1),
                mass=80.0
            )
            
            self.space.place_agent(agent, (x, y))

    def _place_fire(self):
        """Create and place fire randomly in room."""
        # Place agents randomly in left 75% of room
        x = self.random.uniform(2.0, (self.width * 0.75) - 1.0)
        y = self.random.uniform(1.0, self.height - 1.0)

        fire = StaticFire(
                model=self,
                pos=(x, y)
            )
            
        self.space.place_agent(fire, (x, y))
    
    def _calculate_density(self):
        """Calculate local density around exit."""
        if not self.agents:
            return 0
            
        exit_area = 4.0  # Area around exit to measure density
        exit_x = self.width * 0.75
        exit_y = self.height / 2
        
        agents_near_exit = 0
        for agent in self.agents:
            dist = np.sqrt((agent.x - exit_x)**2 + (agent.y - exit_y)**2)
            if dist < exit_area:
                agents_near_exit += 1
                
        return agents_near_exit / (np.pi * exit_area**2)
    
    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        
        self.agents.shuffle_do("step")
        
        if len(self.agents) == 0:
            self.running = False
            
        self.steps += 1
    
    def get_agent_positions(self):
        """Get current agent positions for visualization."""
        return [(agent.x, agent.y) for agent in self.agents]
    
    def get_agent_velocities(self):
        """Get current agent velocities for visualization."""
        return [(agent.vx, agent.vy) for agent in self.agents]