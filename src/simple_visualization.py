import matplotlib.pyplot as plt
import numpy as np
from model import EvacuationModel

def simple_real_time_viz():
    # Simple visualization with matplotlib
    print("Starting real-time evacuation visualization...")
    
    # Create model
    width, height = 20.0, 15.0
    model = EvacuationModel(n_agents=20, width=width, height=height, dt=0.04, integration_method='euler')

    exit_start = (model.height - model.exit_width) / 2
    exit_end = exit_start + model.exit_width
    
    # Set up plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    step_count = 0
    max_steps = 10000
    colorbar_added = False
    
    print(f"Watch {len(model.agents)} agents evacuate!")
    print("Close the plot window to stop.")
    
    try:
        while model.running and step_count < max_steps and plt.get_fignums():

            model.step()
            step_count += 1
            
            # Update visualization every few steps
            if step_count % 8 == 0:
                ax.clear()
                
                # Set up plot
                ax.set_xlim(-0.5, width + 0.5)
                ax.set_ylim(-0.5, height + 0.5)
                ax.set_aspect('equal')
                ax.set_title(f'Evacuation - Step {step_count} - Agents: {len(model.agents)}', fontsize=14)
                
                # Draw room boundaries
                ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'black', linewidth=3, label='Room walls')

                # Draw central wall (with gap for exit)
                ax.plot([15, 15], [0, exit_start], 'black', linewidth=3)  # Below exit
                ax.plot([15, 15], [exit_end, 15], 'black', linewidth=3)   # Above exit
                
                # Draw exit
                ax.plot([15, 15], [exit_start, exit_end], 'green', linewidth=6, label='EXIT')
                
                
                # Draw agents
                if model.agents:
                    positions = model.get_agent_positions()
                    velocities = model.get_agent_velocities()

                    if positions:
                        x_coords, y_coords = zip(*positions)
                        speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in velocities]
                        
                        # Get agent radii and convert to scatter plot sizes
                        radii = [agent.r for agent in model.agents]
                        # Scaling factor to make dots visible but proportional
                        size_scale = 1000
                        sizes = [np.pi * r**2 * size_scale for r in radii]

                        # Simple colored dots
                        scatter = ax.scatter(x_coords, y_coords, 
                                           s=sizes, c=speeds, cmap='plasma', 
                                           alpha=0.7)
                        # Add colorbar once
                        if not colorbar_added:
                            plt.colorbar(scatter, ax=ax, label='Speed (m/s)')
                            colorbar_added = True
                
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                
                # Update display
                plt.draw()
                plt.pause(0.02)
                
                # Simple progress info
                if step_count % 40 == 0:
                    print(f"Step {step_count}: {len(model.agents)} agents remaining")
    
    except KeyboardInterrupt:
        print("Stopped by user.")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        plt.ioff()
        print(f"\nSimulation complete!")
        print(f"Agents evacuated: {model.n_agents - len(model.agents)}/{model.n_agents}")
        plt.show()

if __name__ == "__main__":
    simple_real_time_viz()
