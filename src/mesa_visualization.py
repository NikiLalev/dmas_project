from mesa.visualization import SolaraViz
from mesa.visualization.components.matplotlib_components import make_mpl_plot_component
from mesa.visualization.mpl_space_drawing import draw_space
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.utils import update_counter
import solara
from matplotlib.figure import Figure
from model import EvacuationModel

def agent_portrayal(agent):
    if getattr(agent, "is_fire", False):
        return AgentPortrayalStyle(
            x=float(agent.x), y=float(agent.y),
            color="red", size=140, alpha=0.9,
            edgecolors="black", linewidths=0.2,
        )

    is_leader = bool(getattr(agent, "is_leader", False))
    color = "tab:orange" if is_leader else "tab:blue"
    r = float(getattr(agent, "r", 0.2))
    size = max(30, 1200 * r)  # Point size in screen pixels
    return AgentPortrayalStyle(
        x=float(agent.x), y=float(agent.y),
        color=color, size=size, alpha=0.85,
        edgecolors="black", linewidths=0.2,
    )

@solara.component
def RoomSpace(model):
    update_counter.get()
    fig = Figure()
    ax = fig.add_subplot()

    # Draw the agents
    space = getattr(model, "grid", None) or getattr(model, "space", None)
    draw_space(space, agent_portrayal, ax=ax)

    # Draw the walls, list elements: (x0, y0, x1, y1)
    walls = getattr(model, "walls", [])
    for (x0, y0, x1, y1) in walls:
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=3, zorder=0, alpha=0.9)

    # Draw the exits (the green line), list elements: (x0, y0, x1, y1)
    exits = getattr(model, "exits", [])
    for (x0, y0, x1, y1) in exits:
        ax.plot([x0, x1], [y0, y1], color="green", linewidth=6, zorder=0, alpha=0.9)

    # Unified coordinate scale/margin/grid
    W = getattr(model, "width", None)
    H = getattr(model, "height", None)
    if W and H:
        ax.set_xlim(-0.5, W + 0.5)
        ax.set_ylim(-0.5, H + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title("Evacuation (continuous, with walls & exits)")

    return solara.FigureMatplotlib(fig, format="png", bbox_inches="tight")

# Statistics
agents_plot, _ = make_mpl_plot_component("Agents")                 # Agents Remaining
speed_plot,  _ = make_mpl_plot_component("Average_Speed")          # Average Speed
flow_plot,   _ = make_mpl_plot_component("Exit_Flow", page=1)      # Agents Evacuated
dens_plot,   _ = make_mpl_plot_component("Average_Density", page=1)# Average Density

# It's the parameters we can control in the system. Of course, we can expand it.
model_params = {
    "n_agents":     {"type": "SliderInt",   "value": 40,   "min": 5,   "max": 200, "step": 5,   "label": "Agents"},
    "width":        {"type": "SliderFloat", "value": 20.0, "min": 10., "max": 40., "step": 1.,  "label": "Room width"},
    "height":       {"type": "SliderFloat", "value": 15.0, "min":  8., "max": 30., "step": 1.,  "label": "Room height"},
    "dt":           {"type": "SliderFloat", "value": 0.04, "min": 0.01,"max": 0.10,"step":0.005,"label": "Δt"},
    "exit_width":   {"type": "SliderFloat", "value": 1.2,  "min": 0.4, "max": 3.0, "step": 0.1, "label": "Exit width"},
}

# Initialize the model
model = EvacuationModel(n_agents=40, width=20, height=15, dt=0.04)

viz = SolaraViz(
    model,
    components=[
        RoomSpace,
        agents_plot, speed_plot,
        flow_plot, dens_plot
    ],
    model_params=model_params,
    name="Panic-Driven Evacuation",
    play_interval=100,   # Milliseconds: Autoplay step interval
    render_interval=1,   # Refresh the curve every few steps
    use_threads=False,
)

app = viz