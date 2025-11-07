from mesa.visualization import SolaraViz
from mesa.visualization.components.matplotlib_components import make_mpl_plot_component
from mesa.visualization.utils import update_counter
import solara
from matplotlib.figure import Figure
from model import EvacuationModel
from matplotlib.patches import Circle


@solara.component
def RoomSpace(model):
    update_counter.get()
    fig = Figure()
    ax = fig.add_subplot()

    # 1) Draw agents manually as circles with physical radius
    # Try to get an iterable of agents in a robust way
    if (
        hasattr(model, "schedule")
        and getattr(model.schedule, "agents", None) is not None
    ):
        agents_iter = list(model.schedule.agents)
    elif hasattr(model, "agents"):
        agents_iter = list(model.agents)
    else:
        try:
            agents_iter = list(model.space._agents)
        except Exception:
            agents_iter = []

    # Draw each agent as a Circle patch using physical radius agent.r
    for agent in agents_iter:
        if getattr(agent, "is_fire", False):
            continue

        # read position and radius (fallback default)
        x = float(getattr(agent, "x", 0.0))
        y = float(getattr(agent, "y", 0.0))
        r = float(getattr(agent, "r", 0.2))

        # color / style choices
        if getattr(agent, "injured", False):
            face = "red"
            edge = "black"
            alpha = 0.95
            z = 3.5
            size = max(30, r)
            ax.scatter(
                [x],
                [y],
                s=size,
                marker="x",
                color="black",
                linewidths=1.2,
                zorder=5,
                alpha=0.95,
            )
        else:
            face = "tab:orange" if getattr(agent, "is_leader", False) else "tab:blue"
            edge = "black"
            alpha = 0.85
            z = 3.5

        # Create circle in data coordinates (radius in same units as axis)
        circ = Circle(
            (x, y),
            r,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.25,
            alpha=alpha,
            zorder=z,
        )
        ax.add_patch(circ)

        # Add small marker for orientation or velocity (tiny line)
        vx = getattr(agent, "vx", 0.0)
        vy = getattr(agent, "vy", 0.0)
        speed = (vx**2 + vy**2) ** 0.5
        if speed > 1e-6:
            # draw a short line showing heading, length scaled with radius
            hx = x + (vx / speed) * r * 1.6
            hy = y + (vy / speed) * r * 1.6
            ax.plot([x, hx], [y, hy], color="black", linewidth=0.6, zorder=z + 0.1)

    fire = getattr(model, "fire", None)
    if fire is not None:
        smoke = Circle(
            (float(fire.x), float(fire.y)),
            float(fire.r_smoke),
            facecolor="gray",
            edgecolor="none",
            alpha=0.5,
            zorder=0,
        )
        ax.add_patch(smoke)

        core = Circle(
            (float(fire.x), float(fire.y)),
            float(fire.r),
            facecolor="red",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            zorder=1,
        )
        ax.add_patch(core)

    for x0, y0, x1, y1 in getattr(model, "walls", []):
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=3, zorder=4, alpha=0.9)

    for x0, y0, x1, y1 in getattr(model, "exits", []):
        ax.plot([x0, x1], [y0, y1], color="green", linewidth=6, zorder=4, alpha=0.9)

    W, H = getattr(model, "width", None), getattr(model, "height", None)
    if W and H:
        ax.set_xlim(-0.5, W + 0.5)
        ax.set_ylim(-0.5, H + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title("Evacuation (continuous, with walls & exits)")

    return solara.FigureMatplotlib(fig, format="png", bbox_inches="tight")


agents_plot, _ = make_mpl_plot_component("Agents")
speed_plot, _ = make_mpl_plot_component("Average_Speed")
flow_plot, _ = make_mpl_plot_component("Exit_Flow_Total", page=1)

model_params = {
    "dt": {
        "type": "SliderFloat",
        "value": 0.10,
        "min": 0.01,
        "max": 0.10,
        "step": 0.005,
        "label": "Δt",
    },
    "n_agents": {
        "type": "SliderInt",
        "value": 50,
        "min": 5,
        "max": 200,
        "step": 5,
        "label": "Agents",
    },
    "num_leaders": {
        "type": "SliderInt",
        "value": 10,
        "min": 0,
        "max": 10,
        "step": 1,
        "label": "Leaders",
    },
    "num_exits": {
        "type": "SliderInt",
        "value": 1,
        "min": 1,
        "max": 3,
        "step": 1,
        "label": "Exits",
    },
    "width": {
        "type": "SliderFloat",
        "value": 15.0,
        "min": 10.0,
        "max": 40.0,
        "step": 1.0,
        "label": "Room width",
    },
    "height": {
        "type": "SliderFloat",
        "value": 15.0,
        "min": 8.0,
        "max": 30.0,
        "step": 1.0,
        "label": "Room height",
    },
    "exit_width": {
        "type": "SliderFloat",
        "value": 2.0,
        "min": 0.4,
        "max": 3.0,
        "step": 0.1,
        "label": "Exit width",
    },
    "agent_type": {
        "type": "Select",
        "value": "extended",
        "values": ["simple", "extended"],
        "label": "Agent Type",
    },
    "integration_method": {
        "type": "Select",
        "value": "euler",
        "values": ["euler", "rk4"],
        "label": "Integration Method",
    },
    "enable_fire": {"type": "Checkbox", "value": False, "label": "Enable Fire"},
    "exit_preset": {
        "type": "Select",
        "value": "three_exits",
        "values": [
            "random",
            "center_bottom",
            "center_right",
            "center_left",
            "two_exits",
            "three_exits",
        ],
        "label": "Exit Configuration",
    },
}

viz = SolaraViz(
    EvacuationModel(
        n_agents=50, width=15, height=15, dt=0.10, exit_width=2, num_exits=1
    ),
    components=[RoomSpace, agents_plot, speed_plot, flow_plot],
    model_params=model_params,
    name="Panic-Driven Evacuation",
    play_interval=1,
    render_interval=10,
    use_threads=False,
)

app = viz
