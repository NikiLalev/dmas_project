# Agent-Based Simulation of Panic-Driven Crowd Evacuation

Agent-based simulation of evacuation dynamics under reduced visibility, built on the Mesa framework and extended with leader–follower interactions, dynamic fire propagation, and smoke-driven visibility loss.

---

## Features

- **Leader–follower modeling** with dynamic herding and panic behaviour  
- **Environmental hazards**: fire, smoke diffusion, and visibility decay  
- **Flexible configuration** of exits, crowd size, and agent parameters  
- **Interactive visualization** via [Solara](https://solara.dev)  
- **Data collection** for evacuation time, injuries, and behavioural metrics  

---

## Requirements

- **Python** ≥ 3.12.10  
- Dependencies listed in `requirements.txt`

---

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd dmas_project
   ```

2.	**Create a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate         # Windows
   ```

3.	**Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Launch interactive visualization (Solara)

   ```bash
   python -m solara run src/mesa_visualization.py
   ```