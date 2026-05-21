import random
import numpy as np
import os
import matplotlib.pyplot as plt

def set_random_seed(seed: int):
    """Set random seeds for common libraries to ensure reproducibility."""
    # 1) Python's built-in random module
    random.seed(seed)
    
    # 2) NumPy's RNG
    np.random.seed(seed)
    
    # 3) Python hashing seed (helps with cross-platform reproducibility)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed set to {seed}")

def plot_solution(instance, solution):
    """Visualize the EVRPTW solution with depot, stations, customers, and routes."""
    plt.figure(figsize=(10, 8))
    
    # Plot depot
    plt.scatter(instance.depot.x, instance.depot.y, c='red', marker='s', s=100, label='Depot')
    
    # Plot charging stations
    stations = instance.stations
    xs = [s.x for s in stations]
    ys = [s.y for s in stations]
    plt.scatter(xs, ys, c='green', marker='^', s=80, label='Stations')
    
    # Plot customers
    customers = instance.customers
    xs = [c.x for c in customers]
    ys = [c.y for c in customers]
    plt.scatter(xs, ys, c='blue', marker='o', s=50, label='Customers')
    
    # Plot routes
    colors = plt.cm.tab10.colors
    for i, route in enumerate(solution):
        color = colors[i % len(colors)]
        nodes = route.nodes
        xs = [n.x for n in nodes]
        ys = [n.y for n in nodes]
        plt.plot(xs, ys, '--', color=color, linewidth=1)
        plt.plot(xs[0], ys[0], 'o', color=color, markersize=8)
    
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('EVRPTW Solution Visualization')
    plt.grid(True)
    plt.show()


def back_to_depot(customer, depot, capacity):
    """Raise if any customer is too far to return directly to the depot (using Euclidean distance)."""
    depot_pos = (depot.x, depot.y)
    for i in range(len(customer)):
        customer_pos = (customer[i].x, customer[i].y)  # fixed typo
        battery_use = np.sqrt((customer_pos[0] - depot_pos[0])**2 + (customer_pos[1] - depot_pos[1])**2)
        if battery_use > capacity / 2:
            raise ValueError(f"Customer {i} is too far from depot to return directly.")

def map_param(instance, param, value):
    """Map a symbolic parameter to the instance's vehicle parameters."""
    param_map = {
        'Q': ('fuel_cap', value),
        'C': ('load_cap', value),
        'r': ('consump_rate', value),
        'g': ('charge_rate', 1 / value),  # convert to charging rate (inverse of time per unit)
        'v': ('velocity', value)
    }
    if param in param_map:
        key, val = param_map[param]
        instance.vehicle_params[key] = val

def build_distance_matrix(instance):
    """Build a full Euclidean distance matrix for depot + stations + customers."""
    nodes = np.array([(n.x, n.y) for n in [instance.depot] + instance.stations + instance.customers])
    dx = nodes[:, 0, None] - nodes[:, 0]
    dy = nodes[:, 1, None] - nodes[:, 1]
    instance.dist_matrix = np.sqrt(dx**2 + dy**2)
