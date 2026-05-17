# evrptw_gen/utils/timing.py
from __future__ import annotations
import numpy as np

def minutes_from_distance(distance, speed_kmh: float):
    """
    Convert distance (same unit as your coordinates) to MINUTES,
    assuming speed is given in unit/hour (e.g., km/h).
    Supports scalar or numpy array `distance`.
    """
    return (np.asarray(distance, dtype=float) / float(speed_kmh)) * 60.0

def energy_from_distance(distance, energy_per_distance: float):
    """
    Energy consumption for a given distance when your model uses
    'energy per unit distance' (e.g., kWh/km). Supports scalar/array.
    """
    return np.asarray(distance, dtype=float) * float(energy_per_distance)

def energy_from_time(minutes, energy_per_minute: float):
    """
    Energy consumption for a given time when your model uses
    'energy per minute' (e.g., kWh/min). Supports scalar/array.
    """
    return np.asarray(minutes, dtype=float) * float(energy_per_minute)

def can_travel_direct(distance, battery_capacity: float, energy_per_distance: float) -> np.ndarray:
    """
    Check feasibility of a direct leg by distance-energy model.
    Returns boolean array/scalar: energy_from_distance(distance) <= battery_capacity.
    """
    need = energy_from_distance(distance, energy_per_distance)
    return need <= float(battery_capacity)

def clamp_time_windows(tw_start, tw_end, min_width: float = 0.0):
    """
    Ensure time windows are valid: start <= end, and optionally enforce a min width.
    Inputs can be arrays; output is (start, end) arrays.
    """
    ts = np.asarray(tw_start, dtype=float)
    te = np.asarray(tw_end, dtype=float)
    if min_width > 0:
        width = np.maximum(te - ts, min_width)
        center = (ts + te) / 2.0
        ts = center - width / 2.0
        te = center + width / 2.0
    swap = ts > te
    ts2 = ts.copy(); te2 = te.copy()
    ts2[swap] = te[swap]; te2[swap] = ts[swap]
    return ts2, te2
