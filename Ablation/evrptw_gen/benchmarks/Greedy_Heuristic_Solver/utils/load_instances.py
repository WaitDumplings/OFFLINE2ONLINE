import re
import ast
from typing import Dict, Any, List, Optional

import numpy as np


# -------------------------
# Robust regex patterns
# -------------------------

# Node line example:
# D0 d 4.384 91.170 0.000 0.000 23.983 0.000
NODE_RE = re.compile(
    r"^(?P<id>\S+)\s+(?P<type>[dfc])\s+"
    r"(?P<x>-?\d+(?:\.\d+)?)\s+(?P<y>-?\d+(?:\.\d+)?)\s+"
    r"(?P<demand>-?\d+(?:\.\d+)?)\s+"
    r"(?P<ready>-?\d+(?:\.\d+)?)\s+(?P<due>-?\d+(?:\.\d+)?)\s+(?P<service>-?\d+(?:\.\d+)?)\s*$"
)

# Slash key/value line example:
# Q Vehicle fuel tank capacity /34.85/
# working_startTime (hour) /8.00/
# cs_time_to_depot (hour)/[0.99,2.75]/
KV_SLASH_RE = re.compile(
    r"^\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*.*?/(?P<val>.+?)/\s*$"
)


def _coerce_value(val_str: str) -> Any:
    """
    Coerce the content inside /.../ into python value:
    - list via ast.literal_eval if looks like [...]
    - int/float if numeric
    - otherwise keep as string
    """
    s = val_str.strip()

    # list
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            return v
        except Exception:
            return s

    # int or float
    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        return float(s)
    except Exception:
        return s


def _make_empty_instance() -> Dict[str, Any]:
    return {
        "depot": None,          # dict
        "stations": [],         # list[dict]
        "customers": [],        # list[dict]
        "vehicle": {
            # canonical keys
            "Q": None, "C": None, "r": None, "g": None, "v": None,
            # semantic aliases (optional, but practical)
            "fuel_cap": None,
            "load_cap": None,
            "consump_rate": None,
            "inv_charge_rate": None,
            "velocity": None,
        },
        "meta": {},
        "cs2depot": None,       # list[float] or None
        "dist_matrix": None,    # np.ndarray
        # "time_matrix": None,  # optional
    }


def build_distance_matrix(instance: Dict[str, Any]) -> None:
    """
    Build Euclidean distance matrix over the canonical node order:
    [depot] + stations + customers
    """
    depot = instance.get("depot")
    if depot is None:
        raise ValueError("Instance missing depot; cannot build distance matrix.")

    nodes: List[Dict[str, Any]] = [depot] + instance.get("stations", []) + instance.get("customers", [])
    coords = np.array([[n["x"], n["y"]] for n in nodes], dtype=np.float64)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    instance["dist_matrix"] = dist

    # Optional: build time matrix if v exists
    # v = instance.get("vehicle", {}).get("v")
    # if v is not None and v != 0:
    #     instance["time_matrix"] = dist / float(v)


def load_instance(file_path: str, *, drop_depot_station_duplicate: bool = True, eps: float = 1e-9) -> Dict[str, Any]:
    """
    Load an EVRP-TW instance from the target.txt-like format.

    Key behaviors:
    - Outputs a dict-based instance (easy to use .get()).
    - Parses:
      (1) node table lines into depot/stations/customers
      (2) vehicle params (Q,C,r,g,v) into instance["vehicle"]
      (3) arbitrary metadata into instance["meta"]
      (4) cs_time_to_depot into instance["cs2depot"]
    - Drops the redundant station at the same location/time window as depot (e.g., S0) if enabled.
    """
    instance = _make_empty_instance()

    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # skip header / separator
            if line.startswith("StringID"):
                continue
            if set(line) == {"-"}:
                continue

            # 1) node line
            m_node = NODE_RE.match(line)
            if m_node:
                d = m_node.groupdict()
                node = {
                    "id": d["id"],
                    "type": d["type"],
                    "x": float(d["x"]),
                    "y": float(d["y"]),
                    "demand": float(d["demand"]),
                    "ready": float(d["ready"]),
                    "due": float(d["due"]),
                    "service": float(d["service"]),
                }

                if node["type"] == "d":
                    instance["depot"] = node
                    continue

                if node["type"] == "f":
                    if drop_depot_station_duplicate:
                        depot = instance.get("depot")
                        if depot is not None:
                            same_xy = (abs(node["x"] - depot["x"]) <= eps) and (abs(node["y"] - depot["y"]) <= eps)
                            same_tw = (abs(node["ready"] - depot["ready"]) <= eps) and (abs(node["due"] - depot["due"]) <= eps)
                            same_srv = abs(node["service"] - depot["service"]) <= eps

                            # Treat depot-co-located station as redundant (e.g., S0)
                            if same_xy and same_tw and same_srv:
                                continue

                    instance["stations"].append(node)
                    continue

                # customer
                instance["customers"].append(node)
                continue

            if line.startswith("instance_id"):
                instance["instance_id"] = int(line.split("/")[1].strip())

            # 2) key/value slash line (params + meta)
            m_kv = KV_SLASH_RE.match(line)
            if m_kv:
                key = m_kv.group("key")
                val = _coerce_value(m_kv.group("val"))

                # vehicle parameters
                if key in {"Q", "C", "r", "g", "v"}:
                    try:
                        instance["vehicle"][key] = float(val)
                    except Exception:
                        instance["vehicle"][key] = None

                    # semantic aliases
                    if key == "Q":
                        instance["vehicle"]["fuel_cap"] = instance["vehicle"][key]
                    elif key == "C":
                        instance["vehicle"]["load_cap"] = instance["vehicle"][key]
                    elif key == "r":
                        instance["vehicle"]["consump_rate"] = instance["vehicle"][key]
                    elif key == "g":
                        instance["vehicle"]["inv_charge_rate"] = instance["vehicle"][key]
                    elif key == "v":
                        instance["vehicle"]["velocity"] = instance["vehicle"][key]

                else:
                    instance["meta"][key] = val

                    # special: cs_time_to_depot -> cs2depot
                    if key == "cs_time_to_depot" and isinstance(val, list):
                        instance["cs2depot"] = [float(x) for x in val]
                continue

            # 3) unparsed lines (optional for debugging)
            # instance["meta"].setdefault("_unparsed_lines", []).append(line)

    # Basic sanity checks
    if instance["depot"] is None:
        raise ValueError(f"Missing depot node in: {file_path}")

    # If you rely on cs2depot, validate alignment (optional but strongly recommended)
    if instance["cs2depot"] is not None:
        # IMPORTANT: confirm whether cs_time_to_depot includes the dropped depot-station or not.
        # If your cs_time_to_depot is computed AFTER dropping S0, then it should equal len(stations).
        # If it was computed INCLUDING S0, then it may equal len(stations)+1.
        if len(instance["cs2depot"]) not in {len(instance["stations"]), len(instance["stations"]) + 1}:
            raise ValueError(
                f"cs2depot length mismatch: len(cs2depot)={len(instance['cs2depot'])}, "
                f"len(stations)={len(instance['stations'])}. "
                "Please confirm whether cs_time_to_depot includes the depot-co-located station."
            )
    # Build distance matrix
    build_distance_matrix(instance)

    return instance


# -------------------------
# Route helper (fix mutable default)
# -------------------------
class Route:
    def __init__(self, nodes=None):
        self.nodes = [] if nodes is None else list(nodes)
        self.load = 0.0
        self.time = 0.0
        self.fuel = 0.0
