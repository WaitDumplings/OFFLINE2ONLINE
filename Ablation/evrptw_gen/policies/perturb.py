import numpy as np
from typing import Any, Dict, Optional

class Perturbation:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

        self.decimals_cfg = {
            "speed": 2,
            "battery_capacity": 2,
            "consumption_per_distance": 4,
            "loading_capacity": 2,
            "charging_speed": 2,
            "charging_efficiency": 4,
            "area_size": 2,
        }

    def _mul(self, x, lo: float, hi: float):
        return x * self.rng.uniform(lo, hi)

    def _add(self, x, lo: float, hi: float, is_int: bool = False):
        y = x + self.rng.uniform(lo, hi)
        if is_int:
            y = int(np.rint(y))
        return y

    def _clamp(self, x, lo=None, hi=None):
        if lo is not None:
            x = max(x, lo)
        if hi is not None:
            x = min(x, hi)
        return x

    def _round_key(self, key: str, val: Any):
        """Round value for a given key based on predefined decimal configuration."""
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (float, np.floating)):
            d = self.decimals_cfg.get(key, None)
            return round(float(val), d) if d is not None else float(val)
        return val

    def _scale_area_size(self, area_size, lo=0.9, hi=1.1):
        s = self.rng.uniform(lo, hi)
        a = np.array(area_size, dtype=float)
        xmin, xmax = a[0]
        ymin, ymax = a[1]

        xspan = (xmax - xmin) * s
        yspan = (ymax - ymin) * s

        new_area = [
            [float(xmin), float(xmin + xspan)],
            [float(ymin), float(ymin + yspan)],
        ]

        # area_size
        d = self.decimals_cfg.get("area_size", None)
        if d is not None:
            new_area = [[round(v, d) for v in row] for row in new_area]
        return new_area

    def perturb(self, env: Dict[str, Any], perturb_keys: Dict[str, Any], rng = None) -> Dict[str, Any]:
        if rng is not None:
            self.rng = rng

        out: Dict[str, Any] = {}

        mul_cfg = {
            "speed": (0.7, 1.1),
            "battery_capacity": (0.8, 1.0),
            "consumption_per_distance": (0.8, 1.0),
            "loading_capacity": (0.8, 1.0),
            "charging_speed": (0.8, 1.0),
            "charging_efficiency": (0.7, 1.0),
        }

        add_cfg = {
            "num_cluster": (-3, 3, True),
            "working_startTime": (-60, 60, True),
            "working_endTime": (0, 120, True),
        }

        for key, enabled in perturb_keys.items():
            if not enabled:
                continue

            if key not in env:
                print(f"[WARNING] env has no key: {key}. Skip.")
                continue

            if key == "area_size":
                out[key] = self._scale_area_size(env[key], 0.9, 1.1)
                continue

            if key in mul_cfg:
                lo, hi = mul_cfg[key]
                val = self._mul(env[key], lo, hi)

                if key in ("speed", "battery_capacity", "loading_capacity", "charging_speed"):
                    val = float(max(val, 1e-6))
                if key == "consumption_per_distance":
                    val = float(max(val, 1e-9))

                out[key] = self._round_key(key, val)
                continue

            if key in add_cfg:
                lo, hi, is_int = add_cfg[key]
                val = self._add(env[key], lo, hi, is_int=is_int)
                if key == "num_cluster":
                    val = self._clamp(val, lo=1)
                out[key] = val  # int already
                continue

            print(f"[WARNING] No perturbation method defined for key: {key}. Skipping.")

        # working time consistency
        if ("working_startTime" in out or "working_endTime" in out) and \
           ("working_startTime" in env and "working_endTime" in env):
            st = out.get("working_startTime", env["working_startTime"])
            et = out.get("working_endTime", env["working_endTime"])
            if et <= st:
                out["working_endTime"] = st + 1

            st = min(st, 12 * 60) # no later than 12 pm for work
            et = min(et, env['instance_endTime'])
        return out
