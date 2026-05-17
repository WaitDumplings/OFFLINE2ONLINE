import os
import numpy as np
import time
import matplotlib.pyplot as plt


def _to_2d_points(arr):
    """Coerce array-like into (N,2) float ndarray."""
    a = np.asarray(arr, dtype=float)
    a = np.atleast_2d(a)
    if a.shape[1] != 2:
        if a.ndim == 2 and a.shape[0] == 2:  # e.g., [[x,...],[y,...]] -> transpose
            a = a.T
        else:
            a = a[:, :2]
    return a


def _as_float(x):
    """Robustly convert x (scalar/list/ndarray) to a Python float (take the first element if needed)."""
    arr = np.asarray(x).reshape(-1)
    return float(arr[0])


def _fmt(x, width, decimals=8):
    """Format a number to a fixed-width string with configurable decimal places."""
    return f"{format(_as_float(x), f'.{decimals}f'):>{width}}"


def _xy(pos):
    """Return (x, y) as floats from any pos shape: (2,), (1,2), [[x,y]], etc."""
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Position must have at least 2 numbers, got: {pos}")
    return float(arr[0]), float(arr[1])


def print_type(instances):
    for i in range(len(instances)):
        instance_type = instances[i]['env']['instance_type']
        time_window_type = instances[i]['env']['time_window_type']
        print(f'Instance {i}: {instance_type}, {time_window_type}')


def save_instances(instances, save_path, template='solomon'):
    """Save EVRP instances in the Solomon dataset format."""

    if template == 'solomon':
        save_path_solomon = os.path.join(save_path, "solomon")
        os.makedirs(save_path_solomon, exist_ok=True)
        timestamp = int(time.time())

        # unified text precision for all node table fields
        txt_decimals = 8
        coord_width = 14
        field_width = 14

        for i in range(len(instances)):
            inst = instances[i]

            time_window_type = inst['env']['time_window_type']
            instance_type = inst['env']['instance_type']

            # --------------------------------------------------
            # 1) build stable file name first
            # --------------------------------------------------
            save_file_name = f"solomon_dataset_{i}_{instance_type}_{time_window_type}_{timestamp}.txt"
            instance_key = save_file_name
            instance_id = os.path.splitext(save_file_name)[0]

            # --------------------------------------------------
            # 2) write back into instance so pickle/txt can align
            # --------------------------------------------------
            inst["file"] = instance_key
            inst["instance_id"] = instance_id

            # optional: also keep old id field synchronized
            inst["id"] = instance_id

            instance_end_time = inst['env'].get('instance_endTime', 1440.0) / 60.0

            depot_pos = inst['depot']
            charging_station_pos = np.concatenate((depot_pos, inst['charging_stations']), axis=0)
            customers = inst['customers']
            demands = inst["demands"].reshape(-1, 1)
            time_windows = inst["tw"] / 60.0
            service_time = inst["service_time"].reshape(-1, 1) / 60.0

            Q = inst['env']['battery_capacity']
            C = inst['env']['loading_capacity']
            r = inst['env']['consumption_per_distance']
            g = 1 / inst['env']['charging_speed']
            gv = inst['env']['charging_speed']
            v = inst['env']['speed']

            env = inst.get("env", {})
            num_cluster = env.get("num_cluster", env.get("num_clusters", None))

            ws = env.get("working_startTime", env.get("working_start_time", None))
            we = env.get("working_endTime", env.get("working_end_time", None))
            ie = env.get("instance_endTime", None)
            working_start_h = None if ws is None else float(ws) / 60.0
            working_end_h = None if we is None else float(we) / 60.0
            instance_end_h = None if ie is None else float(ie) / 60.0

            service_time_type = env.get("service_time_type", None)
            demand_type = env.get("demand_type", None)
            x_range = env.get("area_size", None)[0]
            y_range = env.get("area_size", None)[1]

            save_name = os.path.join(save_path_solomon, save_file_name)

            with open(save_name, "w") as f:
                # Header
                f.write(
                    "StringID   Type"
                    f"{'x':>{coord_width}}"
                    f"{'y':>{coord_width}}"
                    f"{'demand':>{field_width}}"
                    f"{'ReadyTime':>{field_width}}"
                    f"{'DueDate':>{field_width}}"
                    f"{'ServiceTime':>{field_width}}\n"
                )

                # Depot
                dx, dy = _xy(depot_pos)
                line = (
                    "D0         d"
                    f"{_fmt(dx, coord_width, txt_decimals)}"
                    f"{_fmt(dy, coord_width, txt_decimals)}"
                    f"{_fmt(0.0, field_width, txt_decimals)}"
                    f"{_fmt(0.0, field_width, txt_decimals)}"
                    f"{_fmt(instance_end_time, field_width, txt_decimals)}"
                    f"{_fmt(0.0, field_width, txt_decimals)}\n"
                )
                f.write(line)

                # Charging stations
                for j, pos in enumerate(charging_station_pos, start=1):
                    sx, sy = _xy(pos)
                    line = (
                        f"S{j}         f"
                        f"{_fmt(sx, coord_width, txt_decimals)}"
                        f"{_fmt(sy, coord_width, txt_decimals)}"
                        f"{_fmt(0.0, field_width, txt_decimals)}"
                        f"{_fmt(0.0, field_width, txt_decimals)}"
                        f"{_fmt(instance_end_time, field_width, txt_decimals)}"
                        f"{_fmt(0.0, field_width, txt_decimals)}\n"
                    )
                    f.write(line)

                # Customers
                customer_info = np.concatenate((customers, demands, time_windows, service_time), axis=1)
                for k, c in enumerate(customer_info):
                    arr = np.asarray(c, dtype=float).reshape(-1)
                    if arr.size < 6:
                        raise ValueError(
                            f"Customer row must have 6 numbers (x,y,demand,ready,due,service), got: {c}"
                        )

                    x, y, demand, ready, due, service = arr[:6]

                    line = (
                        f"C{k}         c"
                        f"{_fmt(x, coord_width, txt_decimals)}"
                        f"{_fmt(y, coord_width, txt_decimals)}"
                        f"{_fmt(demand, field_width, txt_decimals)}"
                        f"{_fmt(ready, field_width, txt_decimals)}"
                        f"{_fmt(due, field_width, txt_decimals)}"
                        f"{_fmt(service, field_width, txt_decimals)}\n"
                    )
                    f.write(line)

                # Environment parameters
                f.write("\n")
                f.write(f"Q Vehicle fuel tank capacity /{format(_as_float(Q), '.8f')}/\n")
                f.write(f"C Vehicle load capacity /{format(_as_float(C), '.8f')}/\n")
                f.write(f"r fuel consumption rate /{format(_as_float(r), '.8f')}/\n")
                f.write(f"g inverse refueling rate /{format(_as_float(g), '.8f')}/\n")
                f.write(f"v average Velocity /{format(_as_float(v), '.8f')}/\n")
                f.write(f"gv charging speed /{format(_as_float(gv), '.8f')}/\n")

                # Extra metadata
                f.write("\n" + "-" * 20 + "\n")
                f.write(f"file /{instance_key}/\n")
                f.write(f"instance_id /{instance_id}/\n")
                f.write(f"number of clusters /{num_cluster}/\n")
                f.write(
                    f"working_startTime (hour) /{format(working_start_h, '.8f') if working_start_h is not None else 'None'}/\n"
                )
                f.write(
                    f"working_endTime (hour) /{format(working_end_h, '.8f') if working_end_h is not None else 'None'}/\n"
                )
                f.write(
                    f"instance_endTime (hour) /{format(instance_end_h, '.8f') if instance_end_h is not None else 'None'}/\n"
                )
                f.write(f"service_time_type /{service_time_type}/\n")
                f.write(f"demand_type /{demand_type}/\n")
                f.write(f"instance_type /{instance_type}/\n")
                f.write(f"instance x range /{x_range}/\n")
                f.write(f"instance y range /{y_range}/\n")

                cs_ttd = env.get("cs_time_to_depot", None)
                cs_ttd = np.asarray(cs_ttd, dtype=float).reshape(-1)
                cs_ttd_str = ",".join(f"{x:.8f}" for x in cs_ttd.tolist())
                f.write(f"cs_time_to_depot (hour)/[{cs_ttd_str}]/\n")

            print(f"✅ Saved: {save_name}")

    elif template == 'pickle':
        save_path_pickle = os.path.join(save_path, "pickle")

        def check_extension(filename):
            if os.path.splitext(filename)[1] != ".pkl":
                return filename + ".pkl"
            return filename

        import pickle
        cus_scale = instances[0]['env']['num_customers']
        rs_scale = instances[0]['env']['num_charging_stations']
        filename = "evrptw_{}C_{}R.pkl".format(cus_scale, rs_scale)
        filedir = os.path.join(save_path_pickle, filename)

        if not os.path.isdir(save_path_pickle):
            os.makedirs(save_path_pickle)

        with open(check_extension(filedir), 'wb') as f:
            pickle.dump(instances, f, pickle.HIGHEST_PROTOCOL)


def fast_check(cs_pos, depot_pos, battery_capacity, consume_rate):
    """Quickly check which charging stations can directly reach the depot."""
    cs_pos_arr = _to_2d_points(cs_pos)
    depot_pos_arr = _to_2d_points(depot_pos).reshape(1, 2)
    dists = np.linalg.norm(cs_pos_arr - depot_pos_arr, axis=1)
    reachable = dists * consume_rate <= battery_capacity
    return reachable.all()


def plot_instance(instances, save_path):
    save_path_plots = os.path.join(save_path, "plots")
    os.makedirs(save_path_plots, exist_ok=True)
    timestamp = int(time.time())

    for i in range(len(instances)):
        inst = instances[i]
        time_window_type = inst['env']['time_window_type']
        instance_type = inst['env']['instance_type']
        save_file_name = f'instance_{i}_{instance_type}_{time_window_type}_{timestamp}.png'

        depot_pos = _to_2d_points(inst['depot'])
        cs_pos = _to_2d_points(inst.get('charging_stations', np.empty((0, 2))))
        cust_raw = np.asarray(inst.get('customers', np.empty((0, 2))), dtype=float)
        customers_pos = _to_2d_points(cust_raw)[:, :2] if cust_raw.size else np.empty((0, 2))

        x_range = inst['env']['area_size'][0]
        y_range = inst['env']['area_size'][1]

        fig, ax = plt.subplots(figsize=(6, 6))

        if customers_pos.size:
            ax.scatter(customers_pos[:, 0], customers_pos[:, 1],
                       s=15, marker='o', label='Customers')
        if cs_pos.size:
            ax.scatter(cs_pos[:, 0], cs_pos[:, 1],
                       s=60, marker='^', label='Charging Stations')
        if depot_pos.size:
            ax.scatter(depot_pos[:, 0], depot_pos[:, 1],
                       s=120, marker='*', label='Depot')

        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'EVRP Instance {i}')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linewidth=0.5, alpha=0.4)
        ax.legend(loc='best')

        out_path = os.path.join(save_path_plots, save_file_name)
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)