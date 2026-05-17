
## Why two formats?

EVRP-TW-D-B provides two complementary instance formats:

- **Pickle (`.pkl`)** is **training-based-model friendly**: it stores a fully
  self-contained Python representation (instance + environment metadata +
  generation policies), enabling fast loading, batch-parallel data pipelines,
  and reproducible diagnostics for learning-based methods.

- **Solomon-style text (`.txt`)** is **solver-friendly** for non-learning
  pipelines: it follows a widely used VRPTW-style layout, making it convenient
  to integrate with classical exact solvers, heuristics/metaheuristics, and
  existing tooling that expects text-based instance files.

## Solomon-style instance format (text)

EVRP-TW-D-B optionally exports instances in a **Solomon-style text format** to maintain compatibility with classic VRPTW tooling and existing solvers. While the layout follows the Solomon convention, **all quantities in EVRP-TW-D-B have explicit physical semantics and units**, consistent with electric vehicle routing.

An instance file consists of three parts:
1) a **node table**,
2) a **vehicle and physical-parameter block**, and
3) an optional **instance metadata block**.

---

### 1) Node table

The node table specifies depots, charging stations, and customers.

**Header fields:**

| Field | Description | Unit |
|---|---|---|
| `StringID` | Node identifier (e.g., `D0`, `S0`, `C0`) | – |
| `Type` | Node type: `d` (depot), `f` (charging station), `c` (customer) | – |
| `x`, `y` | Node coordinates | problem units |
| `demand` | Customer demand (0 for depot and stations) | problem units |
| `ReadyTime` | Time-window start | hour |
| `DueDate` | Time-window end | hour |
| `ServiceTime` | Service duration at the node | hour |

**Example:**

```text
StringID   Type   x       y       demand   ReadyTime  DueDate   ServiceTime
D0         d      9.472   4.505   0.000    0.000     23.983    0.000
S0         f      9.472   4.505   0.000    0.000     23.983    0.000
S1         f      42.910  8.430   0.000    0.000     23.983    0.000
C0         c      72.972  31.713  0.072    19.326    19.799    0.150
C1         c      31.810  18.960  0.059    16.353    16.940    0.117
...
```

### 2） Vehicle & physical parameters
```text
Q Vehicle fuel tank capacity /118.2500/
C Vehicle load capacity /1.0500/
r fuel consumption rate /0.4897/
g inverse refueling rate /0.0083/
v average Velocity /39.7400/
gv charging speed /120.4200/
```

| Symbol | Meaning in EVRP-TW-D-B       | Unit (recommended) | Notes                                                                    |
| ------ | ---------------------------- | ------------------ | ------------------------------------------------------------------------ |
| Q      | Vehicle **battery capacity** | kWh                | Field name follows Solomon-style *fuel tank capacity* for compatibility. |
| C      | Vehicle load capacity        | capacity units     | Customers consume capacity via `demand`; depot/CS have zero demand.      |
| r      | Energy consumption rate      | kWh per distance   | Energy for traveling distance d: E = r · d.                              |
| g      | Inverse charging rate        | hour per kWh       | Charging time for energy ΔE: t = g · ΔE.                                 |
| v      | Average travel speed         | distance per hour  | Travel time for distance d: t = d / v.                                   |
| gv     | Charging power / speed       | kW (kWh/hour)      | Equivalent to 1/g; charging time t = ΔE / gv.                            |

***Note***: Although the label *fuel tank capacity* is retained for compatibility
with Solomon-style parsers, `Q` semantically represents **electric battery
capacity** in EVRP-TW-D-B.

### 3) Instance metadata
An optional metadata block may be appended to facilitate reproducibility, analysis, and regime labeling. These fields are not required by the solver but are consumed by the benchmark tooling.

```text
instance_id /0/
instance_type /RC/
working_startTime (hour) /8.00/
working_endTime (hour) /20.70/
...
```

### Remarks
- All time-related fields (`ReadyTime`, `DueDate`, `ServiceTime`, and metadata
  times) are measured in **hours**.
- While the text format mirrors the Solomon layout, EVRP-TW-D-B enforces
  **unit-consistent physical semantics** internally.
- Users are encouraged to rely on the provided **parsers** and **evaluation
  utilities** to avoid unit mismatches or inconsistent interpretations.

## Pickle instance format (`.pkl`)

For efficient training and evaluation pipelines, EVRP-TW-D-B can also export
instances in a **pickle** format.

### High-level structure

A pickle file stores a **Python list** of instances.  
For example, generating 1000 instances produces a list of length 1000:

- `data[i]` is the *i-th* instance (a Python `dict`).

### Per-instance schema

Each instance is a dictionary with the following top-level keys:

- `env`: environment metadata and generation context (including the policy choices used)
- `depot`: depot node information
- `customers`: customer node information
- `charging_stations`: charging-station node information
- `demands`: customer demands
- `tw`: time-window data (in hours)
- `service_time`: service times (in hours)
- `id`: instance identifier

### Notes on `env`

The `env` field contains the **complete generation context** for the instance,
including (but not limited to):
- scale information (e.g., number of customers / charging stations),
- physical parameters (e.g., speed, battery capacity, charging speed/efficiency),
- time horizon and working hours,
- and the **specific policies/configurations** used to generate the instance
  (e.g., spatial regime, time-window policy, demand policy, service-time policy,
  cluster assignment policies).

This design ensures the pickle format is **self-contained**: it carries both
the instance data and the metadata needed for reproducible analysis and
diagnostics.