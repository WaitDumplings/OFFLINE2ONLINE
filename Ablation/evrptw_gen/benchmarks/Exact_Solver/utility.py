import numpy as np

from collections import defaultdict
from typing import List, Tuple, Dict, Optional

def _parse_edge(edge: str) -> Optional[Tuple[int, int]]:
    # edge format: "x_i_j"
    if not edge.startswith("x_"):
        return None
    parts = edge.split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def extract_routes(sequence: List[str], end_node_idx: int) -> List[List[str]]:
    """
    Given a list of selected edges encoded as 'x_i_j', extract all complete routes
    starting from depot edges 'x_0_k' and ending when reaching node `end_node_idx`.

    Returns: list of routes, each route is a list of edge strings in visiting order.
    """
    # 1) Pre-parse and build adjacency: i -> list of (edge_str, j)
    adj: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    start_edges: List[str] = []

    parsed: Dict[str, Tuple[int, int]] = {}
    for e in sequence:
        ij = _parse_edge(e)
        if ij is None:
            continue
        i, j = ij
        parsed[e] = (i, j)
        adj[i].append((e, j))
        if i == 0:  # start edge out of depot
            start_edges.append(e)

    # Optional: deterministic traversal (keeps stable outputs across runs)
    # sort by destination index, then edge string
    for i in adj:
        adj[i].sort(key=lambda t: (t[1], t[0]))
    start_edges.sort(key=lambda e: parsed[e][1] if e in parsed else e)

    routes: List[List[str]] = []
    used = set()  # mark used edges in current DFS path

    def dfs(edge: str, curr_route: List[str]):
        i, j = parsed[edge]
        curr_route.append(edge)

        if j == end_node_idx:
            routes.append(curr_route.copy())
            curr_route.pop()
            return

        # follow outgoing edges from node j
        for nxt_edge, nxt_j in adj.get(j, []):
            if nxt_edge in used:
                continue
            used.add(nxt_edge)
            dfs(nxt_edge, curr_route)
            used.remove(nxt_edge)

        curr_route.pop()

    # 2) Launch DFS from each start edge
    for se in start_edges:
        if se in used:
            continue
        used.add(se)
        dfs(se, [])
        used.remove(se)

    return routes


def find_fleet_size(sequence: List[str], end_node_idx: int) -> int:
    return len(extract_routes(sequence, end_node_idx))

def find_and_print_routes(sequence: List[str], end_node_idx: int = 5, print_res: bool = False) -> List[List[str]]:
    routes = extract_routes(sequence, end_node_idx)
    if print_res:
        print(f"Total number of complete routes: {len(routes)}")
        for idx, route in enumerate(routes, 1):
            print(f"Route {idx}: {' -> '.join(route)}")
    return routes

# Function to calculate Euclidean distance between two nodes
def euclidean_distance(pos1, pos2):
    """
    Calculate the Euclidean distance between two nodes.
    
    Args:
        pos1 (list): Position [x, y] of the first node.
        pos2 (list): Position [x, y] of the second node.
    
    Returns:
        float: Euclidean distance between the two nodes.
    """
    return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
