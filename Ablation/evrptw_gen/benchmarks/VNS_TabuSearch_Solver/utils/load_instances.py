import re
from collections import namedtuple
from .helpers import map_param, build_distance_matrix

class EVRPTWInstance:
    def __init__(self):
        self.depot = None
        self.customers = []
        self.stations = []
        self.vehicle_params = {
            'fuel_cap': None,
            'load_cap': None,
            'consump_rate': None,
            'charge_rate': None,
            'velocity': None
        }
        self.dist_matrix = None

class Route:
    def __init__(self, nodes = []):
        self.nodes = nodes
        self.load = 0
        self.time = 0
        self.fuel = 0

Customer = namedtuple('Customer', ['id', 'type', 'x', 'y', 'demand', 
                                   'ready', 'due', 'service'])
Station = namedtuple('Station', Customer._fields)
Depot = namedtuple('Depot', ['id', 'type', 'x', 'y', 'demand', 'ready', 'due', 'service'])


def load_instance(file_path):
    """Load an EVRPTW instance from a file."""
    instance = EVRPTWInstance()
    param_pattern = re.compile(r"([A-Za-z]+)\s+.*?/(\d+\.?\d*)/")
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("StringID") or line.startswith('S0'):
                continue
            
            # Parser for parameter lines
            if line[:2] in ['Q ','C ','r ','g ','v ']:
                match = param_pattern.search(line)
                if match:
                    param, value = match.groups()
                    map_param(instance, param, float(value))
                continue
                
            # Parser for node lines
            parts = line.split()
            if len(parts) < 8: continue
            
            node_id = parts[0]
            node_type = parts[1]
            x, y = float(parts[2]), float(parts[3])
            demand = float(parts[4])
            ready, due, service = map(float, parts[5:8])

            if node_type == 'd':
                instance.depot = Depot(node_id, node_type, x, y, demand, ready, due, service)
            elif node_type == 'f':
                instance.stations.append(Station(node_id, node_type, x, y, demand, ready, due, service))
            elif node_type == 'c':
                instance.customers.append(Customer(node_id, node_type, x, y, demand, ready, due, service))

    # pre-calculate distance matrix
    build_distance_matrix(instance)
    return instance
