import re

# File path to the dataset
class Node:
    """
    Represents a node in the dataset, including attributes such as position, demand, time window, and service time.
    The input format should follow the structure:

    StringID   Type   x       y      demand  ReadyTime  DueDate   ServiceTime 
    D0         d      40.0    50.0   0.0     0.0        1236.0    0.0        
    S0         f      40.0    50.0   0.0     0.0        1236.0    0.0        
    S5         f      31.0    84.0   0.0     0.0        1236.0    0.0               
    C30        c      20.0    55.0   10.0    355.0      407.0     90.0       

    Node types:
    - d: depot
    - f: recharging station (RS)
    - c: customer

    Vehicle-related constants:
    - Q: Vehicle fuel tank capacity (77.75)
    - C: Vehicle load capacity (200.0)
    - r: Fuel consumption rate (1.0)
    - g: Inverse refueling rate (3.47)
    - v: Average velocity (1.0)
    """

    def __init__(self, StringId=-1, Type=None, Pos_x=-1, Pos_y=-1, Demand=0, Window_S=0, Window_E=1e5, ServiceTime=0):
        """
        Initialize a Node instance with the provided attributes.

        :param StringId: Node identifier (e.g., 'D0' for depot, 'C30' for customer).
        :param Type: Node type ('d' for depot, 'f' for recharging station, 'c' for customer).
        :param Pos_x: X-coordinate of the node position.
        :param Pos_y: Y-coordinate of the node position.
        :param Demand: Node demand (e.g., for customers).
        :param Window_S: Time window start (ready time).
        :param Window_E: Time window end (due time).
        :param ServiceTime: Time required to service the node.
        """
        self.StringId = StringId  # Unique identifier for the node (e.g., 'D0', 'S0', 'C30')
        self.Type = Type  # Node type ('d', 'f', 'c')
        self.Position = [Pos_x, Pos_y]  # Node's (x, y) coordinates
        self.Demand = Demand  # Demand for service at the node (if applicable)
        self.Window = [Window_S, Window_E]  # Time window [start, end]
        self.ServiceTime = ServiceTime  # Time needed to service the node

# File parsing function
def parse_file(file_path):
    """
    Parses the dataset file and extracts depot, customer, and refueling station nodes,
    as well as vehicle-related parameters.
    
    Args:
        file_path (str): Path to the dataset file.
    
    Returns:
        Depot_nodes (list): List of depot nodes.
        Customer_nodes (list): List of customer nodes.
        RS_nodes (list): List of refueling station nodes.
        parameters (dict): Dictionary of vehicle-related parameters.
    """
    Depot_nodes = []  # Store depot node data
    Customer_nodes = []  # Store customer node data
    RS_nodes = []  # Store refueling station node data
    
    # Q(Vehicle fuel tank capacity), 
    # C(Vehicle load capacity), 
    # r(fuel consumption rate), 
    # g(inverse refueling rate), 
    # v(average velocity)
    parameters = {}  
    
    # Open the file for reading
    with open(file_path, 'r') as file:
        for line in file:
            # Use regular expressions to parse lines for nodes or parameters
            node_match = re.match(r'(\w+)\s+([dcf])\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            parameter_match = re.match(r'([QCrgv])\s.*\/([\d.]+)\/', line)

            if node_match:
                # Extract node data
                node_data = Node()
                node_data.StringId = node_match.group(1)
                node_data.Type = node_match.group(2)
                node_data.Position = [float(node_match.group(3)), float(node_match.group(4))]
                node_data.Demand = float(node_match.group(5))
                node_data.Window = [float(node_match.group(6)), float(node_match.group(7))]
                node_data.ServiceTime = float(node_match.group(8))

                # Categorize nodes based on the ID prefix
                if node_data.StringId.startswith("D"):
                    Depot_nodes.append(node_data)
                elif node_data.StringId.startswith("S"):
                    RS_nodes.append(node_data)
                elif node_data.StringId.startswith("C"):
                    Customer_nodes.append(node_data)
                else:
                    raise ValueError("Unexpected Data Type: {}".format(node_data.StringId[0]))
                
            elif parameter_match:
                # Extract vehicle-related parameters
                parameter_data = {
                    parameter_match.group(1): float(parameter_match.group(2))
                }
                parameters.update(parameter_data)
                
    # Sometimes the RS[0] is the depot, which results in duplicate computation
    if RS_nodes[0].Position == Depot_nodes[0].Position:
        RS_nodes.pop(0)

    # Return the parsed data
    return Depot_nodes, Customer_nodes, RS_nodes, parameters