import numpy as np
from utility import euclidean_distance

class Graph_EVRP_TW():
    """
    Represents the graph structure for the Electric Vehicle Routing Problem with Time Windows (EVRP-TW).
    
    Attributes:
        Depot_nodes (list): List of depot nodes.
        Customer_nodes (list): List of customer nodes.
        RS_nodes (list): List of refueling station nodes.
        parameters (dict): Dictionary of vehicle parameters (fuel capacity, load, etc.).
        RS_dummy_count (int): Number of dummy refueling stations to add to the graph.
    """
    def __init__(self, Depot_nodes, Customer_nodes, RS_nodes, parameters, RS_dummy_count=3):
        self.Depot_nodes = Depot_nodes
        self.Customer_nodes = Customer_nodes
        self.RS_nodes = RS_nodes
        self.parameters = parameters

        # The entire node set: [Depot, Customer, RS, RS_dummy, Depot_Dummy]
        self.Nodes = self.Depot_nodes + self.Customer_nodes + self.RS_nodes * RS_dummy_count + self.Depot_nodes
        self.Depot_Idx = [0, len(self.Nodes) - 1]  # Indices of the depots
        self.Customer_Idx = list(range(1, 1 + len(self.Customer_nodes)))  # Indices of customers
        self.RS_Idx = list(range(1 + len(self.Customer_nodes), len(self.Nodes) - 1))  # Indices of refueling stations

        self.distance_matrix = self.generate_distance_matrix()  # Generate distance matrix
        self.CustomerTW = [node.Window for node in self.Nodes]  # Time windows of customers
        self.CustomerServiceTime = [node.ServiceTime for node in self.Nodes]  # Service times of customers
        self.CustomerDemand = [node.Demand for node in self.Nodes]  # Customer demands

        # Vehicle parameters from the 'parameters' dictionary
        self.Q = parameters['Q']    # Vehicle fuel tank capacity
        self.C = parameters['C']    # Vehicle load capacity
        self.r = parameters['r']    # Fuel consumption rate
        self.g = parameters['g']    # Inverse refueling rate
        self.v = parameters['v']    # Average vehicle velocity

        # Travel time matrix calculated based on distance and velocity
        self.travel_time_matrix = self.distance_matrix / self.v

    def print_info(self):
        """
        For Debug.
        Print the indices of depots, customers, and refueling stations.
        """
        print(self.Depot_Idx)
        print(self.Customer_Idx)
        print(self.RS_Idx)

        # Uncomment to print full node information and distance matrix
        # print()
        # print(self.Nodes)
        # for i in range(len(self.Nodes)):
        #     print(self.distance_matrix[i])

    def generate_distance_matrix(self):
        """
        Generate the distance matrix for all nodes in the graph.
        
        Returns:
            np.array: Distance matrix where each entry [i, j] represents the distance between node i and node j.
        """
        distance_matrix = np.zeros((len(self.Nodes), len(self.Nodes)))
        for i in range(len(self.Nodes)):
            for j in range(i, len(self.Nodes)):
                distance_matrix[i][j] = euclidean_distance(self.Nodes[i].Position, self.Nodes[j].Position)

        # Mirror the matrix to make it symmetric
        distance_matrix += distance_matrix.transpose()
        distance_matrix = distance_matrix * 1.0
        return np.array(distance_matrix)