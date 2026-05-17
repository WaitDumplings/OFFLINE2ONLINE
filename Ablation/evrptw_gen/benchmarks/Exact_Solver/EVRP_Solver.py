from utility import find_and_print_routes, find_fleet_size
from gurobipy import *
from gurobipy import Model, GRB, LinExpr, quicksum
import time

class EVRP_TW_Gurobi_Solver:
    def __init__(self, Graph, time_limit = 300, l_0=1e10):
        # Initial Solver
        self.model = Model()

        # Node index information
        self.V_sequence = Graph.Customer_Idx  # Customer node indices
        self.F_Prime_sequence = Graph.RS_Idx  # Recharging station node indices
        self.D = Graph.Depot_Idx  # Depot node indices (start and end)

        # Depot information
        self.Depot_start = [0]  # Start depot index
        self.Depot_end = [len(Graph.Nodes) - 1]  # End depot index

        # Sequences for node sets
        self.V_Prime_N_plus_1_sequence = self.V_sequence + self.F_Prime_sequence + [self.D[-1]]  # All nodes + end depot
        self.V_Prime_0_sequence = [self.D[0]] + self.V_sequence + self.F_Prime_sequence  # Start depot + all nodes
        self.V_Prime_sequence = self.V_sequence + self.F_Prime_sequence  # All customer and RS nodes

        # Graph-related attributes (parameters)
        self.C = Graph.C  # Vehicle load capacity
        self.Q = Graph.Q  # Vehicle battery capacity
        self.h = Graph.r  # Fuel consumption rate (per distance)
        self.g = Graph.g  # Charging speed
        self.l_0 = l_0  # Large constant for subtour elimination

        # Distance and time data
        self.distance_matrix = Graph.distance_matrix  # Distance matrix between nodes
        self.model_size = len(Graph.Nodes)  # Total number of nodes
        self.CustomerTimeWindow = Graph.CustomerTW  # Time windows for customer nodes
        self.CustomerService = Graph.CustomerServiceTime  # Service times for customers
        self.CustomerDemand = Graph.CustomerDemand  # Customer demands
        self.Travel_Time = Graph.travel_time_matrix  # Travel time between nodes

        # Initialize the optimization model
        self.TimeLimit = time_limit
        self._initialize_model()

    # Add Decision Variables, objective functions and constraints
    def _initialize_model(self):
        """Initializes the CPLEX model by setting decision variables, objective function, and constraints."""
        self.model.setParam('TimeLimit', self.TimeLimit)  # Maximal computing time = 300s
        self._set_decision_variable()  # Define decision variables
        self._set_objective_function()  # Define the objective function
        self._set_constraints()  # Define constraints
    
    def _set_decision_variable(self):
        # We combine set_dv & set_objective function in one.
        self.dv_dict = {}
        objective_fun = LinExpr(0)
        # x_i_j -> self.dv_dict[]
        for i in range(self.model_size):
            for j in range(self.model_size):
                if i != j:
                    var_name = f"x_{i}_{j}"
                    self.dv_dict[var_name] = self.model.addVar(lb = 0, ub = 1, vtype=GRB.BINARY, name=var_name)
                    # Add coef and objective variable
                    objective_fun.addTerms(self.distance_matrix[i][j], self.dv_dict[var_name])
            
            # tao_i -> self.dv_dict[]
            self.dv_dict[f"tao_{i}"] = self.model.addVar(lb = self.CustomerTimeWindow[i][0], ub = self.CustomerTimeWindow[i][1], vtype=GRB.CONTINUOUS, name=f"tao_{i}")
            objective_fun.addTerms(0, self.dv_dict[f"tao_{i}"])
            # u_i -> self.dv_dict[]
            self.dv_dict[f"u_{i}"]   = self.model.addVar(lb = 0, ub = self.C, vtype=GRB.CONTINUOUS, name=f"u_{i}")
            objective_fun.addTerms(0, self.dv_dict[f"u_{i}"])
            # y_i -> self.dv_dict[]
            Initial_Battery = self.Q if i == 0 else 0  # Initial battery for the depot is full
            self.dv_dict[f"y_{i}"]   = self.model.addVar(lb = Initial_Battery, ub = self.Q, vtype=GRB.CONTINUOUS, name=f"y_{i}")
            objective_fun.addTerms(0, self.dv_dict[f"y_{i}"])
        self.model.setObjective(objective_fun, GRB.MINIMIZE)

    def _set_objective_function(self):
        """Define the objective function (minimizing the total travel distance)."""
        pass
    
    def _set_constraints(self):
        """Define the constraints for the EVRP-TW problem."""
        # Constraints for customers (each customer must be visited exactly once)
        for i in self.V_sequence:
            Exp1 = LinExpr(0)
            for j in self.V_Prime_N_plus_1_sequence:
                if i != j:
                    Exp1.addTerms(1, self.dv_dict[f"x_{i}_{j}"])
            self.model.addConstr(Exp1 == 1, name = f"C_visted_once_{i}")

        # Constraints for recharging stations (at most one visit per RS in each route)
        for i in self.F_Prime_sequence:
            Exp1 = LinExpr(0)
            for j in self.V_Prime_N_plus_1_sequence:
                if i != j:
                    Exp1.addTerms(1, self.dv_dict[f"x_{i}_{j}"])
            self.model.addConstr(Exp1 <= 1, name = f"RS_visited_once_{i}")

        # Route consistency constraints
        for j in self.V_Prime_sequence:
            Exp1 = LinExpr(0)
            for i_start in self.V_Prime_N_plus_1_sequence:
                if i_start != j:
                    Exp1.addTerms(1, self.dv_dict[f"x_{j}_{i_start}"])
            for i_end in self.V_Prime_0_sequence:
                if i_end != j:
                    Exp1.addTerms(-1, self.dv_dict[f"x_{i_end}_{j}"])
            
            self.model.addConstr(Exp1 == 0, name = f"Route_Consist_{j}")

        # Subtour elimination: travel time and battery constraints
        # Travel Time Constraint(Customer)
        # tao_i - tao_j + (t_ij + s_i + l_0)x_ij <= l_0
        for i in self.Depot_start + self.V_sequence:
            for j in self.V_Prime_N_plus_1_sequence:
                Exp1 = LinExpr(0)
                if i != j:
                    Exp1.addTerms(1, self.dv_dict[f"tao_{i}"])
                    Exp1.addTerms(-1, self.dv_dict[f"tao_{j}"])
                    Exp1.addTerms(self.Travel_Time[i][j] + self.CustomerService[i] + self.l_0, self.dv_dict[f"x_{i}_{j}"])
                    self.model.addConstr(Exp1 <= self.l_0, name=f"Travel_Time_Constraint_{i}_{j}")

        # Travel Time Constraint(RS)
        # tao_i - tao_j + (l_0 + g*Q)x_ij -g*y_i <= l_0
        for i in self.F_Prime_sequence:
            for j in self.V_Prime_N_plus_1_sequence:
                Exp1 = LinExpr(0)
                if i != j:
                    Exp1.addTerms(1, self.dv_dict[f"tao_{i}"])
                    Exp1.addTerms(-1, self.dv_dict[f"tao_{j}"])
                    Exp1.addTerms(self.l_0 + self.g * self.Q, self.dv_dict[f"x_{i}_{j}"])
                    Exp1.addTerms(-self.g, self.dv_dict[f"y_{i}"])
                    self.model.addConstr(Exp1 <= self.l_0, name=f"Travel_Time_Constraint_{i}_{j}")

        # Load Capacity Constraint
        # u_j - u_i +(C + q_i)x_ij <= C
        for i in self.V_Prime_0_sequence:
            for j in self.V_Prime_N_plus_1_sequence:
                Exp1 = LinExpr(0)
                if i != j:
                    Exp1.addTerms(1, self.dv_dict[f"u_{j}"])
                    Exp1.addTerms(-1, self.dv_dict[f"u_{i}"])
                    Exp1.addTerms(self.C + self.CustomerDemand[i], self.dv_dict[f"x_{i}_{j}"])                    
                    self.model.addConstr(Exp1 <= self.C, name=f"Load_Capacity_{i}_{j}")

        # Battery Constraints(Customers)
        # y_j - y_i + (h*d_ij +Q)x_ij <= Q
        for i in self.V_sequence:
            for j in self.V_Prime_N_plus_1_sequence:
                Exp1 = LinExpr(0)
                if i != j:
                    Exp1.addTerms(1, self.dv_dict[f"y_{j}"])
                    Exp1.addTerms(-1, self.dv_dict[f"y_{i}"])
                    Exp1.addTerms(self.h * self.distance_matrix[i][j] + self.Q, self.dv_dict[f"x_{i}_{j}"])                    
                    self.model.addConstr(Exp1 <= self.Q, name=f"Battery_Capacity_Cus_{i}_{j}")

        # Battery Constraints(RS)
        # y_j + h*d_ij*x_ij <= Q
        for i in self.Depot_start + self.F_Prime_sequence:
            for j in self.V_Prime_N_plus_1_sequence:
                Exp1 = LinExpr(0)
                if i != j:
                    Exp1.addTerms(1, self.dv_dict[f"y_{j}"])
                    Exp1.addTerms(self.h * self.distance_matrix[i][j], self.dv_dict[f"x_{i}_{j}"])                    
                    self.model.addConstr(Exp1 <= self.Q, name=f"Battery_Capacity_RS_{i}_{j}")

    def solver(self):
        self.model.optimize()
        self.st = self.model.Status

    def is_feasible(self):
        return True if self.model.Status != GRB.INFEASIBLE else False

    def get_routes(self):
        Routes_record = {}
        for i in range(self.model_size):
            for j in range(self.model_size):
                if i != j and self.dv_dict[f"x_{i}_{j}"].X > 0.5:
                    Routes_record[f"x_{i}_{j}"] = 1  
        find_and_print_routes(Routes_record, self.D[-1], True)
        return find_and_print_routes(Routes_record, end_node_idx=self.D[-1])

    def fleet_size(self):
        Routes_record = {}
        for i in range(self.model_size):
            for j in range(self.model_size):
                if i != j and self.dv_dict[f"x_{i}_{j}"].X > 0.5:
                    Routes_record[f"x_{i}_{j}"] = 1  
        return find_fleet_size(Routes_record, end_node_idx=self.D[-1])

    def print_results(self, Optimal_Value=True, DV_Info=True, Routes=True, MIDGAP=True):
        """
        Print the results including optimal objective value, decision variables, and (optional) MIPGap.
        """
        if Optimal_Value:
            try:
                print("Optimal Value:", self.model.ObjVal)
            except:
                print("No feasible solution found. ObjVal unavailable.")
                return

        # ✅ Print current MIP gap if available
        if MIDGAP:
            if self.model.SolCount > 0:
                print(f"Current MIP Gap: {self.model.MIPGap * 100:.2f}%")
            else:
                print("No feasible solution yet. Current MIP Gap unavailable.")
                return

        if DV_Info or Routes:
            Routes_record = {}
            for i in range(self.model_size):
                for j in range(self.model_size):
                    if i != j and self.dv_dict[f"x_{i}_{j}"].X > 0.5:
                        if DV_Info:
                            print(f"x_{i}_{j}: {self.dv_dict[f'x_{i}_{j}'].X}")
                        Routes_record[f"x_{i}_{j}"] = 1

            if DV_Info:
                print("Travel Time:", {f"tao_{i}": self.dv_dict[f'tao_{i}'].X for i in range(self.model_size)})
                print("Load Capacity:", {f"u_{i}": self.dv_dict[f'u_{i}'].X for i in range(self.model_size)})
                print("Battery Level:", {f"y_{i}": self.dv_dict[f'y_{i}'].X for i in range(self.model_size)})

            if Routes:
                find_and_print_routes(Routes_record, end_node_idx=self.D[-1])
