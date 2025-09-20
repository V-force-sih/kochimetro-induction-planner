import pulp
import json
import os
from typing import List, Dict

class MetroOptimizer:
    def __init__(self):
        self.trains = self.load_data()
        self.problem = pulp.LpProblem("Metro_Induction_Optimization", pulp.LpMaximize)
        
    def load_data(self) -> List[Dict]:
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, '..', 'data', 'trains.json')
        
        with open(data_path, 'r') as file:
            data = json.load(file)
        
        print(f"Loaded {len(data)} trains for optimization")
        return data
    
    def create_variables(self):
        # We'll create decision variables for each train
        self.service_vars = []
        for i, train in enumerate(self.trains):
            var = pulp.LpVariable(f"service_{train['id']}", 0, 1, pulp.LpInteger)
            self.service_vars.append(var)
        
    def add_constraints(self, required_trains: int):
        # Add constraint: exactly required_trains must be in service
        self.problem += pulp.lpSum(self.service_vars) == required_trains
        
        # Add fitness certificate constraints
        for i, train in enumerate(self.trains):
            if not (train['fitness_certificates']['rolling_stock'] and 
                    train['fitness_certificates']['signalling'] and 
                    train['fitness_certificates']['telecom']):
                self.problem += self.service_vars[i] == 0
        
        # Add job card constraints
        for i, train in enumerate(self.trains):
            if train['job_cards']['open_critical'] > 0:
                self.problem += self.service_vars[i] == 0
    
    def set_objective(self):
        # Maximize branding priority of selected trains
        objective = pulp.lpSum([
            self.service_vars[i] * train['branding_priority'] 
            for i, train in enumerate(self.trains)
        ])
        self.problem += objective
    
    def solve(self, required_trains: int) -> List[Dict]:
        self.create_variables()
        self.add_constraints(required_trains)
        self.set_objective()
        
        # Solve the problem
        self.problem.solve()
        
        # Extract solution
        solution = []
        for i, train in enumerate(self.trains):
            if pulp.value(self.service_vars[i]) == 1:
                solution.append(train)
        
        return solution

if __name__ == "__main__":
    optimizer = MetroOptimizer()
    solution = optimizer.solve(required_trains=20)  # Assume we need 20 trains
    print(f"Selected {len(solution)} trains for service:")
    for train in solution:
        print(f"- {train['id']} (Branding priority: {train['branding_priority']})")