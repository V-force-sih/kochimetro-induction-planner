import pulp
import json
import os
from typing import List, Dict

class MetroOptimizer:
    def __init__(self):
        self.trains = self.load_data()
        self.problem = pulp.LpProblem("Metro_Induction_Optimization", pulp.LpMaximize)
        self.cleaning_slots_available = 4
        self.ibl_capacity = 5
        
    def load_data(self) -> List[Dict]:
        """Load train data from JSON file"""
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, '..', 'data', 'trains.json')
        
        with open(data_path, 'r') as file:
            data = json.load(file)
        
        print(f"✓ Loaded {len(data)} trains for optimization")
        return data
    
    def create_variables(self):
        """Create decision variables for each train and role"""
        self.service_vars = {}
        self.standby_vars = {}
        self.ibl_vars = {}
        
        for train in self.trains:
            train_id = train['id']
            self.service_vars[train_id] = pulp.LpVariable(f"service_{train_id}", 0, 1, pulp.LpInteger)
            self.standby_vars[train_id] = pulp.LpVariable(f"standby_{train_id}", 0, 1, pulp.LpInteger)
            self.ibl_vars[train_id] = pulp.LpVariable(f"ibl_{train_id}", 0, 1, pulp.LpInteger)
    
    def add_constraints(self, required_trains: int):
        """Add all constraints to the optimization problem"""
        
        # 1. Assignment constraint: Each train gets exactly one role
        for train in self.trains:
            train_id = train['id']
            self.problem += (
                self.service_vars[train_id] + self.standby_vars[train_id] + self.ibl_vars[train_id] == 1,
                f"single_assignment_{train_id}"
            )
        
        # 2. Service requirement constraint
        self.problem += (
            pulp.lpSum([self.service_vars[train['id']] for train in self.trains]) == required_trains,
            "service_requirement"
        )
        
        # 3. Fitness certificate constraints (HARD)
        for train in self.trains:
            train_id = train['id']
            certs = train['fitness_certificates']
            if not (certs['rolling_stock'] and certs['signalling'] and certs['telecom']):
                self.problem += (self.service_vars[train_id] == 0, f"fitness_constraint_{train_id}")
        
        # 4. Job card constraints (HARD)
        for train in self.trains:
            train_id = train['id']
            if train['job_cards']['open_critical'] > 0:
                self.problem += (self.ibl_vars[train_id] == 1, f"mandatory_ibl_{train_id}")
        
        # 5. IBL capacity constraint
        self.problem += (
            pulp.lpSum([self.ibl_vars[train['id']] for train in self.trains]) <= self.ibl_capacity,
            "ibl_capacity"
        )
        
        # 6. Cleaning slot constraints
        cleaning_demand = pulp.lpSum([
            (self.standby_vars[train['id']] + self.ibl_vars[train['id']]) * train['requires_cleaning']
            for train in self.trains
        ])
        self.problem += (cleaning_demand <= self.cleaning_slots_available, "cleaning_slots")
    
    def set_objective(self):
        """Set the optimization objective: maximize branding priority while balancing mileage"""
        
        # Primary objective: Maximize branding exposure
        branding_obj = pulp.lpSum([
            self.service_vars[train['id']] * train['branding_priority'] 
            for train in self.trains
        ])
        
        # Secondary objective: Balance mileage (minimize maximum mileage)
        max_mileage = pulp.LpVariable("max_mileage", lowBound=0)
        for train in self.trains:
            self.problem += (max_mileage >= self.service_vars[train['id']] * train['mileage'])
        
        # Combined objective (weighted sum)
        self.problem += branding_obj - 0.0001 * max_mileage
    
    def solve(self, required_trains: int) -> Dict:
        """Solve the optimization problem and return results"""
        print("🧠 Creating optimization model...")
        self.create_variables()
        self.add_constraints(required_trains)
        self.set_objective()
        
        print("⚡ Solving optimization problem...")
        self.problem.solve()
        
        print(f"✅ Status: {pulp.LpStatus[self.problem.status]}")
        print(f"📊 Objective value: {pulp.value(self.problem.objective):.2f}")
        
        # Extract and return solution
        return self.format_solution()
    
    def format_solution(self) -> Dict:
        """Format the solution for easy display"""
        solution = {"service": [], "standby": [], "ibl": []}
        
        for train in self.trains:
            train_id = train['id']
            if self.service_vars[train_id].varValue == 1:
                solution["service"].append(train)
            elif self.standby_vars[train_id].varValue == 1:
                solution["standby"].append(train)
            elif self.ibl_vars[train_id].varValue == 1:
                solution["ibl"].append(train)
        
        return solution

# Example usage
if __name__ == "__main__":
    optimizer = MetroOptimizer()
    solution = optimizer.solve(required_trains=20)
    
    print(f"\n🚆 Service Trains ({len(solution['service'])}):")
    for train in solution["service"]:
        print(f"   {train['id']} | Branding: {train['branding_priority']} | Mileage: {train['mileage']}")
    
    print(f"\n🛑 Standby Trains ({len(solution['standby'])}):")
    for train in solution["standby"]:
        print(f"   {train['id']} | Cleaning: {'Yes' if train['requires_cleaning'] else 'No'}")
    
    print(f"\n🔧 IBL Trains ({len(solution['ibl'])}):")
    for train in solution["ibl"]:
        print(f"   {train['id']} | Critical Jobs: {train['job_cards']['open_critical']}")