import pulp
import json
import os
from typing import List, Dict

class MetroOptimizer:
    def __init__(self, overrides=None):
        self.trains = self.load_data()
        self.problem = pulp.LpProblem("Metro_Induction_Optimization", pulp.LpMaximize)
        self.cleaning_slots_available = 4
        self.ibl_capacity = 5
        self.depot_layout = self.load_depot_layout()
        self.overrides = overrides or []  # Store manual overrides
        
    def load_data(self) -> List[Dict]:
        """Load train data from JSON file"""
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, '..', 'data', 'trains.json')
        
        with open(data_path, 'r') as file:
            data = json.load(file)
        
        print(f"✓ Loaded {len(data)} trains for optimization")
        return data
    
    def load_depot_layout(self):
        """Define the physical depot layout and movement costs based on position patterns"""
        return {
            "zones": {
                "Service": {"patterns": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"]},
                "Standby": {"patterns": ["STB"]},
                "IBL": {"patterns": ["IBL"]}
            },
            "movement_costs": {
                # Cost to move between different zone types
                "Service_to_Service": 0,
                "Service_to_Standby": 2,
                "Service_to_IBL": 3,
                "Standby_to_Service": 2,
                "Standby_to_Standby": 0,
                "Standby_to_IBL": 1,
                "IBL_to_Service": 3,
                "IBL_to_Standby": 1,
                "IBL_to_IBL": 0
            }
        }
    
    def get_zone_from_position(self, position):
        """Determine which zone a position belongs to based on patterns"""
        if position is None:
            return "Unknown"
        
        position_str = str(position)
        for zone, info in self.depot_layout["zones"].items():
            for pattern in info["patterns"]:
                if pattern in position_str:
                    return zone
        return "Unknown"
    
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
        
        # 7. Apply manual overrides
        for override in self.overrides:
            train_id = override['train_id']
            action = override['action']
            force = override.get('force', False)
            
            if action == 'service':
                self.problem += (self.service_vars[train_id] == 1, f"override_service_{train_id}")
                if force:
                    # If force is True, remove fitness constraints for this train
                    # We need to find and remove the constraint
                    for constraint_name, constraint in list(self.problem.constraints.items()):
                        if constraint_name == f"fitness_constraint_{train_id}":
                            del self.problem.constraints[constraint_name]
            elif action == 'standby':
                self.problem += (self.standby_vars[train_id] == 1, f"override_standby_{train_id}")
            elif action == 'ibl':
                self.problem += (self.ibl_vars[train_id] == 1, f"override_ibl_{train_id}")
    
    def add_stabling_constraints(self):
        """Minimize stabling movement cost based on depot layout"""
        self.stabling_cost = 0  # Initialize as a numeric value that we'll build up
        
        for train in self.trains:
            train_id = train['id']
            current_zone = self.get_zone_from_position(train.get('current_stable_position', 'Unknown'))
            
            # Determine cost based on current zone and assigned role
            if current_zone == "Service":
                self.stabling_cost += 0 * self.service_vars[train_id]  # No cost if staying in service
                self.stabling_cost += self.depot_layout["movement_costs"]["Service_to_Standby"] * self.standby_vars[train_id]
                self.stabling_cost += self.depot_layout["movement_costs"]["Service_to_IBL"] * self.ibl_vars[train_id]
                
            elif current_zone == "Standby":
                self.stabling_cost += self.depot_layout["movement_costs"]["Standby_to_Service"] * self.service_vars[train_id]
                self.stabling_cost += 0 * self.standby_vars[train_id]  # No cost if staying in standby
                self.stabling_cost += self.depot_layout["movement_costs"]["Standby_to_IBL"] * self.ibl_vars[train_id]
                
            elif current_zone == "IBL":
                self.stabling_cost += self.depot_layout["movement_costs"]["IBL_to_Service"] * self.service_vars[train_id]
                self.stabling_cost += self.depot_layout["movement_costs"]["IBL_to_Standby"] * self.standby_vars[train_id]
                self.stabling_cost += 0 * self.ibl_vars[train_id]  # No cost if staying in IBL
                
            else:  # Unknown or other position - assume high cost
                self.stabling_cost += 3 * self.service_vars[train_id]
                self.stabling_cost += 3 * self.standby_vars[train_id]
                self.stabling_cost += 3 * self.ibl_vars[train_id]
    
    def set_objective(self):
        """Set the optimization objective: maximize branding priority while balancing mileage and stabling"""
        
        # Primary objective: Maximize branding exposure
        branding_obj = pulp.lpSum([
            self.service_vars[train['id']] * train['branding_priority'] 
            for train in self.trains
        ])
        
        # Secondary objective: Balance mileage (minimize maximum mileage)
        max_mileage = pulp.LpVariable("max_mileage", lowBound=0)
        for train in self.trains:
            self.problem += (max_mileage >= self.service_vars[train['id']] * train['mileage'])
        
        # Combined weighted objective - adjust weights as needed
        self.problem += branding_obj - 0.0001 * max_mileage - 0.01 * self.stabling_cost

    def _is_fit_for_service(self, train):
        """Check if train meets all fitness constraints"""
        certs = train['fitness_certificates']
        return certs['rolling_stock'] and certs['signalling'] and certs['telecom']

    def generate_explanations(self, solution):
        """Generate explanations for optimization decisions"""
        explanations = {}
        
        for train in self.trains:
            train_id = train['id']
            explanation = []
            
            # Check if train is in service
            if any(t['id'] == train_id for t in solution["service"]):
                # Check if this was an override
                is_override = any(o['train_id'] == train_id and o['action'] == 'service' for o in self.overrides)
                
                if is_override:
                    explanation.append("Manually assigned to service (override)")
                else:
                    explanation.append(f"Selected for service due to branding priority {train['branding_priority']}")
                
                # Check constraints
                if not self._is_fit_for_service(train) and not is_override:
                    explanation.append("⚠️ WARNING: Fitness constraints overridden")
                
            # Check if train is in IBL
            elif any(t['id'] == train_id for t in solution["ibl"]):
                # Check if this was an override
                is_override = any(o['train_id'] == train_id and o['action'] == 'ibl' for o in self.overrides)
                
                if is_override:
                    explanation.append("Manually assigned to IBL (override)")
                elif train['job_cards']['open_critical'] > 0:
                    explanation.append("Mandatory IBL assignment due to critical job cards")
                else:
                    explanation.append("Assigned to IBL for maintenance")
            
            # Check if train is in standby
            elif any(t['id'] == train_id for t in solution["standby"]):
                # Check if this was an override
                is_override = any(o['train_id'] == train_id and o['action'] == 'standby' for o in self.overrides)
                
                if is_override:
                    explanation.append("Manually assigned to standby (override)")
                else:
                    explanation.append("Assigned to standby pool")
            
            # Check constraints
            if not self._is_fit_for_service(train) and any(t['id'] == train_id for t in solution["service"]):
                explanation.append("⚠️ CONSTRAINT VIOLATION: Fitness certificates not valid")
                
            explanations[train_id] = explanation
        
        return explanations

    def detect_conflicts(self, required_trains: int):
        """Detect if constraints make solution impossible"""
        # Count available service trains (fit and without critical jobs)
        fit_trains = [t for t in self.trains if self._is_fit_for_service(t) and t['job_cards']['open_critical'] == 0]
        
        # Count overrides that force trains to service
        service_overrides = [o for o in self.overrides if o['action'] == 'service']
        
        available_count = len(fit_trains) + len(service_overrides)
        
        if available_count < required_trains:
            return f"Only {available_count} trains available for service (need {required_trains})"
        
        # Check IBL capacity
        ibl_overrides = [o for o in self.overrides if o['action'] == 'ibl']
        if len(ibl_overrides) > self.ibl_capacity:
            return f"Too many manual IBL assignments ({len(ibl_overrides)}), exceeds IBL capacity ({self.ibl_capacity})"
        
        return None
    
    def solve(self, required_trains: int) -> Dict:
        """Solve the optimization problem and return results"""
        print("🧠 Creating optimization model...")
        self.create_variables()
        self.add_constraints(required_trains)
        self.add_stabling_constraints()
        self.set_objective()
        
        print("⚡ Solving optimization problem...")
        self.problem.solve()
        
        print(f"✅ Status: {pulp.LpStatus[self.problem.status]}")
        print(f"📊 Objective value: {pulp.value(self.problem.objective):.2f}")
        
        # Get the basic solution
        solution = self.format_solution()
        
        # Generate explanations and check for conflicts
        explanations = self.generate_explanations(solution)
        conflict = self.detect_conflicts(required_trains)
        
        # Return enhanced solution with explanations and conflict info
        return {
            "solution": solution,
            "explanations": explanations,
            "conflict": conflict,
            "status": pulp.LpStatus[self.problem.status],
            "objective_value": pulp.value(self.problem.objective)
        }
    
    def format_solution(self) -> Dict:
        """Format the solution for easy display"""
        solution = {"service": [], "standby": [], "ibl": []}
        
        for train in self.trains:
            train_id = train['id']
            if self.service_vars[train_id].varValue > 0.5:
                solution["service"].append(train)
            elif self.standby_vars[train_id].varValue > 0.5:
                solution["standby"].append(train)
            elif self.ibl_vars[train_id].varValue > 0.5:
                solution["ibl"].append(train)
        
        return solution


# Example usage
if __name__ == "__main__":
    # Test with some overrides
    test_overrides = [
        {"train_id": "T01", "action": "service", "force": True},
        {"train_id": "T02", "action": "ibl", "force": False}
    ]
    
    optimizer = MetroOptimizer(overrides=test_overrides)
    result = optimizer.solve(required_trains=20)
    solution = result["solution"]
    
    print(f"\n🚆 Service Trains ({len(solution['service'])}):")
    for train in solution["service"]:
        print(f"   {train['id']} | Branding: {train['branding_priority']} | Mileage: {train['mileage']} | Position: {train['current_stable_position']}")
    
    print(f"\n🛑 Standby Trains ({len(solution['standby'])}):")
    for train in solution["standby"]:
        print(f"   {train['id']} | Cleaning: {'Yes' if train['requires_cleaning'] else 'No'} | Position: {train['current_stable_position']}")
    
    print(f"\n🔧 IBL Trains ({len(solution['ibl'])}):")
    for train in solution["ibl"]:
        print(f"   {train['id']} | Critical Jobs: {train['job_cards']['open_critical']} | Position: {train['current_stable_position']}")
    
    # Print explanations
    print(f"\n📝 Explanations:")
    for train_id, explanation in result["explanations"].items():
        if explanation:  # Only print if there's an explanation
            print(f"   {train_id}: {' | '.join(explanation)}")
    
    # Print conflict if any
    if result["conflict"]:
        print(f"\n⚠️ CONFLICT: {result['conflict']}")