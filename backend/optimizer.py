import pulp
import json
import os
from typing import List, Dict, Optional
import numpy as np

# Import ML components (with fallback handling)
try:
    from ml_trainer import OptimizationTrainer
    ML_AVAILABLE = True
except ImportError:
    print("Warning: ML components not available. Using default weights.")
    ML_AVAILABLE = False

class MetroOptimizer:
    def __init__(self, overrides=None):
        # preserve original behavior: load trains from data/trains.json
        self.trains = self.load_data()
        self.problem = None  # will create per-solve
        self.cleaning_slots_available = 4
        self.ibl_capacity = 5
        self.depot_layout = self.load_depot_layout()
        self.overrides = overrides or []  # Store manual overrides

        # ML components (with fallback to default weights)
        self.ml_weights = self.load_ml_weights()
        
        # Use ML weights if available, otherwise use defaults
        self.weight_stabling = self.ml_weights.get('stabling_weight', 0.01)
        self.weight_mileage_spread = self.ml_weights.get('mileage_weight', 0.0001)
        self.weight_branding = self.ml_weights.get('branding_weight', 1.0)
        
        # If not None, enforce minimum total branding exposure
        self.branding_min_total: Optional[float] = None

    def load_ml_weights(self):
        """Load ML model and get weights for current scenario, with fallback to defaults"""
        if not ML_AVAILABLE:
            return self.get_default_weights()
            
        try:
            # Load trained model
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'optimization_model.joblib')
            trainer = OptimizationTrainer()
            
            if trainer.load_model(model_path):
                # Extract features from current data
                features = self.get_current_features()
                # Predict optimal weights
                weights = trainer.predict_weights(features)
                if weights:
                    print("✓ Using ML-optimized weights")
                    return weights
        except Exception as e:
            print(f"Error loading ML weights: {e}")
        
        # Fallback to default weights
        print("⚠️ Using default weights (ML not available)")
        return self.get_default_weights()
    
    def get_default_weights(self):
        """Return default weights for when ML is not available"""
        return {
            'branding_weight': 1.0,
            'mileage_weight': 0.0001,
            'stabling_weight': 0.01
        }
    
    def get_current_features(self):
        """Extract features from current train data for ML prediction"""
        fit_trains = [t for t in self.trains if self._is_fit_for_service(t)]
        
        return {
            'required_trains': len(self.trains) // 2,  # Default estimate
            'available_trains': len(fit_trains),
            'unfit_trains': len(self.trains) - len(fit_trains),
            'critical_jobs': sum(t['job_cards']['open_critical'] for t in self.trains),
            'branding_priority_avg': sum(t['branding_priority'] for t in self.trains) / len(self.trains),
            'mileage_avg': sum(t['mileage'] for t in self.trains) / len(self.trains)
        }

    def load_data(self) -> List[Dict]:
        """Load train data from JSON file (same path as before)."""
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, '..', 'data', 'trains.json')
        data_path = os.path.normpath(data_path)

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
        self.problem = pulp.LpProblem("Metro_Induction_Optimization", pulp.LpMaximize)
        self.service_vars = {}
        self.standby_vars = {}
        self.ibl_vars = {}

        for train in self.trains:
            train_id = train['id']
            # keep integer/binary as before
            self.service_vars[train_id] = pulp.LpVariable(f"service_{train_id}", 0, 1, pulp.LpInteger)
            self.standby_vars[train_id] = pulp.LpVariable(f"standby_{train_id}", 0, 1, pulp.LpInteger)
            self.ibl_vars[train_id] = pulp.LpVariable(f"ibl_{train_id}", 0, 1, pulp.LpInteger)

    def add_constraints(self, required_trains: int):
        """Add all constraints to the optimization problem (keeps your existing hard constraints)."""

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

        # 3. Fitness certificate constraints (HARD) - preserve original behavior
        for train in self.trains:
            train_id = train['id']
            certs = train['fitness_certificates']
            if not (certs['rolling_stock'] and certs['signalling'] and certs['telecom']):
                self.problem += (self.service_vars[train_id] == 0, f"fitness_constraint_{train_id}")

        # 4. Job card constraints (HARD) - preserve original behavior
        for train in self.trains:
            train_id = train['id']
            if train['job_cards']['open_critical'] > 0:
                self.problem += (self.ibl_vars[train_id] == 1, f"mandatory_ibl_{train_id}")

        # 5. IBL capacity constraint
        self.problem += (
            pulp.lpSum([self.ibl_vars[train['id']] for train in self.trains]) <= self.ibl_capacity,
            "ibl_capacity"
        )

        # 6. Cleaning slot constraints (same formula)
        cleaning_demand = pulp.lpSum([
            (self.standby_vars[train['id']] + self.ibl_vars[train['id']]) * (1 if train['requires_cleaning'] else 0)
            for train in self.trains
        ])
        self.problem += (cleaning_demand <= self.cleaning_slots_available, "cleaning_slots")

        # 7. Apply manual overrides (preserve original override + force behavior)
        for override in self.overrides:
            train_id = override['train_id']
            action = override['action']
            force = override.get('force', False)

            if action == 'service':
                self.problem += (self.service_vars[train_id] == 1, f"override_service_{train_id}")
                if force:
                    # If force is True, remove fitness constraints for this train if present
                    constraint_name = f"fitness_constraint_{train_id}"
                    if constraint_name in self.problem.constraints:
                        del self.problem.constraints[constraint_name]
            elif action == 'standby':
                self.problem += (self.standby_vars[train_id] == 1, f"override_standby_{train_id}")
            elif action == 'ibl':
                self.problem += (self.ibl_vars[train_id] == 1, f"override_ibl_{train_id}")

    def add_stabling_constraints(self):
        """Build stabling movement cost (keeps your approach)"""
        # We'll create a linear expression self.stabling_cost to include in objective
        self.stabling_cost = pulp.lpSum([])  # start empty sum

        for train in self.trains:
            train_id = train['id']
            current_zone = self.get_zone_from_position(train.get('current_stable_position', 'Unknown'))

            if current_zone == "Service":
                self.stabling_cost += 0 * self.service_vars[train_id]
                self.stabling_cost += self.depot_layout["movement_costs"]["Service_to_Standby"] * self.standby_vars[train_id]
                self.stabling_cost += self.depot_layout["movement_costs"]["Service_to_IBL"] * self.ibl_vars[train_id]

            elif current_zone == "Standby":
                self.stabling_cost += self.depot_layout["movement_costs"]["Standby_to_Service"] * self.service_vars[train_id]
                self.stabling_cost += 0 * self.standby_vars[train_id]
                self.stabling_cost += self.depot_layout["movement_costs"]["Standby_to_IBL"] * self.ibl_vars[train_id]

            elif current_zone == "IBL":
                self.stabling_cost += self.depot_layout["movement_costs"]["IBL_to_Service"] * self.service_vars[train_id]
                self.stabling_cost += self.depot_layout["movement_costs"]["IBL_to_Standby"] * self.standby_vars[train_id]
                self.stabling_cost += 0 * self.ibl_vars[train_id]

            else:
                # unknown/noisy positions penalized equally
                self.stabling_cost += 3 * self.service_vars[train_id]
                self.stabling_cost += 3 * self.standby_vars[train_id]
                self.stabling_cost += 3 * self.ibl_vars[train_id]

    def set_objective(self):
        """Set the optimization objective using ML-learned weights"""
        # Primary objective: Maximize branding exposure with ML weight
        branding_obj = pulp.lpSum([
            self.service_vars[train['id']] * train['branding_priority'] 
            for train in self.trains
        ]) * self.weight_branding

        # Mileage spread minimization with ML weight
        max_mileage = pulp.LpVariable("max_mileage", lowBound=0)
        min_mileage = pulp.LpVariable("min_mileage", lowBound=0)

        # compute BIG_M as an upper bound (max mileage in dataset)
        big_m = max(train['mileage'] for train in self.trains) + 1000

        # If a train is selected to service, it constrains max_mileage and min_mileage appropriately
        for train in self.trains:
            tid = train['id']
            m = train['mileage']
            # max_mileage >= m * service_var
            self.problem += (max_mileage >= self.service_vars[tid] * m)
            # min_mileage <= m + (1 - service_var) * BIG_M
            self.problem += (min_mileage <= self.service_vars[tid] * m + (1 - self.service_vars[tid]) * big_m)

        mileage_obj = (max_mileage - min_mileage) * self.weight_mileage_spread

        # Stabling cost objective with ML weight
        stabling_obj = self.stabling_cost * self.weight_stabling

        # branding_min_total enforcement if set (optional business quota)
        if self.branding_min_total is not None:
            self.problem += (
                pulp.lpSum([self.service_vars[train['id']] * train['branding_priority'] for train in self.trains])
                >= self.branding_min_total,
                "branding_min_total_quota"
            )

        # Combined objective using ML-learned weights
        self.problem += branding_obj - mileage_obj - stabling_obj

    def _is_fit_for_service(self, train):
        """Check if train meets all fitness constraints (keeps previous logic)"""
        certs = train['fitness_certificates']
        return certs['rolling_stock'] and certs['signalling'] and certs['telecom']

    def generate_explanations(self, solution):
        """Generate richer multi-factor explanations for each train (keeps backwards compatibility)."""
        explanations = {}

        # compute avg mileage among selected service trains for relative comments
        service_trains = solution.get("service", [])
        avg_mileage = None
        if service_trains:
            avg_mileage = sum(t['mileage'] for t in service_trains) / len(service_trains)

        for train in self.trains:
            train_id = train['id']
            explanation = []

            in_service = any(t['id'] == train_id for t in solution["service"])
            in_ibl = any(t['id'] == train_id for t in solution["ibl"])
            in_standby = any(t['id'] == train_id for t in solution["standby"])

            # Check if this was overridden
            is_override_service = any(o['train_id'] == train_id and o['action'] == 'service' for o in self.overrides)
            is_override_ibl = any(o['train_id'] == train_id and o['action'] == 'ibl' for o in self.overrides)
            is_override_standby = any(o['train_id'] == train_id and o['action'] == 'standby' for o in self.overrides)

            # Role-specific reasons
            if in_service:
                if is_override_service:
                    explanation.append("Manually forced to Service (override)")
                explanation.append(f"Branding priority: {train['branding_priority']}")
                explanation.append(f"Mileage: {train['mileage']} km")
                if avg_mileage is not None:
                    if train['mileage'] > avg_mileage * 1.15:
                        explanation.append("Higher-than-average mileage (wear risk)")
                    elif train['mileage'] < avg_mileage * 0.85:
                        explanation.append("Lower-than-average mileage (good for balancing)")
                # stabling cost for chosen assignment
                zone = self.get_zone_from_position(train.get('current_stable_position'))
                move_cost = 0
                if zone == "Service":
                    move_cost = 0
                elif zone == "Standby":
                    move_cost = self.depot_layout["movement_costs"]["Standby_to_Service"]
                elif zone == "IBL":
                    move_cost = self.depot_layout["movement_costs"]["IBL_to_Service"]
                else:
                    move_cost = 3
                explanation.append(f"Estimated shunting cost: {move_cost}")
                if train['requires_cleaning']:
                    explanation.append("Requires cleaning (scheduled before service)")

                # constraint warnings (if any)
                if not self._is_fit_for_service(train) and not is_override_service:
                    explanation.append("⚠️ Fitness certificates missing (should not be in service)")
                if train['job_cards']['open_critical'] > 0 and not is_override_service:
                    explanation.append("⚠️ Critical job-cards open (should be in IBL)")

            elif in_ibl:
                if is_override_ibl:
                    explanation.append("Manually assigned to IBL (override)")
                if train['job_cards']['open_critical'] > 0:
                    explanation.append("Assigned to IBL due to open critical job-cards")
                else:
                    explanation.append("Assigned to IBL for maintenance / preventive checks")
                zone = self.get_zone_from_position(train.get('current_stable_position'))
                if zone == "IBL":
                    explanation.append("Already in IBL (no shunt needed)")

            elif in_standby:
                if is_override_standby:
                    explanation.append("Manually assigned to Standby (override)")
                explanation.append("Held in standby pool to provide morning availability")
                if train['requires_cleaning']:
                    explanation.append("Requires cleaning (slot reserved)")

            # Additional explicit warnings if override forced a broken constraint
            if is_override_service and not self._is_fit_for_service(train):
                explanation.append("⚠️ FORCED: fitness constraints ignored by override")

            if is_override_service and train['job_cards']['open_critical'] > 0:
                explanation.append("⚠️ FORCED: critical job-cards ignored by override")

            # Add ML weight information to explanations
            if ML_AVAILABLE and (in_service or in_standby or in_ibl):
                explanation.append(f"ML weights: B({self.weight_branding:.3f}) M({self.weight_mileage_spread:.5f}) S({self.weight_stabling:.3f})")

            explanations[train_id] = explanation

        return explanations

    def detect_conflicts(self, required_trains: int, solution: Optional[Dict] = None) -> List[str]:
        """
        Detect if constraints or configuration create conflicts.
        Returns a list of conflict/warning strings (empty list if no conflicts).
        """
        warnings = []

        # 1) Availability check (fit trains + service overrides)
        fit_trains = [t for t in self.trains if self._is_fit_for_service(t) and t['job_cards']['open_critical'] == 0]
        service_overrides = [o for o in self.overrides if o['action'] == 'service']
        available_count = len(fit_trains) + len(service_overrides)
        if available_count < required_trains:
            warnings.append(f"Only {available_count} trains appear available for service (need {required_trains}).")

        # 2) IBL capacity vs forced IBLs
        ibl_overrides = [o for o in self.overrides if o['action'] == 'ibl']
        if len(ibl_overrides) > self.ibl_capacity:
            warnings.append(f"Too many manual IBL assignments ({len(ibl_overrides)}) exceed IBL capacity ({self.ibl_capacity}).")

        # 3) If solution provided, check cleaning slots & IBL capacity & branding quota violations
        if solution:
            # cleaning slot usage among standby + ibl
            cleaning_count = 0
            for t in solution.get("standby", []) + solution.get("ibl", []):
                # solution contains train objects in your original implementation; safe-checking
                tid = t['id'] if isinstance(t, dict) else t
                train_obj = next((x for x in self.trains if x['id'] == tid), None)
                if train_obj and train_obj['requires_cleaning']:
                    cleaning_count += 1
            if cleaning_count > self.cleaning_slots_available:
                warnings.append(f"Cleaning demand {cleaning_count} exceeds available slots ({self.cleaning_slots_available}).")

            # IBL occupancy
            ibl_count = len(solution.get("ibl", []))
            if ibl_count > self.ibl_capacity:
                warnings.append(f"IBL occupancy {ibl_count} > capacity {self.ibl_capacity}.")

            # branding quota (if set)
            if self.branding_min_total is not None:
                total_branding_selected = sum(t['branding_priority'] for t in solution.get("service", []))
                if total_branding_selected < self.branding_min_total:
                    warnings.append(f"Branding exposure shortfall: selected total {total_branding_selected} < required {self.branding_min_total}.")

        return warnings

    def solve(self, required_trains: int) -> Dict:
        """Solve the optimization problem and return results; keeps your return signature."""
        print("🧠 Creating optimization model...")
        self.create_variables()
        self.add_constraints(required_trains)
        self.add_stabling_constraints()
        self.set_objective()

        print("⚡ Solving optimization problem...")
        self.problem.solve()

        print(f"✅ Status: {pulp.LpStatus[self.problem.status]}")
        objective_value = pulp.value(self.problem.objective) if self.problem.objective is not None else None
        print(f"📊 Objective value: {objective_value if objective_value is not None else 'N/A'}")

        # Get the basic solution in the same format as your original `format_solution`
        solution = self.format_solution()

        # Generate richer explanations and improved conflicts (multiple warnings)
        explanations = self.generate_explanations(solution)
        conflicts = self.detect_conflicts(required_trains, solution)

        # Return enhanced solution with explanations and conflict list
        return {
            "solution": solution,
            "explanations": explanations,
            "conflict": conflicts,  # now a list (backward-compatible usage: check type in api)
            "status": pulp.LpStatus[self.problem.status],
            "objective_value": objective_value
        }

    def format_solution(self) -> Dict:
        """Format the solution for easy display (keeps your original output style)."""
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


# Example usage (kept very similar to your original)
if __name__ == "__main__":
    test_overrides = [
        {"train_id": "T01", "action": "service", "force": True},
        {"train_id": "T02", "action": "ibl", "force": False}
    ]

    optimizer = MetroOptimizer(overrides=test_overrides)
    # OPTIONAL: set a branding quota for demo (uncomment to test)
    # optimizer.branding_min_total = 100  # e.g. require total branding_priority >= 100
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

    print(f"\n📝 Explanations (sample):")
    for train_id, explanation in result["explanations"].items():
        if explanation:
            print(f"   {train_id}: {' | '.join(explanation)}")

    if result["conflict"]:
        # `conflict` is now a list of warnings; keep behaviour friendly for API consumers
        print("\n⚠️ CONFLICTS:")
        for c in result["conflict"]:
            print("  -", c)