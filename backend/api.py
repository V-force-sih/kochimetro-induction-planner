from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
import pulp
import json
import os
import sys
import pandas as pd
from datetime import datetime

# Make sure optimizer can be imported when running from backend folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from optimizer import MetroOptimizer
    from ml_trainer import OptimizationTrainer
    ML_AVAILABLE = True
except ImportError:
    # Keep running but warn — endpoints that call optimizer will fail
    print("Warning: Could not import MetroOptimizer or ML components. Some endpoints may not work.")
    ML_AVAILABLE = False

app = FastAPI(title="Kochi Metro Induction Planner API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for data persistence
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OVERRIDES_PATH = os.path.normpath(os.path.join(CURRENT_DIR, "..", "data", "overrides.json"))
HISTORY_PATH = os.path.normpath(os.path.join(CURRENT_DIR, "..", "data", "optimization_history.csv"))
MODEL_PATH = os.path.normpath(os.path.join(CURRENT_DIR, "..", "models", "optimization_model.joblib"))

# Global variable to store manual overrides (loaded from disk on start)
def load_overrides_from_disk() -> List[Dict[str, Any]]:
    try:
        if os.path.exists(OVERRIDES_PATH):
            with open(OVERRIDES_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except Exception as e:
        print(f"Warning: failed to load overrides from disk: {e}")
    return []

def save_overrides_to_disk(overrides: List[Dict[str, Any]]):
    try:
        # ensure parent exists
        os.makedirs(os.path.dirname(OVERRIDES_PATH), exist_ok=True)
        with open(OVERRIDES_PATH, "w") as f:
            json.dump(overrides, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save overrides to disk: {e}")

train_overrides: List[Dict[str, Any]] = load_overrides_from_disk()

# --- Pydantic models ---
class OptimizationRequest(BaseModel):
    required_trains: int
    # NEW optional business quota — sum of branding_priority across selected service trains
    branding_min_total: Optional[float] = None

class OverrideRequest(BaseModel):
    train_id: str
    action: str  # "service", "standby", "ibl"
    force: bool = False

class OutcomeUpdateRequest(BaseModel):
    optimization_id: str  # We'll use timestamp as ID
    actual_service_trains: int
    actual_issues: List[str] = []  # e.g., ["breakdown", "maintenance_delay"]
    punctuality_score: float

class OptimizationResponse(BaseModel):
    status: str
    message: str
    solution: Optional[Dict[str, Any]] = None
    explanations: Optional[Dict[str, List[str]]] = None
    # conflict_alert can be a list of warnings now (keeps backward compat: could be None)
    conflict_alert: Optional[List[str]] = None
    objective_value: Optional[float] = None
    optimization_id: Optional[str] = None  # Added for ML tracking

# --- End models ---

# --- ML Data Collection Functions ---
def log_optimization_result(request: OptimizationRequest, optimizer, result: Dict[str, Any]) -> str:
    """Log optimization results for ML training and return optimization ID"""
    try:
        # Create a unique ID for this optimization run
        optimization_id = datetime.now().isoformat()
        
        # Extract features from current train data
        fit_trains = [t for t in optimizer.trains if optimizer._is_fit_for_service(t)]
        
        record = {
            "optimization_id": optimization_id,
            "timestamp": optimization_id,
            "required_trains": request.required_trains,
            "available_trains": len(fit_trains),
            "unfit_trains": len(optimizer.trains) - len(fit_trains),
            "critical_jobs": sum(t['job_cards']['open_critical'] for t in optimizer.trains),
            "branding_priority_avg": sum(t['branding_priority'] for t in optimizer.trains) / len(optimizer.trains),
            "mileage_avg": sum(t['mileage'] for t in optimizer.trains) / len(optimizer.trains),
            "solution_service": len(result.get("solution", {}).get("service", [])),
            "solution_standby": len(result.get("solution", {}).get("standby", [])),
            "solution_ibl": len(result.get("solution", {}).get("ibl", [])),
            "objective_value": result.get("objective_value", 0),
            "conflict": result.get("conflict") is not None,
            "branding_weight": getattr(optimizer, 'ml_weights', {}).get('branding_weight', 1.0),
            "mileage_weight": getattr(optimizer, 'ml_weights', {}).get('mileage_weight', 0.0001),
            "stabling_weight": getattr(optimizer, 'ml_weights', {}).get('stabling_weight', 0.01),
            "success": None  # This will be updated later based on actual outcomes
        }
        
        # Convert to DataFrame and save
        new_record = pd.DataFrame([record])
        
        if os.path.exists(HISTORY_PATH):
            history = pd.read_csv(HISTORY_PATH)
            history = pd.concat([history, new_record], ignore_index=True)
        else:
            history = new_record
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        history.to_csv(HISTORY_PATH, index=False)
        print(f"Logged optimization results with ID: {optimization_id}")
        
        return optimization_id
        
    except Exception as e:
        print(f"Error logging optimization results: {e}")
        return datetime.now().isoformat()  # Fallback ID

def update_optimization_outcome(optimization_id: str, outcomes: Dict[str, Any]):
    """Update optimization result with actual outcomes"""
    try:
        if not os.path.exists(HISTORY_PATH):
            return False
            
        history = pd.read_csv(HISTORY_PATH)
        
        # Find the optimization record
        if optimization_id not in history['optimization_id'].values:
            print(f"Optimization ID {optimization_id} not found in history")
            return False
            
        # Update with actual outcomes
        idx = history[history['optimization_id'] == optimization_id].index[0]
        
        history.at[idx, 'actual_service_trains'] = outcomes.get('actual_service_trains', 0)
        history.at[idx, 'issues'] = ", ".join(outcomes.get('actual_issues', []))
        history.at[idx, 'punctuality_score'] = outcomes.get('punctuality_score', 0)
        
        # Determine success (simplified heuristic)
        success = (outcomes.get('punctuality_score', 0) > 0.95 and 
                  len(outcomes.get('actual_issues', [])) == 0 and
                  abs(history.at[idx, 'solution_service'] - outcomes.get('actual_service_trains', 0)) <= 2)
        
        history.at[idx, 'success'] = success
        
        # Save updated history
        history.to_csv(HISTORY_PATH, index=False)
        print(f"Updated optimization outcome for ID: {optimization_id}")
        
        return True, success
        
    except Exception as e:
        print(f"Error updating optimization outcome: {e}")
        return False, False

def retrain_model():
    """Retrain ML model with new data"""
    if not ML_AVAILABLE:
        print("ML components not available - skipping retraining")
        return False
        
    try:
        trainer = OptimizationTrainer()
        if trainer.load_data(HISTORY_PATH):
            if trainer.prepare_data():
                if trainer.train_model():
                    # Ensure models directory exists
                    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                    trainer.save_model(MODEL_PATH)
                    print("ML model retrained successfully")
                    return True
    except Exception as e:
        print(f"Error retraining model: {e}")
    
    return False

def should_retrain_model() -> bool:
    """Check if we have enough successful data points to retrain"""
    try:
        if not os.path.exists(HISTORY_PATH):
            return False
            
        history = pd.read_csv(HISTORY_PATH)
        
        # Count successful optimizations
        successful_optimizations = history[history['success'] == True]
        
        # Retrain every 10 successful optimizations
        return len(successful_optimizations) % 10 == 0 and len(successful_optimizations) > 0
        
    except Exception as e:
        print(f"Error checking retrain condition: {e}")
        return False

# --- End ML Data Collection Functions ---

@app.get("/")
async def root():
    return {"message": "Kochi Metro Induction Planner API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "operational"}

@app.get("/trains")
async def get_trains():
    """Get all train data"""
    try:
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level and then into the data folder
        data_path = os.path.join(current_dir, "..", "data", "trains.json")
        # Normalize the path to handle the ".."
        data_path = os.path.normpath(data_path)
        
        # Debug log
        print(f"Looking for trains.json at: {data_path}")
        
        # Check if file exists
        if not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail=f"File not found: {data_path}")
        
        with open(data_path, 'r') as file:
            data = json.load(file)
        
        return {"status": "success", "data": data}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Train data file not found")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading train data: {str(e)}")

# ----------------- schedule endpoint + helpers (ADDED, non-destructive) -----------------

def _pos_to_bay(position: str) -> str:
    """
    Convert a depot position string (eg 'A12', 'IBL_02', 'STB_05', 'R25') to a Bay label.
    Deterministic mapping so frontend timelines are reproducible.
    """
    if not position:
        return "Bay5"
    p = str(position)
    # IBL and STB have clear markers
    if "IBL" in p.upper():
        return "Bay5"   # map IBL positions to Bay5 for UI
    if "STB" in p.upper():
        return "Bay4"   # map standby positions to Bay4
    # otherwise map first alpha char to Bay1..Bay5
    for ch in p:
        if ch.isalpha():
            idx = (ord(ch.upper()) - ord('A')) % 5 + 1
            return f"Bay{idx}"
    return "Bay5"

def _arrival_end_cost_for_service(trains_list, depot_layout):
    """
    Deterministic schedule assignment for service trains:
      - arrival: spaced by 4 minutes starting at 0 (0,4,8,...)
      - end: arrival + 10 + (mileage % 60)  (simple deterministic service duration)
      - cost: movement cost computed from depot_layout movement_costs using zones
    """
    schedule = []
    movement_costs = depot_layout.get("movement_costs", {})
    for i, t in enumerate(trains_list):
        arrival = i * 4
        end = arrival + 10 + (t.get("mileage", 0) % 60)
        current_pos = t.get('current_stable_position', "")
        # determine zone simply: IBL, STB -> Standby/IBL else Service
        p = str(current_pos)
        if "IBL" in p.upper():
            zone = "IBL"
        elif "STB" in p.upper():
            zone = "Standby"
        else:
            zone = "Service"
        key = f"{zone}_to_Service"
        cost = movement_costs.get(key, 3)
        # Map branding_priority to a coarse 1..3 priority for UI badges
        bp = int(t.get('branding_priority', 1))
        if bp >= 12:
            pr = 3
        elif bp >= 6:
            pr = 2
        else:
            pr = 1
        schedule.append({
            "train": t["id"],
            "bay": _pos_to_bay(current_pos),
            "path": f"{_pos_to_bay(current_pos)} → Service",
            "arrival": arrival,
            "end": end,
            "cost": cost,
            "priority": pr,
            "status": "Accepted"
        })
    return schedule

@app.get("/schedule")
async def get_schedule(required_trains: int = Query(20, ge=1)):
    """
    Frontend-friendly schedule endpoint.
    Query param: required_trains (default 20)
    Returns: {
       "scheduled": [...], "rejected": [...], "raw_solution": {...}, "explanations": {...}, "conflicts": [...]
    }
    """
    try:
        optimizer = MetroOptimizer(overrides=train_overrides)
        if not getattr(optimizer, "trains", None):
            raise HTTPException(status_code=500, detail="Train data not loaded")

        # Validate required_trains
        max_trains = len(optimizer.trains)
        if required_trains < 1 or required_trains > max_trains:
            raise HTTPException(status_code=400, detail=f"required_trains must be between 1 and {max_trains}")

        result = optimizer.solve(required_trains)
        sol = result.get("solution", {})
        service = sol.get("service", []) or []
        standby = sol.get("standby", []) or []
        ibl = sol.get("ibl", []) or []

        # Build schedule entries for service trains
        schedule = _arrival_end_cost_for_service(service, optimizer.depot_layout)

        # Rejected trains: surface conflicts (if any) as rejected-like entries for UI
        rejected = []
        conflicts = result.get("conflict") or []
        if isinstance(conflicts, list) and conflicts:
            for c in conflicts:
                rejected.append({
                    "train": "-",
                    "bay": "-",
                    "path": "-",
                    "arrival": "-",
                    "priority": "-",
                    "status": str(c)
                })

        return {
            "scheduled": schedule,
            "rejected": rejected,
            "raw_solution": sol,
            "explanations": result.get("explanations", {}),
            "conflicts": conflicts,
            "objective_value": result.get("objective_value")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- end of schedule endpoint block -----------------

@app.post("/override")
async def apply_override(request: OverrideRequest):
    """Apply or update a manual override to a train's assignment and persist to disk"""
    global train_overrides
    try:
        # Validate action
        if request.action not in {"service", "standby", "ibl"}:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'service', 'standby' or 'ibl'.")

        # Remove any existing override for this train
        train_overrides = [o for o in train_overrides if o.get('train_id') != request.train_id]

        # Add the new override
        train_overrides.append({
            'train_id': request.train_id,
            'action': request.action,
            'force': bool(request.force)
        })

        # Persist
        save_overrides_to_disk(train_overrides)

        return {"status": "success", "message": f"Override applied to {request.train_id}", "overrides": train_overrides}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying override: {str(e)}")

@app.delete("/override/{train_id}")
async def remove_override(train_id: str):
    """Remove a manual override for a specific train"""
    global train_overrides
    try:
        before = len(train_overrides)
        train_overrides = [o for o in train_overrides if o.get('train_id') != train_id]
        save_overrides_to_disk(train_overrides)
        after = len(train_overrides)
        return {"status": "success", "message": f"Removed override for {train_id}" if before != after else f"No override found for {train_id}", "overrides": train_overrides}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing override: {str(e)}")

@app.get("/overrides")
async def get_overrides():
    """Get all current manual overrides"""
    return {"status": "success", "overrides": train_overrides}

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize(request: OptimizationRequest):
    try:
        # Create optimizer with current overrides
        optimizer = MetroOptimizer(overrides=train_overrides)
        
        if not getattr(optimizer, "trains", None):
            return OptimizationResponse(
                status="error",
                message="Failed to load train data. Please check data files."
            )
        
        # Validate input
        if request.required_trains < 1 or request.required_trains > len(optimizer.trains):
            return OptimizationResponse(
                status="error",
                message=f"Required trains must be between 1 and {len(optimizer.trains)}"
            )

        # Validate branding_min_total if provided
        if request.branding_min_total is not None:
            try:
                if request.branding_min_total < 0:
                    return OptimizationResponse(
                        status="error",
                        message="branding_min_total must be non-negative if provided."
                    )
            except Exception:
                return OptimizationResponse(
                    status="error",
                    message="branding_min_total must be a number."
                )
        
        # Pass the optional quota to the optimizer (backwards compatible)
        optimizer.branding_min_total = request.branding_min_total

        # Run optimization
        result = optimizer.solve(request.required_trains)
        
        # Log this optimization for ML training
        optimization_id = log_optimization_result(request, optimizer, result)

        # Normalize conflict output to a list of strings (backwards compatible)
        raw_conflict = result.get("conflict")
        if raw_conflict is None:
            conflict_list: Optional[List[str]] = None
        elif isinstance(raw_conflict, list):
            # ensure all items are strings
            conflict_list = [str(x) for x in raw_conflict]
        else:
            # single string or other: wrap into a list
            conflict_list = [str(raw_conflict)]

        # Explanations expected as dict[str, list[str]] in the frontend
        explanations = result.get("explanations", {})

        return OptimizationResponse(
            status="success",
            message=f"Optimized induction plan for {request.required_trains} service trains",
            solution=result.get("solution"),
            explanations=explanations,
            conflict_alert=conflict_list,
            objective_value=result.get("objective_value"),
            optimization_id=optimization_id  # Return the ID for outcome tracking
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-outcome")
async def update_outcome(request: OutcomeUpdateRequest):
    """Update optimization result with actual outcomes for ML training"""
    try:
        success, was_successful = update_optimization_outcome(
            request.optimization_id,
            {
                "actual_service_trains": request.actual_service_trains,
                "actual_issues": request.actual_issues,
                "punctuality_score": request.punctuality_score
            }
        )
        
        if not success:
            return {"status": "error", "message": "Failed to update outcome"}
            
        # Retrain model if this was a successful optimization and we have enough data
        if was_successful and should_retrain_model():
            retrain_success = retrain_model()
            if retrain_success:
                return {"status": "success", "message": "Outcome updated and model retrained"}
            else:
                return {"status": "success", "message": "Outcome updated but model retraining failed"}
        
        return {"status": "success", "message": "Outcome updated successfully"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating outcome: {str(e)}")

@app.post("/retrain-model")
async def trigger_retrain():
    """Manually trigger model retraining"""
    try:
        if retrain_model():
            return {"status": "success", "message": "Model retrained successfully"}
        else:
            return {"status": "error", "message": "Model retraining failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining model: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)