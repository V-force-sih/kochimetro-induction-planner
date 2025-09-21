from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from optimizer import MetroOptimizer
import pulp
import json
import os
import sys

# Add the parent directory to the path so we can import optimizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from optimizer import MetroOptimizer
except ImportError:
    print("Warning: Could not import MetroOptimizer. Some endpoints may not work.")

app = FastAPI(title="Kochi Metro Induction Planner API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store manual overrides
train_overrides = []

class OptimizationRequest(BaseModel):
    required_trains: int

class OverrideRequest(BaseModel):
    train_id: str
    action: str  # "service", "standby", "ibl"
    force: bool = False

class OptimizationResponse(BaseModel):
    status: str
    message: str
    solution: Optional[Dict[str, Any]] = None
    explanations: Optional[Dict[str, List[str]]] = None
    conflict_alert: Optional[str] = None
    objective_value: Optional[float] = None

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

@app.post("/override")
async def apply_override(request: OverrideRequest):
    """Apply a manual override to a train's assignment"""
    global train_overrides
    
    try:
        # Remove any existing override for this train
        train_overrides = [o for o in train_overrides if o['train_id'] != request.train_id]
        
        # Add the new override
        train_overrides.append({
            'train_id': request.train_id,
            'action': request.action,
            'force': request.force
        })
        
        return {"status": "success", "message": f"Override applied to {request.train_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying override: {str(e)}")

@app.get("/overrides")
async def get_overrides():
    """Get all current manual overrides"""
    return {"status": "success", "overrides": train_overrides}

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize(request: OptimizationRequest):
    try:
        # Create optimizer with current overrides
        optimizer = MetroOptimizer(overrides=train_overrides)
        
        if not optimizer.trains:
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
        
        # Run optimization
        result = optimizer.solve(request.required_trains)
        
        return OptimizationResponse(
            status="success",
            message=f"Optimized induction plan for {request.required_trains} service trains",
            solution=result["solution"],
            explanations=result["explanations"],
            conflict_alert=result["conflict"],
            objective_value=result["objective_value"]
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)