from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from optimizer import MetroOptimizer
import pulp

app = FastAPI(title="Kochi Metro Induction Planner API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizationRequest(BaseModel):
    required_trains: int

class OptimizationResponse(BaseModel):
    status: str
    message: str
    solution: Optional[Dict[str, Any]] = None
    objective_value: Optional[float] = None

@app.get("/")
async def root():
    return {"message": "Kochi Metro Induction Planner API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "operational"}

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize(request: OptimizationRequest):
    try:
        optimizer = MetroOptimizer()
        
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
        solution = optimizer.solve(request.required_trains)
        
        return OptimizationResponse(
            status="success",
            message=f"Optimized induction plan for {request.required_trains} service trains",
            solution=solution,
            objective_value=pulp.value(optimizer.problem.objective)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)