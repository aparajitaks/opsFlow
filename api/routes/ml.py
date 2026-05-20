from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.schemas import PredictRequest, PredictResponse, ModelStatusResponse
from services.ml_service import ml_service
from models.pipeline import run_ml_pipeline

router = APIRouter(prefix="/api/model", tags=["ML Predictive Maintenance"])

@router.post("/predict", response_model=PredictResponse)
async def predict_equipment_failure(req: PredictRequest):
    """
    Predict machine failure class (0 or 1) and failure probability using telemetry measurements.
    """
    try:
        # Convert Pydantic fields to dictionary format expected by the model
        telemetry_data = {
            "Type": req.Type,
            "Air temperature [K]": req.Air_temperature,
            "Process temperature [K]": req.Process_temperature,
            "Rotational speed [rpm]": req.Rotational_speed,
            "Torque [Nm]": req.Torque,
            "Tool wear [min]": req.Tool_wear
        }
        
        # Call ML Service
        result = ml_service.predict_failure(telemetry_data, model_type=req.model_type)
        return result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=400, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference processing error: {str(e)}")

@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Retrieves the active model status: best model type, F1-scores, best hyperparameters,
    and feature importance.
    """
    try:
        status = ml_service.get_model_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model status: {str(e)}")

@router.post("/retrain")
async def trigger_retraining_pipeline(background_tasks: BackgroundTasks):
    """
    Triggers the ML training and evaluation pipeline in the background.
    Logs metrics to MLflow and serializes best models.
    """
    try:
        # Launch retraining in background task to avoid blocking the API thread
        background_tasks.add_task(run_ml_pipeline)
        return {
            "status": "training_started",
            "message": "Equipment failure ML retraining pipeline triggered in the background. Check logs or model status endpoint for completions."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start retraining pipeline: {str(e)}")
