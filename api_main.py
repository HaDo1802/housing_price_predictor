"""
FastAPI Inference Service

Production-ready API for housing price predictions.
Includes: health checks, model versioning, error handling, request validation.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List
import logging
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.inference_pipeline import InferencePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Housing Price Prediction API",
    description="Production ML API for predicting house prices",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference pipeline (loaded once at startup)
inference_pipeline: Optional[InferencePipeline] = None


# ==========================================
# Request/Response Models
# ==========================================

class HouseFeatures(BaseModel):
    """Input features for prediction"""
    # Numerical features
    lot_area: float = Field(..., gt=0, description="Lot size in square feet")
    total_bsmt_sf: float = Field(..., ge=0, description="Total basement area in square feet")
    first_flr_sf: float = Field(..., gt=0, alias="1st Flr SF")
    second_flr_sf: float = Field(..., ge=0, alias="2nd Flr SF")
    gr_liv_area: float = Field(..., gt=0, description="Above grade living area")
    garage_area: float = Field(..., ge=0, description="Garage size in square feet")
    overall_qual: int = Field(..., ge=1, le=10, description="Overall quality rating")
    overall_cond: int = Field(..., ge=1, le=10, description="Overall condition rating")
    year_built: int = Field(..., ge=1800, le=2026, description="Original construction year")
    year_remod_add: int = Field(..., ge=1800, le=2026, description="Remodel year")
    bedroom_abvgr: int = Field(..., ge=0, le=10, description="Bedrooms above grade")
    full_bath: int = Field(..., ge=0, le=5, description="Full bathrooms")
    half_bath: int = Field(..., ge=0, le=3, description="Half bathrooms")
    totrms_abvgrd: int = Field(..., ge=0, le=20, description="Total rooms above grade")
    fireplaces: int = Field(..., ge=0, le=5, description="Number of fireplaces")
    garage_cars: int = Field(..., ge=0, le=5, description="Garage capacity in cars")
    
    # Categorical features
    neighborhood: str = Field(..., description="Physical neighborhood")
    ms_zoning: str = Field(..., description="General zoning classification")
    bldg_type: str = Field(..., description="Type of dwelling")
    house_style: str = Field(..., description="Style of dwelling")
    foundation: str = Field(..., description="Type of foundation")
    central_air: str = Field(..., pattern="^[NY]$", description="Central air conditioning")
    garage_type: str = Field(..., description="Garage location")
    
    @validator('year_remod_add')
    def validate_remod_year(cls, v, values):
        """Ensure remodel year is not before build year"""
        if 'year_built' in values and v < values['year_built']:
            raise ValueError('Remodel year cannot be before build year')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "lot_area": 8450,
                "total_bsmt_sf": 856,
                "1st Flr SF": 856,
                "2nd Flr SF": 854,
                "gr_liv_area": 1710,
                "garage_area": 500,
                "overall_qual": 7,
                "overall_cond": 5,
                "year_built": 2003,
                "year_remod_add": 2003,
                "bedroom_abvgr": 3,
                "full_bath": 2,
                "half_bath": 1,
                "totrms_abvgrd": 8,
                "fireplaces": 1,
                "garage_cars": 2,
                "neighborhood": "CollgCr",
                "ms_zoning": "RL",
                "bldg_type": "1Fam",
                "house_style": "2Story",
                "foundation": "PConc",
                "central_air": "Y",
                "garage_type": "Attchd"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: float = Field(..., description="Predicted sale price in USD")
    confidence_interval: Dict[str, float] = Field(..., description="95% confidence interval")
    model_version: str = Field(..., description="Model version used")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 195000.00,
                "confidence_interval": {
                    "lower": 175000.00,
                    "upper": 215000.00
                },
                "model_version": "production_v1.0"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    features: List[HouseFeatures]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_processed: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


# ==========================================
# Startup/Shutdown Events
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global inference_pipeline
    try:
        logger.info("Loading inference pipeline...")
        inference_pipeline = InferencePipeline('models/production')
        logger.info("✓ Inference pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load inference pipeline: {e}")
        # In production, you might want to fail fast here
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")


# ==========================================
# API Endpoints
# ==========================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Housing Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    model_loaded = inference_pipeline is not None
    model_version = None
    
    if model_loaded:
        try:
            model_version = inference_pipeline.metadata.get('model_type', 'unknown')
        except:
            pass
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=model_version
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"]
)
async def predict(features: HouseFeatures):
    """
    Make a single prediction
    
    Returns the predicted house price with confidence interval.
    """
    if inference_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Convert Pydantic model to dict with proper column names
        feature_dict = {
            'Lot Area': features.lot_area,
            'Total Bsmt SF': features.total_bsmt_sf,
            '1st Flr SF': features.first_flr_sf,
            '2nd Flr SF': features.second_flr_sf,
            'Gr Liv Area': features.gr_liv_area,
            'Garage Area': features.garage_area,
            'Overall Qual': features.overall_qual,
            'Overall Cond': features.overall_cond,
            'Year Built': features.year_built,
            'Year Remod/Add': features.year_remod_add,
            'Bedroom AbvGr': features.bedroom_abvgr,
            'Full Bath': features.full_bath,
            'Half Bath': features.half_bath,
            'TotRms AbvGrd': features.totrms_abvgrd,
            'Fireplaces': features.fireplaces,
            'Garage Cars': features.garage_cars,
            'Neighborhood': features.neighborhood,
            'MS Zoning': features.ms_zoning,
            'Bldg Type': features.bldg_type,
            'House Style': features.house_style,
            'Foundation': features.foundation,
            'Central Air': features.central_air,
            'Garage Type': features.garage_type
        }
        
        # Make prediction
        import pandas as pd
        df = pd.DataFrame([feature_dict])
        
        predictions, lower_bounds, upper_bounds = inference_pipeline.predict_with_uncertainty(df)
        
        prediction = float(predictions[0])
        lower = float(prediction + lower_bounds[0])
        upper = float(prediction + upper_bounds[0])
        
        logger.info(f"Prediction made: ${prediction:,.2f}")
        
        return PredictionResponse(
            prediction=prediction,
            confidence_interval={
                "lower": lower,
                "upper": upper
            },
            model_version="production_v1.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"]
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions
    
    Process multiple predictions in a single request for efficiency.
    """
    if inference_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        predictions_list = []
        
        for features in request.features:
            # Convert features (similar to single prediction)
            feature_dict = {
                'Lot Area': features.lot_area,
                'Total Bsmt SF': features.total_bsmt_sf,
                '1st Flr SF': features.first_flr_sf,
                '2nd Flr SF': features.second_flr_sf,
                'Gr Liv Area': features.gr_liv_area,
                'Garage Area': features.garage_area,
                'Overall Qual': features.overall_qual,
                'Overall Cond': features.overall_cond,
                'Year Built': features.year_built,
                'Year Remod/Add': features.year_remod_add,
                'Bedroom AbvGr': features.bedroom_abvgr,
                'Full Bath': features.full_bath,
                'Half Bath': features.half_bath,
                'TotRms AbvGrd': features.totrms_abvgrd,
                'Fireplaces': features.fireplaces,
                'Garage Cars': features.garage_cars,
                'Neighborhood': features.neighborhood,
                'MS Zoning': features.ms_zoning,
                'Bldg Type': features.bldg_type,
                'House Style': features.house_style,
                'Foundation': features.foundation,
                'Central Air': features.central_air,
                'Garage Type': features.garage_type
            }
            
            import pandas as pd
            df = pd.DataFrame([feature_dict])
            
            preds, lower_bounds, upper_bounds = inference_pipeline.predict_with_uncertainty(df)
            
            prediction = float(preds[0])
            lower = float(prediction + lower_bounds[0])
            upper = float(prediction + upper_bounds[0])
            
            predictions_list.append(
                PredictionResponse(
                    prediction=prediction,
                    confidence_interval={"lower": lower, "upper": upper},
                    model_version="production_v1.0"
                )
            )
        
        logger.info(f"Batch prediction completed: {len(predictions_list)} predictions")
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            total_processed=len(predictions_list)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model metadata and information"""
    if inference_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        return {
            "model_type": inference_pipeline.metadata.get('model_type'),
            "hyperparameters": inference_pipeline.metadata.get('hyperparameters'),
            "metrics": {
                "test": inference_pipeline.metadata.get('test_metrics'),
                "validation": inference_pipeline.metadata.get('val_metrics')
            },
            "features": {
                "count": len(inference_pipeline.metadata.get('feature_names', [])),
                "names": inference_pipeline.metadata.get('feature_names', [])
            },
            "training_info": {
                "train_size": inference_pipeline.metadata.get('train_size'),
                "val_size": inference_pipeline.metadata.get('val_size'),
                "test_size": inference_pipeline.metadata.get('test_size')
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}"
        )


# ==========================================
# Error Handlers
# ==========================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )