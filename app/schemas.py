from pydantic import BaseModel, Field

class CropInput(BaseModel):
    """Input features for crop prediction."""

    N: int = Field(..., ge=0, le=200, description="Nitrogen content (mg/kg)")
    P: int = Field(..., ge=0, le=200, description="Phosphorus content (mg/kg)")
    K: int = Field(..., ge=0, le=250, description="Potassium content (mg/kg)")
    temperature: float = Field(..., ge=-10, le=60, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall (mm)")

class TopPrediction(BaseModel):
    """One entry in the top-3 predictions list."""

    crop: str
    probability: float

class PredictionResponse(BaseModel):
    """Response returned by /predict"""

    predicted_crop: str
    confidence: float
    top_3: list[TopPrediction]
