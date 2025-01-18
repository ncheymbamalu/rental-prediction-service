"""This module contains a POST endpoint to trigger a ML model for predicting rental home prices."""

from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat, PositiveInt

from src.model_inference import ModelInferenceService

app: FastAPI = FastAPI(
    title="Rental Home Price Prediction Service",
    description="REST API to predict rental home prices in Amsterdam"
)


class RentalHome(BaseModel):
    """Represents a rental home in Amsterdam."""
    year_built: PositiveFloat = Field(
        default=datetime.now().year, ge=1900, le=datetime.now().year, alias="Year Built",
    )
    area: PositiveFloat = Field(default=150.0, gt=0, alias="Area (m²)")
    bedrooms: PositiveInt = Field(default=3, ge=1, alias="Beds")
    bathrooms: PositiveFloat = Field(default=2.0, ge=1, alias="Baths")
    furnished: str = Field(default="no", alias="Furnished")
    storage: str = Field(default="no", alias="Storage")
    garage: str = Field(default="no", alias="Garage")
    parking: str = Field(default="no", alias="Parking")
    balcony: str = Field(default="no", alias="Balcony")
    garden_size: NonNegativeFloat = Field(default=5.0, ge=0, alias="Garden Size (m²)")
    neighborhood_id: PositiveInt = Field(default=10, ge=1, le=282, alias="Neighborhood ID")


@app.post("/predict", response_model=dict[str, int])
def get_prediction(user_input: RentalHome):
    """Returns the estimated rent of a potential rental home.

    Args:
        input_data (RentalHome): Information about the rental home.

    Returns:
        dict[str, int]: Estimated rent of the rental home.
    """
    try:
        # get the input record
        record: dict[str, float | int | str] = user_input.model_dump()

        # instantiate the inference service
        service: ModelInferenceService = ModelInferenceService()

        # load the trained ML model
        service.load_model()

        # get the prediction
        prediction: int = service.predict(record)
        return {"Estimated rent (USD)": prediction}
    except Exception as e:
        raise e
