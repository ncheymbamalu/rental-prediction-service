"""This module provides the functionality for executing the rental prediction service."""

import random

from src.logger import logger
from src.model_inference import ModelInferenceService


@logger.catch
def main() -> None:
    """Executes the rental prediction service."""
    try:
        logger.info("Starting the rental prediction service...")
        # create an input record
        record: dict[str, float | int | str] = {
            "year_built": random.choice(range(1900, 2024)),
            "area": random.choice(range(0, 300)),
            "bedrooms": random.choice(range(1, 6)),
            "bathrooms": random.choice(range(1, 4)),
            "furnished": random.choice(["no", "yes"]),
            "storage": random.choice(["no", "yes"]),
            "garage": random.choice(["no", "yes"]),
            "parking": random.choice(["no", "yes"]),
            "balcony": random.choice(["no", "yes"]),
            "garden_size": random.choice(range(0, 500)),
            "neighborhood_id": random.choice(range(1, 283))
        }

        # instantiate an object of type, 'ModelInferenceService'
        service: ModelInferenceService = ModelInferenceService()

        # load the trained ML model
        service.load_model()

        # get the input record's prediction
        prediction: int = service.predict(record)
        logger.info(f"The estimated rent is ${prediction}.")
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
