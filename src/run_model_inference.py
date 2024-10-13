"""This module provides the functionality for executing the rental prediction service."""

import random

from src.logger import logger
from src.model_inference import ModelInferenceService, Record


@logger.catch
def main() -> None:
    """Executes the rental prediction service."""
    try:
        logger.info("Starting the rental prediction service...")
        # instantiate an object of type, 'Record'
        record: Record = Record(
            year_built=random.choice(range(1900, 2024)),
            area=float(random.choice(range(0, 300))),
            bedrooms=random.choice(range(1, 6)),
            bathrooms=random.choice(range(1, 4)),
            garden_size=float(random.choice(range(0, 500))),
            balcony=random.choice([True, False]),
            parking=random.choice([True, False]),
            furnished=random.choice([True, False]),
            garage=random.choice([True, False]),
            storage=random.choice([True, False]),
            neighborhood_id=random.choice(range(1, 282))
        )

        # instantiate an object of type, 'ModelInferenceService'
        service: ModelInferenceService = ModelInferenceService()

        # load the trained ML model
        service.load_model()

        # get the prediction
        prediction: int = service.predict(record)
        logger.info(f"The estimated rent is ${prediction}.")
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
