"""This module provides functionality to execute the model prediction service."""

import random

from src.logger import logger
from src.model_service import ModelService, Record


@logger.catch
def main() -> None:
    """Executes the prediction pipeline service."""
    try:
        logger.info("Starting the rental prediction service...")
        # instantiate an object of type, 'Record'
        record: Record = Record(
            year_built=random.choice(range(1960, 2024)),
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

        # instantiate an object of type, 'ModelService'
        model_service: ModelService = ModelService()

        # load the trained model
        model_service.load_model()

        # get the prediction
        prediction: int = model_service.predict(record)
        logger.info(f"Estimated rent: ${prediction}")
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
