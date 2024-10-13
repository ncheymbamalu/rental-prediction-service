"""This module provides the functionality for executing the ML model building process."""

from src.logger import logger
from src.model_builder import ModelBuilderService


@logger.catch
def main() -> None:
    """Executes the ML model building process."""
    try:
        # instantiate an object of type, 'ModelBuilderService'
        service: ModelBuilderService = ModelBuilderService()

        # execute the model building process
        service.build_model()
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
