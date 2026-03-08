from src.pinn_landslide.exception.exception import customexception
from src.pinn_landslide.pipeline.stage_1_data_ingestion_pipeline import DataIngestionPipeline
from src.pinn_landslide.pipeline.stage_2_data_loader_pipeline import DataLoaderPipeline
from src.pinn_landslide.pipeline.stage_3_model_training import ModelTrainingPipeline
from src.pinn_landslide.logger.logger import logger
import sys

STAGE_NAME = "Data Ingestion Stage"
try:
    print(f">>>>>>>>>{STAGE_NAME}<<<<<<<<<<<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.run()
    logger.info(f">>>>>>>>{STAGE_NAME}  completed<<<<<<<<<<<<")

except Exception as e:
    raise customexception(e, sys)

STAGE_NAME = "Data Loader Stage"
try:
    print(f">>>>>>>>>{STAGE_NAME}<<<<<<<<<<<<<<<")
    data_loader = DataLoaderPipeline()
    data_loader.run()
    logger.info(f">>>>>>>>{STAGE_NAME}  completed<<<<<<<<<<<<")

except Exception as e:
    raise customexception(e, sys)

STAGE_NAME = "Model Training Stage"
try:
    print(f">>>>>>>>>{STAGE_NAME}<<<<<<<<<<<<<<<")
    model_training = ModelTrainingPipeline()
    model_training.run()
    logger.info(f">>>>>>>>{STAGE_NAME}  completed<<<<<<<<<<<<")

except Exception as e:
    raise customexception(e, sys)
