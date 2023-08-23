
from src.utils.utils import *

# Config Directory Path
ROOT_DIR = os.getcwd()
CONFIG_DIR_NAME = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR_NAME, CONFIG_FILE_NAME)

# Time Stamp Constant
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"


# Logger Constant
LOG_DIR_NAME = "logs"

# Service Pipeline Config Constants

SOURCE_PIPELINE_CONFIG_KEY = "source_pipeline_config"
ROOT_PIPELINE_DIR_NAME = "root_pipeline_dir"
SERVICE_PIPELINE_DIR_NAME = "service_pipeline_dir"
ARTIFACT_DIR_NAME = "artifact_dir"

# Data Ingestion Config Constants

DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR_NAME = "data_ingestion_artifact_dir"
DATA_INGESTION_RECOMMENDATION_RAW_DATA_DIR_NAME = "recommendation_raw_data_dir"
DATA_INGESTION_COLLABORATIVE_FILTERING_CLEAN_DATA_DIR_NAME = "collaborative_filtering_clean_data_dir"
DATA_INGESTION_COLLABORATIVE_FILTERING_INGESTED_DATA_DIR_NAME = "collaborative_filtering_ingested_data_dir"
DATA_INGESTION_COLLABORATIVE_FILTERING_INGESTED_TRAIN_DATA_DIR_NAME = "collaborative_filtering_ingested_train_data_dir"
DATA_INGESTION_COLLABORATIVE_FILTERING_INGESTED_TEST_DATA_DIR_NAME = "collaborative_filtering_ingested_test_data_dir"


# Data Modeling Config Constants

DATA_MODELING_CONFIG_KEY = "data_modeling_config"
DATA_MODELING_ARTFACT_DIR_NAME = "data_modeling"
COLLABORATIVE_FILTERING_TRAINED_MODEL_DIR_NAME = "collaborative_filtering_trained_model_dir"
COLLABORATIVE_FILTERING_TRAINED_MODEL_FILE_NAME = "collaborative_filtering_trained_model_file_name"


