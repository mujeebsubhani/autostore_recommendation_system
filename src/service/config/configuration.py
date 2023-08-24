

from typing import Any
from src.utils.utils import *
from src.constants import *
from src.logger import logging
from src.service.entity.artifact_entity import *


class ServicePipelineConfiguration:

    config_file_path: str
    current_time_stamp: datetime

    def __init__(self, config_file_path = CONFIG_FILE_PATH, current_time_stamp = CURRENT_TIME_STAMP):

        self.config_file_info:dict[str:str] = read_yaml_file(config_file_path)
        self.source_pipline_artifact_path = self.get_source_pipline_artifact_path()
        self.time_stamp:datetime = current_time_stamp

    def get_source_pipline_artifact_path(self) -> Union[SourcePipelineArtifactPaths, None]:
        
        try:
            source_pipeline_config = self.config_file_info[SOURCE_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(ROOT_DIR, source_pipeline_config[ROOT_PIPELINE_DIR_NAME],
                                        source_pipeline_config[ARTIFACT_DIR_NAME])
            
            source_pipeline_artifact_path = SourcePipelineArtifactPaths(artifact_dir=artifact_dir)

            logging.info(f"-- Source Artifact Directory Path : [{source_pipeline_artifact_path}]")

            return source_pipeline_artifact_path
        except Exception as e:
            logging.info(f'-- Error in get_source_pipline_artifact_path : {e}')
            return None

    def get_service_data_ingestion_artifact_path(self) -> Union[ServiceDataIngestionArtifactPaths,None]:
        
        try:

            artifact_dir_path = self.source_pipline_artifact_path.artifact_dir

            data_ingestion_artifact_dir_path = os.path.join(artifact_dir_path, DATA_INGESTION_ARTIFACT_DIR_NAME)
            
            data_ingestion_config_file_info = self.config_file_info[DATA_INGESTION_CONFIG_KEY]
            
            collaborative_filtering_clean_data_dir_path = os.path.join(data_ingestion_artifact_dir_path,
                        data_ingestion_config_file_info[DATA_INGESTION_COLLABORATIVE_FILTERING_CLEAN_DATA_DIR_NAME])
            
            collaborative_filtering_clean_data_file_path = os.path.join(collaborative_filtering_clean_data_dir_path,
                data_ingestion_config_file_info[DATA_INGESTION_COLLABORATIVE_FILTERING_CLEAN_DATA_FILE_NAME]
            )
            
            collaborative_filtering_ingested_train_data_dir_path = os.path.join(data_ingestion_artifact_dir_path,
                        data_ingestion_config_file_info[DATA_INGESTION_COLLABORATIVE_FILTERING_INGESTED_DATA_DIR_NAME],
                        data_ingestion_config_file_info[DATA_INGESTION_COLLABORATIVE_FILTERING_INGESTED_TRAIN_DATA_DIR_NAME])
            
            collaborative_filtering_ingested_train_data_file_path = os.path.join(collaborative_filtering_ingested_train_data_dir_path,
                        data_ingestion_config_file_info[DATA_INGESTION_COLLABORATIVE_FILTERING_INGESTED_TRAIN_DATA_FILE_NAME])
            
            collaborative_filtering_ingested_test_data_dir_path = os.path.join(data_ingestion_artifact_dir_path,
                        data_ingestion_config_file_info[DATA_INGESTION_COLLABORATIVE_FILTERING_INGESTED_DATA_DIR_NAME],
                        data_ingestion_config_file_info[DATA_INGESTION_COLLABORATIVE_FILTERING_INGESTED_TEST_DATA_DIR_NAME])
            
            collaborative_filtering_ingested_test_data_file_path = os.path.join(collaborative_filtering_ingested_test_data_dir_path,
                        data_ingestion_config_file_info[DATA_INGESTION_COLLABORATIVE_FILTERING_INGESTED_TEST_DATA_FILE_NAME])
            

            
            service_data_ingestion_artifact_paths = ServiceDataIngestionArtifactPaths(

                data_ingestion_artifact_dir = data_ingestion_artifact_dir_path,
                collaborative_filtering_clean_data_dir = collaborative_filtering_clean_data_dir_path,
                collaborative_filtering_ingested_train_data_dir = collaborative_filtering_ingested_train_data_dir_path,
                collaborative_filtering_ingested_test_data_dir = collaborative_filtering_ingested_test_data_dir_path,
                collaborative_filtering_clean_data_file = collaborative_filtering_clean_data_file_path,
                collaborative_filtering_ingested_train_data_file = collaborative_filtering_ingested_train_data_file_path,
                collaborative_filtering_ingested_test_data_file = collaborative_filtering_ingested_test_data_file_path
            )

            logging.info(f"-- Service Pipeline Data Ingestion Artifact Paths : [{service_data_ingestion_artifact_paths}]")

            return service_data_ingestion_artifact_paths
        except Exception as e:
            logging.info(f'-- Error in get_service_data_ingestion_artifact_path : {e}')
            return None
        
    def get_service_data_modeling_artifact_path(self) -> ServiceDataModelingArtifactPaths:

        try:
            artifact_dir_path = self.source_pipline_artifact_path.artifact_dir

            data_modeling_artifact_dir_path = os.path.join(
                artifact_dir_path,
                DATA_MODELING_ARTFACT_DIR_NAME)

            data_modeling_config_file_info = self.config_file_info[DATA_MODELING_CONFIG_KEY]

            collaborative_filtering_trained_model_dir_path = os.path.join(
                data_modeling_artifact_dir_path,
                data_modeling_config_file_info[COLLABORATIVE_FILTERING_TRAINED_MODEL_DIR_NAME]
            )

            collaborative_filtering_trained_model_file_name = os.path.join(
                collaborative_filtering_trained_model_dir_path,
                data_modeling_config_file_info[COLLABORATIVE_FILTERING_TRAINED_MODEL_FILE_NAME]
            )

            service_data_modeling_artifact_path = ServiceDataModelingArtifactPaths(

                data_modeling_artifact_dir = data_modeling_artifact_dir_path,
                collaborative_filtering_trained_model_dir = collaborative_filtering_trained_model_dir_path,
                collaborative_filtering_trained_model_file_name = collaborative_filtering_trained_model_file_name
            )

            logging.info(f"-- Service Pipeline Data Modeling Artifact Paths : [{service_data_modeling_artifact_path}]")

            return service_data_modeling_artifact_path
        except Exception as e:
            logging.info(f'-- Error in get_service_data_modeling_artifact_path : {e}')
            return None


    

    
    
