

from src.utils.utils import *

SourcePipelineArtifactPaths = namedtuple('SourcePipelineArtifactPath', ['artifact_dir'])

ServiceDataIngestionArtifactPaths = namedtuple("ServiceDataIngestionArtifactPath",[

    "data_ingestion_artifact_dir",
    "collaborative_filtering_clean_data_dir",
    "collaborative_filtering_ingested_train_data_dir",
    "collaborative_filtering_ingested_test_data_dir"])

ServiceDataModelingArtifactPaths = namedtuple("ServiceDataModelingArtifactPaths", [

    "data_modeling_artifact_dir",
    "collaborative_filtering_trained_model_dir",
    "collaborative_filtering_trained_model_file_name"])

