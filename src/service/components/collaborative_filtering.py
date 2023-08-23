
from src.utils.utils import *
from src.service.entity.artifact_entity import *
from src.training.components.data_modeling import *


class CollaborativeFilteringRecommendations:

    neural_collaborative_filter : NeuralCollaborativeFiltering
    service_data_ingestion_artifact_paths: ServiceDataIngestionArtifactPaths

    def __init__(self, neural_collaborative_filter :NeuralCollaborativeFiltering,
                 service_data_ingestion_artifact_paths: ServiceDataIngestionArtifactPaths):

        self.neural_collaborative_filter = neural_collaborative_filter
        self.service_data_ingestion_artifact_paths = service_data_ingestion_artifact_paths

    
    
    
