
from src.service.components.collaborative_filtering import *
from src.service.config.configuration import *


class ServicePipelineRecommendations:

    def __init__(self):

        self.service_pipeline_configuration_obj = ServicePipelineConfiguration()
        self.collaborative_filter_recommendation_obj = CollaborativeFilteringRecommendations(
                                service_data_ingestion_artifact_paths = self.service_pipeline_configuration_obj.get_service_data_ingestion_artifact_path(),
                                service_data_modeling_artifact_paths = self.service_pipeline_configuration_obj.get_service_data_modeling_artifact_path())

    def recommendation_compilation(self, user_id, num_recommendations):
        try:
            recommendations = None
            _,collaborative_filtering_train_dataframe,_ = self.collaborative_filter_recommendation_obj.get_collaborative_filtering_clean_train_test_dataframes()
            maximum_user_id = self.collaborative_filter_recommendation_obj.get_maximum_user_id()
            maximum_product_id = self.collaborative_filter_recommendation_obj.get_maximum_product_id()
            unique_product_ids = self.collaborative_filter_recommendation_obj.get_all_unique_product_ids()
            unique_user_ids = self.collaborative_filter_recommendation_obj.get_all_unique_user_ids()

            if user_id in unique_user_ids:
                recommendations = self.collaborative_filter_recommendation_obj.initiate_collaborative_filtering(user_id, num_recommendations,
                 collaborative_filtering_train_dataframe,  maximum_user_id, maximum_product_id,
                 unique_product_ids)
            else:
                print ("Top Items")

            return recommendations

        except Exception as e:
            logging.info(f"Error in get_collaborative_filtering_clean_train_test_dataframes : {e}")
            return None
        
def initiate_service_pipeline_recommendations(user_id:int, num_recommendations:int) -> Union[List[int], None]:

    try:
        service_pipeline_recommendations_obj = ServicePipelineRecommendations()
        return service_pipeline_recommendations_obj.recommendation_compilation(user_id, num_recommendations)
    except Exception as e:
            logging.info(f"Error in initiate_service_pipeline_recommendations : {e}")
            return None
    