
from src.utils.utils import *
from src.service.config.configuration import *
from src.training.components.data_modeling import *


class CollaborativeFilteringRecommendations:

    service_data_ingestion_artifact_paths: ServiceDataIngestionArtifactPaths
    service_data_modeling_artifact_paths: ServiceDataModelingArtifactPaths

    def __init__(self,service_data_ingestion_artifact_paths: ServiceDataIngestionArtifactPaths,
                 service_data_modeling_artifact_paths: ServiceDataModelingArtifactPaths):

        self.service_data_ingestion_artifact_paths = service_data_ingestion_artifact_paths
        self.service_data_modeling_artifact_paths = service_data_modeling_artifact_paths

    def get_collaborative_filtering_clean_train_test_dataframes(self) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                                                                                Tuple[None, None, None]]:

        try:
            collaborative_filtering_dataframe = pd.read_csv(self.service_data_ingestion_artifact_paths.collaborative_filtering_clean_data_file)
            collaborative_filtering_train_dataframe = pd.read_csv(self.service_data_ingestion_artifact_paths.collaborative_filtering_ingested_train_data_file)
            collaborative_filtering_test_dataframe = pd.read_csv(self.service_data_ingestion_artifact_paths.collaborative_filtering_ingested_test_data_file)

            return collaborative_filtering_dataframe, collaborative_filtering_train_dataframe, collaborative_filtering_test_dataframe
        except Exception as e:
            logging.info(f"Error in get_collaborative_filtering_clean_train_test_dataframes : {e}")
            return None, None, None


    def get_maximum_user_id(self) -> Union[int,None]:

        try:
            collaborative_filtering_dataframe, _, _ = self.get_collaborative_filtering_clean_train_test_dataframes()
            num_users = collaborative_filtering_dataframe.user_id.max() + 1
            return num_users
        except Exception as e:
            logging.info(f"Error in get_maximum_user_id : {e}")
            return None
        
    def get_maximum_product_id(self) -> Union[int,None]:

        try:
            collaborative_filtering_dataframe, _, _ = self.get_collaborative_filtering_clean_train_test_dataframes()
            num_items = collaborative_filtering_dataframe.product_id.max() + 1
            return num_items
        except Exception as e:
            logging.info(f"Error in get_maximum_product_id : {e}")
            return None
        
    def get_all_unique_product_ids(self) -> Union[int,None]:

        try:
            collaborative_filtering_dataframe, _, _ = self.get_collaborative_filtering_clean_train_test_dataframes()
            unique_product_ids = collaborative_filtering_dataframe.product_id.unique()
            return unique_product_ids
        except Exception as e:
            logging.info(f"Error in get_all_unique_product_ids : {e}")
            return None
        
    def get_all_unique_user_ids(self) -> Union[List[int],None]:

        try:
            collaborative_filtering_dataframe, _, _ = self.get_collaborative_filtering_clean_train_test_dataframes()
            unique_user_ids = collaborative_filtering_dataframe.user_id.unique().tolist()
            return unique_user_ids
        except Exception as e:
            logging.info(f"Error in get_all_unique_user_ids : {e}")
            return None
        
    def get_collaborative_filtering_recommendation(self, user_id, num_recommendations,
                 collaborative_filtering_train_dataframe,  maximum_user_id, maximum_product_id,
                 unique_product_ids) -> Union[List[int],None]:

        try:

            model = NeuralCollaborativeFiltering(maximum_user_id, maximum_product_id, collaborative_filtering_train_dataframe, unique_product_ids)
            model_path = self.service_data_modeling_artifact_paths.collaborative_filtering_trained_model_file_name
            model.load_state_dict(torch.load(model_path))
            unique_product_ids = self.get_all_unique_product_ids()

            user_tensor = torch.tensor([user_id])
            item_ids = torch.tensor(unique_product_ids)

            user_tensor = user_tensor.to(model.device)
            item_ids = item_ids.to(model.device)

            with torch.no_grad():
                user_embedding = model.user_embedding(user_tensor)
                item_embeddings = model.item_embedding(item_ids)
                user_embedding = user_embedding.repeat(item_embeddings.size(0), 1)

                scores = torch.sigmoid(
                    model.output(
                        model.fc4(
                            model.fc3(
                                model.fc2(
                                    model.fc1(torch.cat([user_embedding, item_embeddings], dim=-1))
                                )
                            )
                        )
                    )
                )

            _, top_indices = torch.topk(scores.view(-1), num_recommendations)
            recommended_items = item_ids[top_indices].cpu().numpy().tolist()

            return recommended_items
        except Exception as e:
            logging.info(f"Error in collaborative_filtering_recommendation : {e}")
            return None
        
    def initiate_collaborative_filtering(self, user_id, num_recommendations,
                 collaborative_filtering_train_dataframe,  maximum_user_id, maximum_product_id,
                 unique_product_ids):
    
        try:
            return self.get_collaborative_filtering_recommendation(user_id, num_recommendations,
                    collaborative_filtering_train_dataframe,  maximum_user_id, maximum_product_id,
                    unique_product_ids)
        except Exception as e:
            logging.info(f"Error in initiate_collaborative_filtering : {e}")
            return None
        


    
