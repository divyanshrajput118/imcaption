from imgCaption.constants import *
from imgCaption.utils.common import read_yaml,create_directories
from imgCaption.entity.config_entity import DataIngestionConfig
from imgCaption.entity.config_entity import DataTransformationConfig
from imgCaption.entity.config_entity import PrepareBaseModelConfig
from imgCaption.entity.config_entity import ModelTrainerConfig
from imgCaption.entity.config_entity import ModelEvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
            config = self.config.data_ingestion

            create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                                            root_dir=Path(config.root_dir),
                                            source_URL=config.source_URL,
                                            local_data_file=Path(config.local_data_file),
                                            unzip_dir=Path(config.unzip_dir) 
                                        )

            return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation
        params = self.params

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
                                                root_dir=Path(config.root_dir),
                                                captions_file=Path(config.captions_file),
                                                vectorizer_path=Path(config.vectorizer_path),
                                                train_images_captions_path=Path(config.train_images_captions_path),
                                                test_images_captions_path=Path(config.test_images_captions_path),
                                                SPLIT_SIZE=params.SPLIT_SIZE,
                                                RANDOM_STATE=params.RANDOM_STATE
                                                )

        return data_transformation_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
                                                root_dir=Path(config.root_dir),
                                                image_feex_path=Path(config.image_feex_path),
                                                main_model_path=Path(config.main_model_path),
                                                IMAGE_SIZE=params.IMAGE_SIZE,
                                                WEIGHTS=params.WEIGHTS,
                                                INCLUDE_TOP=params.INCLUDE_TOP,
                                                POOLING=params.POOLING,
                                                CNN_DIM=params.CNN_DIM,
                                                VOCAB_SIZE=params.VOCAB_SIZE,
                                                MAX_LENGTH=params.MAX_LENGTH,
                                                EMBEDDING_DIM=params.EMBEDDING_DIM,
                                                LSTM_UNITS=params.LSTM_UNITS,
                                                DROPOUT=params.DROPOUT
                                            )

        return prepare_base_model_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        training = self.config.model_trainer
        prepare_base_model = self.config.prepare_base_model
        data_transformation = self.config.data_transformation
        params = self.params

        create_directories([training.root_dir])

        prepare_model_trainer_config = ModelTrainerConfig(
                                            root_dir=Path(training.root_dir),
                                            image_feex_path=Path(prepare_base_model.image_feex_path),
                                            main_model_path=Path(prepare_base_model.main_model_path),
                                            vectorizer_path=Path(data_transformation.vectorizer_path),
                                            images_dir=Path(training.images_dir),
                                            trained_model_path=Path(training.trained_model_path),
                                            train_images_captions_path=Path(data_transformation.train_images_captions_path),
                                            best_model_path=Path(training.best_model_path),
                                            LEARNING_RATE=params.LEARNING_RATE,
                                            EPOCHS=params.EPOCHS,
                                            PATIENCE=params.PATIENCE,
                                            SPLIT_SIZE=params.SPLIT_SIZE,
                                            RANDOM_STATE=params.RANDOM_STATE,
                                            MAX_LENGTH=params.MAX_LENGTH,
                                            BATCH_SIZE=params.BATCH_SIZE
                                        )

        return prepare_model_trainer_config

    def get_evaluation_config(self) -> ModelEvaluationConfig:
        eval_cfg = self.config.model_evaluation
        training = self.config.model_trainer
        data_transformation = self.config.data_transformation
        prepare_base_model = self.config.prepare_base_model

        create_directories([eval_cfg.root_dir])

        evaluation_config = ModelEvaluationConfig(
                                                    root_dir=Path(eval_cfg.root_dir),
                                                    vectorizer_path=Path(data_transformation.vectorizer_path),
                                                    images_dir=Path(training.images_dir),
                                                    test_images_captions_path=Path(data_transformation.test_images_captions_path),
                                                    image_feex_path=Path(prepare_base_model.image_feex_path),
                                                    trained_model_path=Path(training.trained_model_path),
                                                    scores_path=Path(eval_cfg.scores_path),
                                                    MAX_LENGTH=self.params.MAX_LENGTH,
                                                    BEAM_WIDTH=self.params.BEAM_WIDTH,
                                                )

        return evaluation_config