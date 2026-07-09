from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    captions_file: Path
    vectorizer_path: Path
    train_images_captions_path: Path
    test_images_captions_path: Path

    SPLIT_SIZE: float
    RANDOM_STATE: int

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    image_feex_path: Path           
    main_model_path: Path   

    IMAGE_SIZE: list         
    WEIGHTS: str            
    INCLUDE_TOP: bool        
    POOLING: str

    CNN_DIM: int
    VOCAB_SIZE: int          
    MAX_LENGTH: int          
    EMBEDDING_DIM: int       
    LSTM_UNITS: int              
    DROPOUT: float     

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    image_feex_path: Path
    main_model_path: Path
    vectorizer_path: Path
    images_dir: Path
    trained_model_path: Path
    train_images_captions_path: Path
    best_model_path: Path
    LEARNING_RATE: float
    EPOCHS: int
    PATIENCE: int
    SPLIT_SIZE: int
    RANDOM_STATE: int         
    MAX_LENGTH: int
    BATCH_SIZE: int  

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    vectorizer_path: Path
    images_dir: Path
    test_images_captions_path: Path
    image_feex_path: Path
    trained_model_path: Path
    scores_path: Path
    all_params: dict
    mlflow_uri: str 

    MAX_LENGTH: int
    BEAM_WIDTH: int