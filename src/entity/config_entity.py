from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    constructors_data_path: Path
    drivers_data_path: Path
    races_data_path: Path
    results_data_path: Path
    status_data_path: Path
    processed_data_path: Path
    train_data_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    train_array_path: Path
    test_array_path: Path
    preprocessor_obj_file_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_file_path: Path
    train_array_path: Path
    test_array_path: Path