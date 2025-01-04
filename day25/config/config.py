import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
    DATA_PATH = ARTIFACTS_DIR / "boston.csv"
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    NUM_FEATURES = 8
    
    # Model hyperparameters
    PARAMS = {
        'regressor__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'regressor__learning_rate': [0.001, 0.01, 0.1]
    }