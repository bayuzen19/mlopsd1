import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mrmr import mrmr_regression
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('data_preparation')

def load_and_prepare_data():
    """Load and prepare data for modeling"""
    try:
        # Load data
        logger.info("Loading data from CSV...")
        df = pd.read_csv(Config.DATA_PATH)
        
        # Split features and target
        X = df.drop("MEDV", axis=1)
        y = np.log(df["MEDV"])
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE
        )
        
        # Feature selection
        feature_list = mrmr_regression(X_train, y_train, K=Config.NUM_FEATURES)
        logger.info(f"Selected features: {', '.join(feature_list)}")
        
        X_train = X_train[feature_list]
        X_test = X_test[feature_list]
        
        logger.info("Data preparation completed successfully")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise
