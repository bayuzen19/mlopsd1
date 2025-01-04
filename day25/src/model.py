from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('model')

def create_pipeline():
    """Create preprocessing and model pipeline"""
    try:
        pipeline = Pipeline([
            ('preprocessor', StandardScaler()),
            ('regressor', XGBRegressor(random_state=Config.RANDOM_STATE))
        ])
        logger.info("Pipeline created successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}")
        raise

def train_model(pipeline, X_train, y_train):
    """Train model with hyperparameter optimization"""
    try:
        logger.info("Starting model training with RandomizedSearchCV...")
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=Config.PARAMS,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1,
            random_state=Config.RANDOM_STATE
        )
        
        random_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best score: {random_search.best_score_}")
        
        # Save model
        with open(Config.MODEL_PATH, 'wb') as f:
            pickle.dump(random_search.best_estimator_, f)
        logger.info(f"Model saved to {Config.MODEL_PATH}")
        
        return random_search.best_estimator_
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise