from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from utils.logger import setup_logger

logger = setup_logger('evaluation')

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance"""
    try:
        # Make predictions
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_r2': r2_score(y_train, pred_train),
            'test_r2': r2_score(y_test, pred_test),
            'train_rmse': mean_squared_error(np.exp(y_train), np.exp(pred_train), squared=False),
            'test_rmse': mean_squared_error(np.exp(y_test), np.exp(pred_test), squared=False)
        }
        
        # Log metrics
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            
        return metrics
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise