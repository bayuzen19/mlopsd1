# 🏠 Housing Price Prediction - MLOps Project

## Overview
This project implements an end-to-end MLOps pipeline for predicting housing prices using the Boston Housing dataset. The system includes model training, API deployment with FastAPI, containerization with Docker, and a user-friendly Streamlit interface.

## 🏗️ Project Structure
```
mlopsd1/
│
├── frontend.py         # Streamlit interface
│
└── day25/
    ├── artifacts/
    │   ├── best_model.pkl    # Trained model
    │   └── boston.csv        # Dataset
    ├── config/
    │   └── config.py         # Configuration settings
    ├── notebook/
    │   └── research.ipynb    # Research and analysis notebook
    ├── src/
    │   ├── data_preparation.py
    │   ├── evaluation.py
    │   └── model.py
    ├── utils/
    │   └── logger.py
    ├── logs/              # Log files directory
    ├── app.py            # FastAPI application
    ├── train.py          # Model training script
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    └── .gitignore
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/bayuzen19/mlopsd1.git
cd mlopsd1

# Create virtual environment
conda create -n mlopsdbb python==3.9 -y

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model
```bash
python train.py
```

### 3. Run with Docker
```bash
# Build and start services
docker-compose up --build

# Stop services
docker-compose down
```

### 4. Access Applications
- FastAPI Swagger UI: http://localhost:8000/docs
- Streamlit Interface: http://localhost:5000

## 📊 Features

### Model Training
- Feature selection using MRMR
- XGBoost Regression model
- Hyperparameter optimization
- Model evaluation metrics

### API (FastAPI)
- RESTful endpoints for predictions
- Input validation
- Error handling
- Swagger documentation

### Frontend (Streamlit)
- Interactive input form
- Real-time predictions
- Historical prediction tracking
- Data visualization

## 📈 Model Performance

### Best Parameters
- max_depth: 8
- learning_rate: 0.1

### Evaluation Metrics
- Training R² Score: 0.9992
- Test R² Score: 0.8194
- Training RMSE: 0.2850
- Test RMSE: 3.2608

### Model Characteristics
- Strong performance on training data (R² = 0.9992)
- Good generalization to test data (R² = 0.8194)
- Low training error (RMSE = 0.2850)
- Higher but reasonable test error (RMSE = 3.2608)
- Slight overfitting indicated by the difference between train and test metrics

## 🛠️ Technologies Used
- **Machine Learning**: scikit-learn, XGBoost
- **API**: FastAPI
- **Frontend**: Streamlit
- **Containerization**: Docker
- **Other Tools**: Pandas, NumPy, Plotly

## 🔧 API Usage

### Make Prediction
```python
import requests

# Example input data
data = {
    "LSTAT": 4.98,
    "RM": 6.575,
    "CRIM": 0.00632,
    "PTRATIO": 15.3,
    "INDUS": 2.31,
    "TAX": 296,
    "NOX": 0.538,
    "B": 396.9
}

# Make prediction
response = requests.post("http://localhost:8000/predict", json=data)
prediction = response.json()["prediction"]
print(f"Predicted Price: ${prediction:,.2f}")
```

## 👥 Authors
- Bayu Zen
- [LinkedIn](https://www.linkedin.com/in/bayuzen)
- [GitHub](https://github.com/bayuzen19)

## 📧 Contact
For any queries, please reach out to:
- Email: bayuzen19@gmail.com
- LinkedIn: [Bayu Zen](https://www.linkedin.com/in/bayuzen)
