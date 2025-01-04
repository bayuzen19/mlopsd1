from src.data_preparation import load_and_prepare_data
from src.model import create_pipeline, train_model
from src.evaluation import evaluate_model

def main():
    # Load dan prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Create dan train model
    pipeline = create_pipeline()
    model = train_model(pipeline, X_train, y_train)

    # Evaluasi model
    evaluate_model(model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()