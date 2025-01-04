import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏠 House Price Prediction")
st.markdown("""
This app predicts house prices based on various features. Enter the details below to get a prediction.
""")

def main():
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter House Details")
        
        # Create input form
        with st.form("prediction_form"):
            # Input fields with descriptions and reasonable ranges
            lstat = st.slider("% Lower status of the population (LSTAT)", 
                            min_value=0.0, max_value=40.0, value=10.0,
                            help="Percentage of lower status population in the area")
            
            rm = st.slider("Average number of rooms (RM)", 
                         min_value=3.0, max_value=9.0, value=6.0,
                         help="Average number of rooms per dwelling")
            
            crim = st.number_input("Crime rate (CRIM)", 
                                min_value=0.0, max_value=100.0, value=0.1,
                                help="Per capita crime rate by town")
            
            ptratio = st.slider("Pupil-Teacher Ratio (PTRATIO)", 
                              min_value=12.0, max_value=22.0, value=15.0,
                              help="Pupil-Teacher ratio by town")
            
            indus = st.slider("Proportion of non-retail business acres (INDUS)", 
                            min_value=0.0, max_value=30.0, value=10.0,
                            help="Proportion of non-retail business acres per town")
            
            tax = st.slider("Property tax rate (TAX)", 
                          min_value=150.0, max_value=800.0, value=300.0,
                          help="Full-value property tax rate per $10,000")
            
            nox = st.slider("Nitric oxide concentration (NOX)", 
                          min_value=0.3, max_value=0.9, value=0.5,
                          help="Nitric oxide concentration (parts per 10 million)")
            
            b = st.slider("Black population ratio (B)", 
                        min_value=0.0, max_value=400.0, value=300.0,
                        help="1000(Bk - 0.63)^2, Bk is proportion of blacks by town")

            submitted = st.form_submit_button("Predict Price")

            if submitted:
                # Prepare the data for API request
                input_data = {
                    "LSTAT": lstat,
                    "RM": rm,
                    "CRIM": crim,
                    "PTRATIO": ptratio,
                    "INDUS": indus,
                    "TAX": tax,
                    "NOX": nox,
                    "B": b
                }

                try:
                    # Make prediction using FastAPI endpoint
                    response = requests.post("http://localhost:8000/predict", 
                                          json=input_data)
                    
                    if response.status_code == 200:
                        prediction = response.json()["prediction"]
                        
                        # Store the prediction and input data in session state
                        if 'predictions' not in st.session_state:
                            st.session_state.predictions = []
                        
                        st.session_state.predictions.append({
                            "prediction": prediction,
                            **input_data
                        })
                        
                        # Display prediction
                        st.success(f"Predicted House Price: ${prediction:,.2f}")
                        
                    else:
                        st.error("Error making prediction. Please try again.")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Error connecting to the prediction service. Please make sure the API is running.")

    with col2:
        st.subheader("Previous Predictions")
        
        if 'predictions' in st.session_state and st.session_state.predictions:
            # Create DataFrame of predictions
            df_predictions = pd.DataFrame(st.session_state.predictions)
            
            # Display recent predictions
            for idx, pred in enumerate(df_predictions.tail(5).iloc[::-1].to_dict('records')):
                with st.expander(f"Prediction {len(df_predictions) - idx}", expanded=idx == 0):
                    st.write(f"Price: ${pred['prediction']:,.2f}")
                    st.write(f"Rooms: {pred['RM']}")
                    st.write(f"Crime Rate: {pred['CRIM']:.4f}")
                    st.write(f"Lower Status %: {pred['LSTAT']}%")
            
            # Add visualization
            st.subheader("Prediction Analysis")
            
            # Scatter plot of predictions vs rooms
            fig = px.scatter(df_predictions, x='RM', y='prediction',
                           title='Price vs Number of Rooms',
                           labels={'RM': 'Number of Rooms',
                                  'prediction': 'Predicted Price ($)'})
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()