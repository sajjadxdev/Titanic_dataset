import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('Titanic_csv.pkl')

model = load_model()

# Create the web app
st.title("Titanic Survival Analysis")
st.write("Enter passenger information for survival prediction analysis")

# Create input form
with st.form("prediction_form"):
    # Create two columns for a better layout
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class"
        )
        
        sex = st.selectbox(
            "Sex (1 = Male, 0 = Female)",
            options=[1, 0],
            help="1 = Male, 0 = Female"
        )
        
        age = st.number_input(
            "Age",
            min_value=0,
            max_value=100,
            value=30,
            help="Enter passenger's age"
        )
    
    with col2:
        sibsp = st.number_input(
            "Number of Siblings/Spouses Aboard",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of siblings or spouses traveling with the passenger"
        )
        
        parch = st.number_input(
            "Number of Parents/Children Aboard",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of parents or children traveling with the passenger"
        )
        
        fare = st.number_input(
            "Fare",
            min_value=0.0,
            max_value=600.0,
            value=32.0,
            help="Passenger fare in pounds"
        )

    submit_button = st.form_submit_button("Analyze")

# Make prediction when form is submitted
if submit_button:
    # Prepare the input data
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display results
    st.header("Analysis Results")
    
    # Calculate survival probability
    survival_prob = prediction_proba[0][1] * 100
    death_prob = prediction_proba[0][0] * 100 # Calculate death probability

    # Create two columns for the results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Survival Prediction",
            value="Lived" if prediction[0] == 1 else "Died"  # Changed to "Lived" or "Died"
        )
    
    with col2:
       with col2:
        if prediction[0] == 1: # If the prediction is survival
            st.metric(
                label="Survival Probability",  # Label changed for clarity
                value=f"{survival_prob:.1f}%"
            )
        else: # If the prediction is death
            st.metric(
                label="Death Probability",  # Label changed for clarity
                value=f"{death_prob:.1f}%"  # Display death probability
            )

# Add information about the features
with st.expander("Model Information"):
    st.write("""
        This analysis uses a logistic regression model based on historical Titanic passenger data. 
        The model considers the following factors:
        
        - **Passenger Class**: Ticket class (1st, 2nd, or 3rd)
        - **Sex**: Passenger's sex (1 = Male, 0 = Female)
        - **Age**: Passenger's age
        - **SibSp**: Number of siblings/spouses aboard
        - **Parch**: Number of parents/children aboard
        - **Fare**: Passenger fare in pounds
    """)

# Add footer
st.markdown("""
    <div style='text-align: center; color: red; padding: 10px;'>
        Analysis based on historical data from the RMS Titanic (1912)
    </div>
""", unsafe_allow_html=True)