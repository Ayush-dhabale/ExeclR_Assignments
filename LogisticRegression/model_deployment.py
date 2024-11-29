import streamlit as st
import numpy as np
import pickle  # Assuming you've saved your trained model using pickle or joblib

# Load the trained Logistic Regression model
with open('Logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set up page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
    <style>
        /* Adjust footer alignment */
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            color: #999999;
        }

        /* Style the title and description */
        .main-title {
            font-size: 2.5rem;
            color: #2E86C1;
            font-weight: 700;
        }

        .description {
            font-size: 1rem;
            color: #555555;
        }

        /* Style prediction results */
        .success-result {
            color: #1D8348;
            font-weight: bold;
        }

        .error-result {
            color: #C0392B;
            font-weight: bold;
        }

        /* Link styling */
        a:link, a:visited {
            color: #2E86C1;
            text-decoration: none;
        }

        a:hover {
            color: #21618C;
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-title'>🚢 Titanic Survival Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='description'>Predict whether a Titanic passenger survived or not based on their details. Fill in the inputs below and click <b>Predict</b> to see the result!</p>", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Passenger Details")
pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
age = st.sidebar.slider("Age", 0, 80, 25, step=1, help="Age of the passenger")
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0, step=1)
parch = st.sidebar.number_input("Parents/Children Aboard (Parch)", 0, 10, 0, step=1)
fare = st.sidebar.slider("Fare", 0.0, 520.0, 30.0, step=1.0, help="Ticket fare paid by the passenger")
gender = st.sidebar.radio("Gender", ["Male", "Female"], help="Select the passenger's gender")
embarked = st.sidebar.radio(
    "Port of Embarkation",
    ["Queenstown (Q)", "Southampton (S)"],
    help="Port where the passenger embarked"
)

# Convert inputs to model format
male = 1 if gender == "Male" else 0
q_embarked = 1 if embarked == "Queenstown (Q)" else 0
s_embarked = 1 if embarked == "Southampton (S)" else 0

# Predict button
if st.sidebar.button("Predict 🚀"):
    # Prepare the input array for prediction
    features = np.array([[pclass, age, sibsp, parch, fare, male, q_embarked, s_embarked]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    # Display results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.markdown(f"<p class='success-result'>The passenger <b>survived</b> with a probability of {probability:.2%}.</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='error-result'>The passenger <b>did not survive</b> with a probability of {1 - probability:.2%}.</p>", unsafe_allow_html=True)

# Add a footer
st.markdown("<div class='footer'>Developed by <a href='https://www.linkedin.com/in/ayush-dhabale-515a98207/' target='_blank'>Ayush Dhabale</a>. Powered by <b>Streamlit</b>.</div>", unsafe_allow_html=True)
