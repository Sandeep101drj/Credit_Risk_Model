import streamlit as st
import pickle
import pandas as pd

# CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f0f5;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px;
    }
    .stNumberInput input, .stSelectbox select, .stTextInput input, .stSlider, .stTextArea textarea {
        background-color: white !important;
        border-radius: 8px;
        border: 1px solid #ccc !important;
        padding: 10px !important;
        color: #333;
    }
    .stSelectbox div {
        background-color: transparent !important;
    }
    footer {
        visibility: hidden;
    }
    .footer-text {
        font-size: 12px;
        text-align: center;
        padding: 10px;
        color: grey;
    }
    .css-1v3fvcr {
        background-color: #f5f5f5 !important;
        color: black !important;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .block-container {
        max-width: 900px;
        margin: auto;
    }
    .input-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
    }
    .input-container > div {
        flex: 1;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load transformer and model
transformer = pickle.load(open('transformer.pkl', 'rb'))
model = pickle.load(open('gb_model.pkl', 'rb'))

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["💳 Credit Risk Prediction", "📊 Batch Prediction", "ℹ️ About"])

if page == "💳 Credit Risk Prediction":
    # Credit Risk Prediction Page
    st.title("💳 Credit Risk Prediction")

    # Center input form
    with st.container():
        st.subheader("Enter Credit application details")

        # Grouping input fields into sections
        st.markdown('<div class="section-title">Personal Details</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 3])  # Adjusting column widths for better alignment

        with col1:
            person_age = st.number_input('🧑‍🦱 Age of Borrower', min_value=18, max_value=100, step=1, help="Enter the age of the borrower (18-100).")
            person_income = st.number_input('💵 Income of Borrower (in USD)', min_value=1000, step=100, help="Enter the annual income in USD.")
            person_emp_length = st.number_input('👨‍💼 Employment Length (in years)', min_value=0, step=1, help="Enter the number of years the borrower has been employed.")
                    
        with col2:
            cb_person_cred_hist_length = st.number_input('🗂️ Credit History Length (in years)', min_value=0, step=1, help="Enter the length of the borrower's credit history.")
            person_home_ownership = st.selectbox('🏠 Home Ownership', ['RENT', 'MORTGAGE', 'OWN','OTHER'], help="Select the home ownership status of the borrower.")
            loan_intent = st.selectbox('🎯 Loan Intent', ['EDUCATION', 'MEDICAL', 'VENTURE','PERSONAL','DEBTCONSOLIDATION','HOMEIMPROVEMENT'], help="Select the purpose of the loan.")

        # Loan Details Section
        st.markdown('<div class="section-title">Loan Details</div>', unsafe_allow_html=True)
        col3, col4 = st.columns([2, 3])  # Adjusting columns to keep it aligned with Personal Details

        with col3:
            loan_amnt = st.number_input('💰 Loan Amount (in USD)', min_value=1000, step=100, help="Enter the loan amount in USD.")
            loan_int_rate = st.number_input('📊 Loan Interest Rate (%)', min_value=1.0, max_value=30.0, step=0.1, help="Enter the loan interest rate (1-30%).")

        loan_percent_income=loan_amnt/person_income if person_income > 0 else 0
        # Credit Bureau Section (Below Loan Details)
        st.markdown('<div class="section-title">Credit Bureau Report</div>', unsafe_allow_html=True)
        cb_person_default_on_file = st.selectbox('⚠️ Default on file', ['Y', 'N'], help="Select if the borrower has a default on file.")

        # Addding some space before the prediction button
        st.markdown("<br>", unsafe_allow_html=True)

        # Prediction button
        if st.button('🔮 Predict Credit Risk'):
            # Prepare the data for prediction
            input_data = pd.DataFrame({
                'person_age': [person_age],
                'person_income': [person_income],
                'person_home_ownership': [person_home_ownership],
                'person_emp_length': [person_emp_length],
                'loan_intent': [loan_intent],
                'loan_amnt': [loan_amnt],
                'loan_int_rate': [loan_int_rate],
                'loan_percent_income': [loan_percent_income],
                'cb_person_default_on_file': [cb_person_default_on_file],
                'cb_person_cred_hist_length': [cb_person_cred_hist_length]
            })

            # Transforming the input using the pre-saved transformer
            transformed_input = transformer.transform(input_data)

            # Predicting the credit risk using the pre-trained model
            prediction = model.predict(transformed_input)
            prediction_proba = model.predict_proba(transformed_input)

            # Get the prediction result and probabilities
            risk_prob = prediction_proba[0][1]  # Probability of Default (1)
            no_risk_prob = prediction_proba[0][0]  # Probability of No Default (0)

            # Display results with both default and non-default probabilities
            if risk_prob > 0.5:  # Using a threshold to decide high risk
                st.error(f"🚨 The borrower has a high credit risk (default probability: {round(risk_prob * 100, 2)}%)")
            else:
                st.success(f"✅ The borrower is at low credit risk (default probability: {round(risk_prob * 100, 2)}%)")

            # Show both probabilities (default and non-default)
            st.write(f"**Probability of Default**: {round(risk_prob * 100, 2)}%")
            st.write(f"**Probability of No Default**: {round(no_risk_prob * 100, 2)}%")

elif page == "📊 Batch Prediction":
    # New page for CSV Upload and Prediction
    st.title("📊 Upload CSV and View Prediction Results")

    # Display required data format information
    st.markdown("""
    ### Required Data Format for CSV File
    The CSV file should contain the following columns with respective data types:

    | Column Name                        | Data Type      | Description                                   |
    |-------------------------------------|----------------|-----------------------------------------------|
    | `person_age`                        | Numeric (int)  | Age of the borrower                 |
    | `person_income`                     | Numeric (float)| Annual income of the borrower (in USD)       |
    | `person_home_ownership`             | Categorical    | Home ownership status: 'RENT', 'MORTGAGE', 'OWN', 'OTHER' |
    | `person_emp_length`                 | Numeric (int)  | Number of years the borrower has been employed |
    | `loan_intent`                       | Categorical    | Loan intent: ' EDUCATION ' , ' MEDICAL ' ,' VENUTRE ', ' PERSONAL ', ' DEBTCONSOLIDATION ' , ' HOMEIMPROVEMENT ' |
    | `loan_amnt`                         | Numeric (float)| Loan amount requested (in USD)               |
    | `loan_int_rate`                     | Numeric (float)| Loan interest rate ( percentage figure )                  |
    | `cb_person_default_on_file`         | Categorical    | Whether the borrower has a default on file: 'Y' or 'N' |
    | `cb_person_cred_hist_length`        | Numeric (int)  | Length of the borrower's credit history (in years) |

    Please ensure that the column names are exact and the data types match the description.
    """, unsafe_allow_html=True)

    # Option for file upload
    uploaded_file = st.file_uploader("Upload CSV File for Predictions", type=["csv"])

    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        input_data = pd.read_csv(uploaded_file)

        # Show the uploaded file's preview
        st.write("### Preview of Uploaded Data:")
        st.write(input_data.head())

        # Check if the required columns are present
        required_columns = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
                            'loan_intent', 'loan_amnt', 'loan_int_rate',
                            'cb_person_default_on_file', 'cb_person_cred_hist_length']
        if all(col in input_data.columns for col in required_columns):
            
            # Calculate loan_percent_income as an internal calculation
            input_data['loan_percent_income'] = input_data['loan_amnt'] / input_data['person_income']
            
            # Transform the input using the pre-saved transformer
            transformed_input = transformer.transform(input_data[required_columns +['loan_percent_income']])

            # Predict the credit risk using the pre-trained model
            prediction = model.predict(transformed_input)
            prediction_proba = model.predict_proba(transformed_input)

            # Add predictions and probabilities to the original data
            input_data['Prediction'] = prediction
            input_data['Default Probability'] = prediction_proba[:, 1]  # Probability of Default (1)
            input_data['No Default Probability'] = prediction_proba[:, 0]  # Probability of No Default (0)

            # Show results with both default and non-default probabilities
            st.write("### Prediction Results:")
            st.write(input_data)

            # Button to download the prediction results
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="Download Prediction Results",
                data=input_data.to_csv(index=False),
                file_name="credit_risk_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("The uploaded CSV file does not contain the required columns.")
    
elif page == "ℹ️ About":
    st.title("ℹ️ About")
    st.write("This app predicts the credit risk based on borrower information using a machine learning model.")
    st.write("Choose the 'Credit Risk Prediction' page to input data manually or the 'CSV Prediction Results' page to upload and predict using a CSV file.")

# Add a footer
st.markdown('<div class="footer-text">Made by Sandeep Pradhan.</div>', unsafe_allow_html=True)
st.markdown('<div class="footer-text">© 2024 Credit Risk Prediction App. All Rights Reserved.</div>', unsafe_allow_html=True)

