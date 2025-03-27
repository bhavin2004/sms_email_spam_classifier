import streamlit as st
import pandas as pd
import os
from src.sms_spam_classifier.pipelines.prediction_pipeline import PredictionPipeline
from src.sms_spam_classifier.pipelines.training_pipeline import Training_Pipeline
from streamlit_option_menu import option_menu

# Set Streamlit page settings
st.set_page_config(page_title="Spam Checker", page_icon="artifacts/spam.svg")

# Sidebar menu
with st.sidebar:
    selection = option_menu(
        menu_title="Menu",                      # Title of the menu
        options=["Home", "Predict with CSV", "About Project"],  # Menu options
        icons=["house", "filetype-csv", "file-earmark-person-fill"],  # Optional icons (from FontAwesome)
        menu_icon="list",                      # Main menu icon
        default_index=0,                       # Default selected item
        orientation="vertical",                # Options: "horizontal" or "vertical"
    )
# Check which page is selected
if selection == "Home":
    st.title("📩 SMS Spam Classifier System")
    
    # Initialize session state for SMS input
    if "sms" not in st.session_state:
        st.session_state.sms = ""

    # Load test data
    csv_path = os.path.join("artifacts", "test_data.csv")

    # Check if CSV file exists
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Check if enough spam and ham samples are available
        if len(df[df['v1'] == 'spam']) >= 10 and len(df[df['v1'] == 'ham']) >= 10:
            spam_df = df[df['v1'] == 'spam'].sample(10)
            ham_df = df[df['v1'] == 'ham'].sample(10)
        else:
            st.warning("⚠️ Not enough sample data available to display.")
            spam_df, ham_df = None, None
    else:
        st.error(f"❌ CSV file not found at path: {csv_path}")
        spam_df, ham_df = None, None

    # Layout with two columns for spam and ham sample selection
    col1, col2 = st.columns(2)

    # Spam Sample Selector
    with col1:
        with st.expander("📛 Spam Sample"):
            sample = st.selectbox(
                label="Select The Sample Data",
                options=["Choose an option"] + [f"Sample Data {i}" for i in range(1, 6)],
                index=0,
                key=1,
            )
            spam_sample_btn = st.button("Select Spam Sample", key=13, use_container_width=True, type="primary")
            if sample != "Choose an option" and spam_sample_btn:
                i = int(sample.split(" ")[-1]) - 1
                if spam_df is not None:
                    st.session_state.sms = spam_df.iloc[i]["v2"]

    # Ham Sample Selector
    with col2:
        with st.expander("✅ Ham Sample"):
            ham_sample = st.selectbox(
                label="Select The Sample Data",
                options=["Choose an option"] + [f"Sample Data {i}" for i in range(1, 6)],
                index=0,
                key=11,
            )
            ham_sample_btn = st.button("Select Ham Sample", key=2, use_container_width=True, type="primary")
            if ham_sample != "Choose an option" and ham_sample_btn:
                i = int(ham_sample.split(" ")[-1]) - 1
                if ham_df is not None:
                    st.session_state.sms = ham_df.iloc[i]["v2"]

    # Input for SMS text (with sample pre-filled if selected)
    st.header("💬 Enter the SMS here:")
    sms1 = st.text_area("", value=st.session_state.sms).strip()

    # Warn if SMS is empty before prediction
    if len(sms1) == 0:
        st.warning("⚠️ Please enter an SMS before predicting.")

    # Prediction button
    if st.button("🔮 Predict Spam/Ham"):
        if len(sms1) > 0:
            try:
                # Create PredictionPipeline object
                predict_obj = PredictionPipeline()

                # Run the prediction
                prediction = predict_obj.run_pipeline(sms=sms1)

                # Display prediction result
                if prediction.lower() == "spam":
                    st.error(f"🚨 The SMS you provided is classified as **{prediction}**.")
                else:
                    st.success(f"✅ The SMS you provided is classified as **{prediction}**.")
            except Exception as e:
                st.error(f"❌ Error occurred during prediction: {e}")
        else:
            st.warning("⚠️ Please enter a message before predicting.")


elif selection == "Predict with CSV":
    st.title("📊 Predict Spam in CSV")
    st.warning("CSV FILE SHOULD HAVE MESSAGE COLUMN IN IT")
    # File uploader for CSV
    uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read CSV file
            df_uploaded = pd.read_csv(uploaded_file)
            
            # Check if 'v2' column exists in the uploaded CSV
            if 'v2' in df_uploaded.columns or 'message' in df_uploaded.columns:
                st.success("✅ CSV loaded successfully!")
                st.dataframe(df_uploaded.head(5))  # Show a preview of the data

                # Instantiate the Prediction Pipeline
                predict_obj = PredictionPipeline()
                if 'v2' in df_uploaded.columns:
                    df_uploaded['message'] = df_uploaded['v2']
                # Run predictions on the uploaded SMS data
                predictions = []
                for sms in df_uploaded['message']:
                    result = predict_obj.run_pipeline(sms)
                    predictions.append(result)
                
                # Add predictions to the DataFrame
                df_uploaded["Prediction"] = predictions

                # Show results
                st.subheader("📊 Prediction Results:")
                st.dataframe(df_uploaded[["message", "Prediction"]])

                # Downloadable link for the predictions
                csv_with_predictions = df_uploaded.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_with_predictions,
                    file_name="sms_spam_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error("❌ CSV file should have a column named 'v2' containing SMS messages.")
        except Exception as e:
            st.error(f"❌ Error processing the CSV file: {e}")

elif selection == "About Project":
    st.title("📚 About This Project")
    
    st.markdown("""
    ### 🤖 SMS Spam Classifier Using Machine Learning
    
    This project is a **machine learning-based SMS spam classifier** built with Python and Streamlit.
    
    #### 💡 Key Features:
    - Classifies SMS messages as either **Spam** or **Ham**.
    - Provides sample SMS data for easy testing.
    - Allows the user to input custom SMS for real-time classification.

    #### 🧠 Model and Pipeline:
    - The model is trained using **TF-IDF Vectorizer** and a **Naive Bayes Classifier**.
    - Pipeline includes pre-processing steps like tokenization, stemming, and vectorization.
    
    #### 📄 Dataset Information:
    - **SMS Spam Collection Dataset**: Contains labeled SMS messages (spam/ham) collected from various sources.
    - Available on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).
    
    #### 🛠️ Technologies Used:
    - Python 🐍
    - Scikit-Learn 🤖
    - Pandas 🧮
    - Streamlit 🎈
    
    #### 📢 Instructions:
    - Select a sample SMS from the **Spam** or **Ham** category.
    - Or enter a custom SMS message to predict whether it’s spam or not.
    
    ---
    #### 👨‍💻 Developed by:
    - **Pa Win (Bhavin Karangia)** 🎓
    - Aspiring Data Scientist with expertise in ML, AI, and Web Scraping.
    
    ---
    🚀 **GitHub Repository:** [View Project](https://github.com/bhavin2004/SMS-Spam-Classifier)
    """)

    # Add a thank you note
    st.success("🎉 Thank you for exploring this project!")
