import streamlit as st
from src.sms_spam_classifier.pipelines.predicrion_pipeline import PredictionPipeline
from src.sms_spam_classifier.pipelines.training_pipeline import Training_Pipeline 
from src.sms_spam_classifier.utlis import *
import nltk
import os    



st.title("SMS Spam Classifier System")



sms = st.text_area("Enter the sms here:-")



if st.button("Predict Spam/Ham"):

    
    predict_obj = PredictionPipeline()
    prediction=predict_obj.run_pipeline(sms=sms)
    st.header("The sms you provided is "+ prediction)
