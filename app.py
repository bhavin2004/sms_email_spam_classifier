import streamlit as st
from src.sms_spam_classifier.pipelines.predicrion_pipeline import PredictionPipeline
from src.sms_spam_classifier.pipelines.training_pipeline import Training_Pipeline 
from src.sms_spam_classifier.utlis import *
import nltk
        


st.title("SMS Spam Classifier System")


sms = st.text_area("Enter the sms here:-")


# if st.button("Recommand the Movies"):
#     predict_obj = PredictionPipeline()
#     recommended_movies=predict_obj.run_pipeline(movie=movie)
#     # for movie_id,movie_name in recommended_movies.items():
#     movie_id=list(recommended_movies.keys())
#     movie_name = list(recommended_movies.values())
#     for index,col in enumerate(st.columns(5,gap='medium')):
#         with col:
#             st.image(fetch_poster(movie_id=movie_id[index]))
#             st.write(movie_name[index])
#     # st.write(recommended_movies)
#     # st.write(type(recommended_movies))
#     # for i in recommended_movies:
#     #     st.write(i)