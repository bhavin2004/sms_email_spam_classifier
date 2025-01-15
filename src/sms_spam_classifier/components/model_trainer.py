from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from pathlib import Path
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
# from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
from src.utlis import download_nltk_resource,save_pkl

@dataclass
class ModelTrainerConfig():
    similarity_path = Path('artifacts/similarity.pkl')
    
class ModelTrainer():
    
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs(os.path.dirname(self.config.similarity_path),exist_ok=True)

    def initiate_model_trainer(self,df):
        try:

            logging.info("Model Trainer Is Initialized")
            
            
            df.tags=df.tags.apply(self.hybrid_lemmatize_stem)
            vectors = self.get_vectorized_data(df.tags)
            
            #Initializing the object of Cosine_similarity
            similarity = cosine_similarity(vectors)
            # self.recommend("Avatar",new_df=df,simalarities=similarity)
            
            save_pkl(similarity,self.config.similarity_path)
            
            logging.info("Model Trainer Process Is Complete ")
            # self.recommend('Batman',df,similarity)
            
        except Exception as e:
            logging.error("Error occurres in model trainer due to {e}")
            raise CustomException(e,sys)
        
    def hybrid_lemmatize_stem(self,text):
        """
        Applies lemmatization followed by stemming on unique words from the text.

        Args:
            text (str): Input text.

        Returns:
            dict: A dictionary where keys are the original unique words,
                and values are their lemmatized and stemmed forms.
        """
        # Tokenize and get unique words
        words = set(nltk.word_tokenize(text))

        # Initialize lemmatizer and stemmer
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()

        # Apply lemmatization followed by stemming
        processed_words = [
            stemmer.stem(lemmatizer.lemmatize(word)) for word in words
        ]

        return " ".join(processed_words)    
        
    def get_vectorized_data(self,data):
        cv = CountVectorizer(max_features=5000,stop_words='english')
        vectors = cv.fit_transform(data).toarray()  
        
        return vectors
        