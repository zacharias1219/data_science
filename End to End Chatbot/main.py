import os
import nltk
import ssl
import random
import streamlit as st 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

