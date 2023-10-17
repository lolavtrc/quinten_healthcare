import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from utils.styles import title_style, paragraph_style
from utils.plotting import plot_sentiment_distribution

st.set_page_config(page_title="Quinten DataScience | Sentiment Analysis", layout="wide")

sentiment_data = pd.read_csv("./data/sentiment_data2.csv").drop("text_index",axis=1)
fig_sentiment = plot_sentiment_distribution(sentiment_data)

# Top Page
st.markdown(title_style, unsafe_allow_html=True)
st.markdown(paragraph_style, unsafe_allow_html=True)
st.markdown('<h1 class="title">Sentiment Analysis</h1>', unsafe_allow_html=True)

st.divider()

# Pie chart Sentimental analysis (Maxime)

st.pyplot(fig_sentiment)