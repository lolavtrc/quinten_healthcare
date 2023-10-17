import streamlit as st
import pandas as pd 
from PIL import Image

from utils.styles import title_style, paragraph_style
from utils.function import process_dataframe

st.set_page_config(page_title="Quinten DataScience | Topic Modeling", layout="wide")

# data_raw = pd.read_csv("raw_data_healthcare.csv").drop("text_index",axis=1)
# data = process_dataframe(data_raw)

# Top Page
st.markdown(title_style, unsafe_allow_html=True)
st.markdown(paragraph_style, unsafe_allow_html=True)
st.markdown('<h1 class="title">Topic Modeling</h1>', unsafe_allow_html=True)

comments_map = Image.open('./images/comments_map.png')
topics_distance = Image.open('./images/topics_distance.png')
topics_words = Image.open('./images/topics_words.png')

st.subheader("Analysis of the different comments")
st.image(comments_map, caption='Map of the different comments',use_column_width=True)

st.image(topics_words, caption='Main words used for each topic', use_column_width=True)
st.image(topics_distance, caption='Relation between topics',use_column_width=False)

# LDAvis (Stanford) (Aristide)
  

