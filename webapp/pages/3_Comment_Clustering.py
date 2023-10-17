import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from utils.styles import title_style, paragraph_style
from utils.function import process_dataframe

st.set_page_config(page_title="Quinten DataScience | Comment Clustering", layout="wide")

data_raw = pd.read_csv("raw_data_healthcare.csv").drop("text_index",axis=1)
data = process_dataframe(data_raw)

# Top Page
st.markdown(title_style, unsafe_allow_html=True)
st.markdown(paragraph_style, unsafe_allow_html=True)
st.markdown('<h1 class="title">Comment Clustering</h1>', unsafe_allow_html=True)

# Clustering (Aristide)