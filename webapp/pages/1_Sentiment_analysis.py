import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from Utils.styles import title_style, paragraph_style
from Utils.function import process_dataframe

st.set_page_config(page_title="Quinten DataScience | Sentiment Analysis", layout="wide")

data_raw = pd.read_csv("raw_data_healthcare.csv").drop("text_index",axis=1)
data = process_dataframe(data_raw)

# Top Page
st.markdown(title_style, unsafe_allow_html=True)
st.markdown(paragraph_style, unsafe_allow_html=True)
st.markdown('<h1 class="title">Sentiment Analysis</h1>', unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    fig1 = plt.figure(figsize=(16, 8))
    # Treatment Distribution
    sns.countplot(data=data, x='Treatment name', order=data['Treatment name'].value_counts().index)
    plt.title('Treatment Distribution')
    plt.xlabel('')
    plt.ylabel('Count')
    st.pyplot(fig1)

with col2:
    fig2 = plt.figure(figsize=(16, 8))
    # Disease Distribution
    sns.countplot(data=data, x='Disease', order=data['Disease'].value_counts().index)
    plt.title('Disease Distribution')
    plt.xlabel('')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig2)