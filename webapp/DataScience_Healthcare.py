import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from Utils.styles import title_style, paragraph_style
from Utils.function import process_dataframe

st.set_page_config(page_title="Quinten DataScience", layout="wide")

data_raw = pd.read_csv("data/raw_data_healthcare.csv").drop("text_index",axis=1)
data = process_dataframe(data_raw)

# Top Page
st.markdown(title_style, unsafe_allow_html=True)
st.markdown(paragraph_style, unsafe_allow_html=True)
st.markdown('<h1 class="title">Quinten DataScience</h1>', unsafe_allow_html=True)

st.subheader("Introduction:")

st.markdown(f"""<p class="paragraph"> This project aims to extract essential insights from a dataset of consumer 
                reviews about medications used to treat intestinal diseases, such as Crohn's disease and colitis, 
                which often require long-term medication. The goal is to assist a pharmaceutical company in understanding 
                how its products are perceived by consumers and in identifying potential side effects.</p>""", unsafe_allow_html=True)


st.markdown('<h1 class="small-title">Project Objectives</h1>', unsafe_allow_html=True)

st.markdown(f"""
                <ul class="paragraph">
                    <li><strong>Data Preprocessing:</strong> Preprocess the data by performing operations such as punctuation removal, 
                    converting text to lowercase, stop word removal, and creating new columns for useful information.</li>
                    <li><strong>Sentiment Analysis:</strong> Conduct sentiment analysis to evaluate whether reviews are positive, 
                    neutral, or negative regarding the medications. This can help identify the overall consumer perception.</li>
                    <li><strong>Information Extraction:</strong> Extract key information regarding medication side effects, benefits, 
                    drawbacks, and other relevant details.</li>
                    <li><strong>Data Visualization:</strong> Create visualizations to illustrate and communicate effectively the 
                    results of the analysis.</li>
                </ul>
                """, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["The data", "Treatment Distribution", "Deases Distribution"])


# Creating the figures from the data

with tab1:
    st.subheader("About the dataset under study:")

    st.markdown(f"""<p class="paragraph"> The dataset provides insights into patients' experiences with different treatments  
                    for various diseases. Each entry in the dataset represents an individual patient's feedback and includes several 
                    fields. The `medication` column describes the medication prescribed along with the disease it aims to treat. 
                    The `rate` column gives a numerical rating for the effectiveness of the treatment, although some entries 
                    may have missing ratings. The `comment` section captures detailed patient experiences and may include mention 
                    of side effects or other observations. Additionally, each treatment is categorized under a specific 
                    `Treatment name` and `Treatment code`, which are helpful for identifying medications that share the 
                    same active ingredient. Lastly, the `Disease` column specifies the condition that the treatment is intended for, 
                    such as Crohn's Disease or Rheumatoid Arthritis. This dataset is a valuable resource for analyzing the efficacy 
                    and side effects of various treatments across different diseases.</p>""", unsafe_allow_html=True)
    with st.expander("See the data"):
        st.dataframe(data)

with tab2:
    fig1 = plt.figure(figsize=(16, 8))
    # Treatment Distribution
    sns.countplot(data=data, x='Treatment name', order=data['Treatment name'].value_counts().index)
    plt.title('Treatment Distribution')
    plt.xlabel('')
    plt.ylabel('Count')
    st.pyplot(fig1)

with tab3:
    fig2 = plt.figure(figsize=(16, 8))
    # Disease Distribution
    sns.countplot(data=data, x='Disease', order=data['Disease'].value_counts().index)
    plt.title('Disease Distribution')
    plt.xlabel('')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig2)

