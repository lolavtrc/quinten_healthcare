import streamlit as st
import pandas as pd 
import numpy as np 
from utils.plotting import plot_treatments_pie_chart,plot_rates_histograms,plot_most_pair_of_words

from utils.styles import title_style, paragraph_style
from utils.function import process_dataframe

# Page Configuration

st.set_page_config(page_title="Quinten DataScience", layout="wide")

data_raw = pd.read_csv("data/raw_data_healthcare.csv").drop("text_index",axis=1)
data = process_dataframe(data_raw)

# Top of the Page

st.markdown(title_style, unsafe_allow_html=True)
st.markdown(paragraph_style, unsafe_allow_html=True)
st.markdown('<h1 class="title">Quinten DataScience</h1>', unsafe_allow_html=True)

st.subheader("Introduction:")

st.markdown(f"""<p class="paragraph"> This project aims to extract essential insights from a dataset of consumer 
                reviews about medications used to treat intestinal diseases, such as Crohn's disease and colitis, 
                which often require long-term medication. The goal is to assist a pharmaceutical company in understanding 
                how its products are perceived by consumers and in identifying potential side effects.</p>""", unsafe_allow_html=True)

# Project Presentation

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

tab1, tab2, tab3, tab4 = st.tabs(["The data","Treatment Distribution", "Rating Distribution", "Words Pairs Analysis"])


# Creating the figures from the data

with tab1:
    st.subheader("About the dataset under study:")
    st.markdown("<hr style='height:15px; border:0px;'>",unsafe_allow_html=True)
    #Add few data points

    col1,col2,col3,col4,col5 = st.columns([2,3,1,3,2])
    with col2:
        st.metric(label="Number of deases treated", value=data["Disease"].nunique())
        st.metric(label="Number of treatment", value=data["Treatment name"].nunique())
    with col4:
        st.metric(label="Total number of comments", value=len(data) )
        st.metric(label="Avergage rating", value=np.round(data["rate"].mean(),2))

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
        # Selectors
        col1,col2,col3= st.columns(3)
        rating_value = col1.slider("Rating Value", min_value=None, max_value=10, step=1, key="Rating1")
        dease_type = col2.multiselect("Deases", 
                                    data["Disease"].unique(),
                                    key="Deases1",
                                    default=data["Disease"].unique())
        treatment_type = col3.multiselect("Treaments", 
                                    data["Treatment name"].unique(),
                                    key="Treatment1",
                                    default=data["Treatment name"].unique())

        st.markdown("<hr style='height:15px; border:0px;'>",unsafe_allow_html=True)

        st.dataframe(data[(data["Disease"].isin(dease_type)) & 
                          (data["Treatment name"].isin(treatment_type)) &
                          (data["rate"] > rating_value)]['comment'],
                    use_container_width=True)

with tab2:
    col1,col2,col3= st.columns(3)
    rating_value = col2.slider("Rating Value", min_value=0, max_value=10, step=1,key="Rating2")

    st.markdown("<hr style='height:15px; border:0px;'>",unsafe_allow_html=True)
    
    fig_treatment = plot_treatments_pie_chart(data,rate_filter= True if rating_value>0 else False ,rate_value=rating_value )
    st.pyplot(fig_treatment)

with tab3:
    col1,col2,col3= st.columns([4,1,4])
    dease_type = col1.multiselect("Deases", data["Disease"].unique(),key="Deases3")
    treatment_type = col3.multiselect("Treaments", data["Treatment name"].unique(),key="Treatment2")

    st.markdown("<hr style='height:15px; border:0px;'>",unsafe_allow_html=True)
    
    fig_rating = plot_rates_histograms(data[(data["Disease"].isin(dease_type)) & (data["Treatment name"].isin(treatment_type))],
                                       treatment_filter=False)
    st.pyplot(fig_rating)

with tab4:
    col1,col2,col3,col4= st.columns(4)
    dease_type = col1.multiselect("Deases", data["Disease"].unique(),key="Deases4")
    treatment_type = col2.multiselect("Treaments", data["Treatment name"].unique(), key="Treatment3")
    rating_value = col3.slider("Rating Value", min_value=0, max_value=10, step=1, key="Rating3")
    nb_inputs = col4.number_input("Number of pair", step=1, value=20)
    
    fig_pairs = plot_most_pair_of_words(data[(data["Disease"].isin(dease_type))& \
                                             (data["Treatment name"].isin(treatment_type))& \
                                             (data["rate"]>rating_value)],
                                        n=nb_inputs)

    st.pyplot(fig_pairs)



