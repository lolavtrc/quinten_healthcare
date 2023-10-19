from textblob import TextBlob
import plotly.express as px
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk import pos_tag

# The two functions takes as input the csv located in data/sorted_feelings.csv

##################################################################################################
# First Plot : Most represented topics 
##################################################################################################
def create_tree_map(df):
    """
    Create a treemap visualization to represent the sentiments in the provided DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing sentiment data.

    Returns:
    --------
    None

    Notes:
    ------
    - Calculates the total weight of sentiments in the DataFrame.
    - Removes unuseful or irrelevant sentiments based on predefined list.
    - Sorts the DataFrame by weight in descending order.
    - Computes the percentage of each sentiment based on its weight.
    - Generates a treemap using Plotly Express to visualize the sentiment distribution.
    """
    df = df.copy()
    total_weight = df['weight'].sum()
    # Delete unusefull feelings
    sentiments_to_delete = ["psoriasis", "methotrexate", "nothing", "week", "lot", "day", "dr", "month", "im", "everything", "ms",
                            "something", "anything", "anyone", "prednisone", "work", "thing", "difference", "arthritis", "way", "rheumatologist",
                            "one", "immune"]
    df = df[df['weight']>15]
    df = df[~df['sentiment'].isin(sentiments_to_delete)]
    df = df.sort_values(by='weight', ascending=False)
    df['percentage'] = (df['weight'] / total_weight) * 100
    fig = px.treemap(df, path=['sentiment'], values='percentage', color_continuous_scale='rainbow')
    fig.update_traces(textinfo='label+percent entry')
    fig.show()

##################################################################################################
# Second Plot : Feelings associated to most represented topics
# The plot function plotting is show_treatment_treemap
# Possible filters : topics = ['treatment', 'pain', 'body', 'skin', 'fatigue', 'hair', 'sideeffect', 'weight', 'energy', 'legs']
##################################################################################################
nltk.download('sentiwordnet')

def tag_words_with_nltk(words):
    """
    Tokenize and tag words using NLTK's part-of-speech tagging.

    Parameters:
    -----------
    words : str
        A string containing space-separated words.

    Returns:
    --------
    list of (str, str)
        A list of word-tag pairs.

    Notes:
    ------
    - Tokenizes the input text into words.
    - Tags each word with its part of speech (e.g., noun, adjective).
    """
    word_tokens = word_tokenize(words)
    tagged_words = pos_tag(word_tokens)
    word_tag_list = [[word, tag] for word, tag in tagged_words]
    return word_tag_list

def filter_nouns_adjectives(word_tag_list):
    """
    Filter nouns and adjectives from a list of word-tag pairs.

    Parameters:
    -----------
    word_tag_list : list of (str, str)
        A list of word-tag pairs.

    Returns:
    --------
    list of str
        A list of filtered nouns and adjectives.

    Notes:
    ------
    - Filters words from the input list based on their part-of-speech tags (Nouns and Adjectives).
    """
    filtered_nouns = [word for word, tag in word_tag_list if tag.startswith('N')]
    filtered_adjectives = [word for word, tag in word_tag_list if tag.startswith('J')]
    combined = filtered_nouns + filtered_adjectives
    return combined

def classify_feelings(word_list):
    """
    Classify words into positive, negative, or other feelings based on sentiment analysis.

    Parameters:
    -----------
    word_list : list of str
        A list of words to classify.

    Returns:
    --------
    list of str, list of str, list of str
        Lists of positive, negative, and other feelings.

    Notes:
    ------
    - Classifies words into positive and negative feelings based on sentiment analysis using TextBlob.
    - Words with positive sentiment are categorized as positive feelings.
    - Words with negative sentiment are categorized as negative feelings.
    - Words with neutral sentiment are categorized as other feelings.
    - Uses predefined lists for negative and positive feelings.
    """
    positive_feelings = []
    negative_feelings = []
    other_feelings = []

    # Possible values for sentiments : 
    neg_feelings = ['hairloss', 'headache', 'weightgain', 'treatment', 'thinner', 'gain', 'vomit',  'disappoint', 'nausea', 'cramp', 'regain', 
                    'loss', 'pain', 'body', 'skin', 'fatigue', 'hair', 'sideeffect', 'weight', 'energy', 
                    'diarrhea', 'legs', 'nausea', 'stomachpain', 'infections', 'sideeffects', 'constipation', 'ribpain', 'lesions', 'worsen']
    pos_feelings = ['improvement', 'eyebrow', 'stable', 'increase', 'help'] 

    for word in word_list:
        # Vérifiez si le mot est dans les listes neg_feelings ou pos_feelings
        if word in neg_feelings:
            negative_feelings.append(word)
        elif word in pos_feelings:
            positive_feelings.append(word)
        else:
            analysis = TextBlob(word)
            sentiment = analysis.sentiment.polarity

            if sentiment > 0:
                positive_feelings.append(word)
            elif sentiment < 0:
                negative_feelings.append(word)
            else:
                other_feelings.append(word)

    return positive_feelings, negative_feelings, other_feelings

def generate_associated_feelings(df):
    """
    Generate associated feelings for selected sentiments and create a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing sentiment data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing associated feelings for selected sentiments.

    Notes:
    ------
    - Filters the input DataFrame to select specific sentiments.
    - Tags words with their part of speech, filters nouns and adjectives.
    - Classifies words into positive, negative, or other feelings.
    """
    df = df.copy()
    sentiment_to_extract = ['treatment', 'pain', 'body', 'skin', 'sideeffect', 'fatigue', 'hair', 'weight', 'energy', 'legs', 'diarrhea']
    df_sorted_sentiments_selected = df[df['sentiment'].isin(sentiment_to_extract)]
    df_sorted_sentiments_selected['tagged words'] = df_sorted_sentiments_selected['associated words'].apply(lambda x: tag_words_with_nltk(x))
    df_sorted_sentiments_selected['nouns and adjectives'] = df_sorted_sentiments_selected['tagged words'].apply(lambda x: filter_nouns_adjectives(x))
    words_to_delete = ['live', 'due', 'second', 'god', 'bring', 'normal', 'fit', 'periods', 'able', 'much', 'clear', 'skip', 'needle', 'free', 'overall', 'normal', 'deny',
                   'return', 'lose', 'day', 'heat', 'eliminate', 'cover', 'diseases', 'mean', 'medicine', 'try', 'turn', 'life', 'star', 'report', 'decision', 'hip', 'work',
                   'use', 'wear', 'fistulas', 'thing', 'load', 'cause', 'refuse', 'bar', 'burn', 'start', 'pills', 'start', 'slight', 'cause', 'experience',
                   'notice', 'possible', 'push', 'hit', 'arianinfusions', "choose", "age", "way", "walk", "postinfusion", "wasnt", "years", "pilate"]
    for word in words_to_delete:
        df_sorted_sentiments_selected['nouns and adjectives'] = df_sorted_sentiments_selected['nouns and adjectives'].apply(lambda x: [item for item in x if item != word])
    df_sorted_sentiments_selected['Positive feelings'], df_sorted_sentiments_selected['Negative feelings'], df_sorted_sentiments_selected['Other feelings'] = zip(*df_sorted_sentiments_selected['nouns and adjectives'].apply(lambda x: classify_feelings(x)))

    return df_sorted_sentiments_selected

def show_treatment_treemap(df, sentiment_filter='treatment'):
    """
    Generate and display a treemap of associated feelings for a specific sentiment.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing associated feelings data.

    sentiment_filter : str, optional
        The sentiment to filter and display associated feelings for. Default is 'treatment'.

    Returns:
    --------
    None

    Notes:
    ------
    - Generates a treemap to visualize associated feelings for a specific sentiment.
    - Filters and selects the associated feelings for the chosen sentiment.
    """
    df = df.copy()
    df = generate_associated_feelings(df)
    df_treatment = df[df['sentiment'] == sentiment_filter].copy()

    positive_feelings = df_treatment['Positive feelings'].values[0]
    negative_feelings = df_treatment['Negative feelings'].values[0]

    positive_feelings = positive_feelings[:80] if len(positive_feelings) > 80 else positive_feelings
    negative_feelings = negative_feelings[:80] if len(negative_feelings) > 80 else negative_feelings

    treemap_data = pd.DataFrame({
        'Category': ['Positive feelings', 'Negative feelings'],
        'Sentiments': [
            ", ".join(positive_feelings),
            ", ".join(negative_feelings),
        ],
        'Size': [len(positive_feelings), len(negative_feelings)]
    })

    fig = px.treemap(
        treemap_data,
        path=['Category', 'Sentiments'],
        values='Size',
        title=f"Treemap of Associated Feelings for '{sentiment_filter}'",  
    )
    fig.update_layout(
        title_text=f"Treemap of Associated Feelings for '{sentiment_filter}'", 
        title_font=dict(size=20),  
        font=dict(size=15),  
        width=800,  
        height=500  
    )

    fig.update_traces(textinfo='label+value')
    fig.show()
