import nltk
import string
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

##################################################################################################
# First plot in EDA Section : the Pie chart of treatments
# Filter: By rating
##################################################################################################

def generate_treatments_pie_chart_dataframe(df, rate_filter=False, rate_value=1):
    """
    Generate a DataFrame for pie chart visualization of treatments and their counts.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing treatment data.

    rate_filter : bool, optional
        If True, filter the DataFrame based on a specific 'rate' value. Default is False.

    rate_value : int, optional
        The value to filter the DataFrame by 'rate' if rate_filter is set to True. Default is 1.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns 'Treatment name', 'Rate' (if rate_filter is True), 'count', and 'percentage'.

    Notes:
    ------
    - The function drops unnecessary columns from the input DataFrame.
    - If rate_filter is False, it calculates counts and percentages for all treatments.
    - If rate_filter is True, it filters the DataFrame by rate_value and calculates counts and percentages.
    """
    df = df.copy()
    columns_to_drop = ['text_index', 'medication', 'comment', 'cleaned_comment', 'Treatment code', 'Disease']
    df.drop(columns=columns_to_drop, inplace=True)
    
    if rate_filter is False :
        group_by_treatment = df.groupby('Treatment name').size()
        group_by_treatment_percent = (group_by_treatment / group_by_treatment.sum()) * 100
        result_df = pd.DataFrame({'Treatment name': group_by_treatment.index,
                                  'count': group_by_treatment.values,
                                  'percentage': np.round(group_by_treatment_percent.values, 1)})
        result_df['Treatment name'] = result_df['Treatment name'].str.title()

    if rate_filter :
        df = df[df['rate']==rate_value].copy()
        group_by_treatment = df.groupby(['Treatment name', 'rate']).size()
        group_by_treatment_percent = (group_by_treatment / group_by_treatment.sum()) * 100
        result_df = pd.DataFrame({'Treatment name': group_by_treatment.index.get_level_values('Treatment name'),
                                'Rate': group_by_treatment.index.get_level_values('rate'),
                                'count': group_by_treatment.values,
                                'percentage': np.round(group_by_treatment_percent.values, 1)})
        result_df['Treatment name'] = result_df['Treatment name'].str.title()

    return result_df

def plot_treatments_pie_chart(df, 
                            rate_filter=False, 
                            rate_value=3):
    """
    Plot a pie chart to visualize the distribution of treatments in the provided DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing treatment data.

    rate_filter : bool, optional
        If True, filter the DataFrame based on a specific 'rate' value. Default is False.

    rate_value : int, optional
        The value to filter the DataFrame by 'rate' if rate_filter is set to True. Default is 3.

    Returns:
    --------
    None (plt.show)

    Notes:
    ------
    - Calls the 'generate_treatments_pie_chart_dataframe' function to create the necessary data.
    - Uses a pie chart to visualize the treatment distribution.
    - If rate_filter is True, the chart will be labeled with the specified rate value.
    """
    df_to_plot = generate_treatments_pie_chart_dataframe(df, 
                                        rate_filter=rate_filter, 
                                        rate_value=rate_value)
    
    colors = plt.cm.rainbow(df_to_plot.index / float(len(df_to_plot)))
    plt.figure(figsize=(8, 8))
    pie, texts, autotexts = plt.pie(df_to_plot['percentage'], labels=None, colors=colors, autopct='%1.1f%%', startangle=140)
    title = 'Treatment distribution'

    # Display the filter
    if rate_filter:
        title += f' for Rate value : {rate_value}'

    plt.title(title, fontsize=16) 
    legend_labels = df_to_plot['Treatment name']
    plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.2, 1), fontsize = 11)
    plt.axis('equal')  
    plt.show()
    
  
##################################################################################################
# Second plot : Pie chart of ratings
# Filter: By treatment, By sickness
##################################################################################################

def generate_rating_charts(df, treatment_filter=False, treatment_value='inflectra'):
    """
    Filter and preprocess the DataFrame for generating rating-related charts or visualizations.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing rating data.

    treatment_filter : bool, optional
        If True, filter the DataFrame based on a specific 'Treatment name'. Default is False.

    treatment_value : str, optional
        The treatment name to filter the DataFrame if treatment_filter is set to True. Default is 'inflectra'.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with selected columns after applying any optional filters.

    Notes:
    ------
    - The function drops unnecessary columns from the input DataFrame.
    - If treatment_filter is True, it filters the DataFrame by the specified treatment name.
    """
    df = df.copy()
    columns_to_drop = ['text_index', 'medication', 'comment', 'cleaned_comment', 'Treatment code', 'Disease']
    df.drop(columns=columns_to_drop, inplace=True)
    
    if treatment_filter :
        df = df[df['Treatment name']==treatment_value].copy()

    return df

def plot_rates_histograms(df, 
                        treatment_filter=False, 
                        treatment_value='inflectra'):
    """
    Generate a histogram to visualize the distribution of ratings in the provided DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing rating data.

    treatment_filter : bool, optional
        If True, filter the DataFrame based on a specific 'Treatment name'. Default is False.

    treatment_value : str, optional
        The treatment name to filter the DataFrame if treatment_filter is set to True. Default is 'inflectra'.

    Returns:
    --------
    None

    Notes:
    ------
    - Calls the 'generate_rating_charts' function to preprocess the data.
    - Creates a histogram to visualize the distribution of ratings.
    - If treatment_filter is True, the chart will be labeled with the specified treatment name.
    """
    df_to_plot = generate_rating_charts(df, 
                                        treatment_filter=treatment_filter, 
                                        treatment_value=treatment_value)
    plt.figure(figsize=(6, 4))
    plt.hist(df_to_plot['rate'], bins=10, color='skyblue', edgecolor='black', alpha=0.7, )
    
    average_rating = df_to_plot['rate'].mean()
    plt.text(0.05, 0.9, f'Mean Rating: {average_rating:.2f}', transform=plt.gca().transAxes, fontsize=12)

    title = 'Distribution of Ratings'
    
    # Display the filter
    if treatment_filter:
        title += f' for Treatment: {treatment_value}'
    
    plt.title(title, fontsize=16) 
    plt.xlabel('Rating', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

##################################################################################################
# Third plot : Most Common group of words 
##################################################################################################

def word_tokenization(row):
    """
    Tokenize a cleaned comment into a list of words.

    Parameters:
    -----------
    row : pd.Series
        A row from the DataFrame containing a 'cleaned_comment' column.

    Returns:
    --------
    list of str
        A list of words extracted from the 'cleaned_comment' in lowercase.

    Notes:
    ------
    - Uses NLTK for word tokenization.
    """
    comment = row['cleaned_comment'].lower()
    words = nltk.word_tokenize(comment)  # Tokenization
    return words

def create_word_pairs(row):
    """
    Create word pairs from a list of tokenized words in a row.

    Parameters:
    -----------
    row : pd.Series
        A row from the DataFrame containing a 'tokenized_comment' column.

    Returns:
    --------
    list of str
        A list of word pairs created from the 'tokenized_comment'.

    Notes:
    ------
    - Creates word pairs by combining adjacent words in the tokenized comment.
    """
    tokenized_comment = row['tokenized_comment']
    word_pairs = [" ".join([tokenized_comment[i], tokenized_comment[i+1]]) for i in range(len(tokenized_comment) - 1)]
    return word_pairs

def generate_frequencies_dataframes(df):
    """
    Preprocess the DataFrame to include tokenized comments and word pairs.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing data for analysis.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with 'tokenized_comment' and 'word_pairs' columns.

    Notes:
    ------
    - Tokenizes the cleaned comments and creates word pairs for analysis.
    """
    df = df.copy()
    df['tokenized_comment'] = df.apply(word_tokenization, axis=1) 
    df['word_pairs'] = df.apply(create_word_pairs, axis=1) 
    return df

def plot_most_common_unique_words(df, n=20):
    """
    Plot a horizontal bar chart of the most common unique words in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing tokenized data.

    n : int, optional
        The number of top words to display in the chart. Default is 20.

    Returns:
    --------
    None

    Notes:
    ------
    - Generates a bar chart to visualize the frequency of the top unique words.
    - Filters out specific words and characters from the analysis.
    """
    df = df.copy()
    df = generate_frequencies_dataframes(df)
    all_words = [word for word_list in df['tokenized_comment'] for word in word_list]
    all_words = [word for word in all_words if word.isalpha()]
    delete_words = ['years', 'year', 'month', 'week', 'weeks', 'day', 'months', 'days', 'im', 
                    'since', 'like', 'every', 'time', 'back', 'ive', 'ago', 'take', 'go', 'start', ]
    all_words = [word for word in all_words if word not in delete_words]
    word_freq = nltk.FreqDist(all_words)
    words, frequencies = zip(*word_freq.items())
    sorted_data = sorted(zip(words, frequencies), key=lambda x: x[1], reverse=True)
    words, frequencies = zip(*sorted_data)
    top_words = words[:n]
    top_frequencies = frequencies[:n]
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    
    # Plot the horizontal bar chart with rainbow colors
    plt.figure(figsize=(8, 6))
    plt.barh(top_words, top_frequencies, color=colors)
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Words', fontsize=14)
    plt.title(f'Top {n} Most Common Words', fontsize=16)
    plt.show()

def plot_most_pair_of_words(df, n=20):
    """
    Plot a horizontal bar chart of the most common word pairs in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing word pairs.

    n : int, optional
        The number of top word pairs to display in the chart. Default is 20.

    Returns:
    --------
    None

    Notes:
    ------
    - Generates a bar chart to visualize the frequency of the top word pairs.
    - Filters out specific word pairs from the analysis.
    """
    df = df.copy()
    df = generate_frequencies_dataframes(df)

    def is_valid_word_pair(word_pair):
        for word in word_pair.split():
            if any(char.isdigit() or char in string.punctuation or char == "’" for char in word):
                return False
        return True

    all_word_pairs = [word_pair for word_pairs_list in df['word_pairs'] for word_pair in word_pairs_list if is_valid_word_pair(word_pair)]
    delete_words = ['week ago', 'month ago', 'two year', 'two week', 'disease disease', 
                    'year old', 'year ago', 'disease since']
    all_word_pairs = [word for word in all_word_pairs if word not in delete_words]
    
    if not all_word_pairs:
        print("No word pairs found in the dataset after filtering.")
        return

    word_freq = nltk.FreqDist(all_word_pairs)
    top_word_pairs = word_freq.most_common(n)
    top_words, top_frequencies = zip(*top_word_pairs)
    colors = plt.cm.rainbow(np.linspace(0, 1, n))
    
    # Plot the horizontal bar chart with rainbow colors
    plt.figure(figsize=(8, 6))
    plt.barh(top_words, top_frequencies, color=colors)
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Words', fontsize=14)
    plt.title(f'Top {n} Most Common Words', fontsize=16)
    plt.show()

