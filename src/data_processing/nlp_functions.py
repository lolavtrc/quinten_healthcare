def train_nlp_model(df):
    """
    From a dataframe with the column of comments, we train a sentiment analysis
    pipeline from HuggingFace.
    Returns the model as well as the dataframe with 2 added columns:
    sentiment, the label of classification
    score, the associated probablity to the label
    """
    # Using Huggingface's pipeline for sentiment analysis
    sentiment_analyzer = pipeline("sentiment-analysis")

    # Perform sentiment analysis and add results to the DataFrame
    sentiment_results = sentiment_analyzer(df["comment"].tolist())

    # Extract sentiment labels and scores
    sentiment_labels = [entry['label'] for entry in sentiment_results]
    sentiment_scores = [entry['score'] for entry in sentiment_results]

    # Add the sentiment label and score as new columns to the DataFrame
    df['sentiment'] = sentiment_labels
    df['sentiment_score'] = sentiment_scores

    return sentiment_analyzer, df

def plot_sentiment_distribution(csv_file):
    """
    Plot a pie chart to visualize the distribution of sentiment values in a CSV file.

    Parameters:
    csv_file (str): Path to the CSV file containing the data. The CSV file should have a 'sentiment' column.

    Returns:
    None

    Example:
    plot_sentiment_distribution('your_data.csv')
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if 'sentiment' column exists in the DataFrame
        if 'sentiment' not in df.columns:
            raise ValueError("The 'sentiment' column does not exist in the CSV file.")

        # Count the number of unique values in the 'sentiment' column
        sentiment_counts = df['sentiment'].value_counts()

        # Create a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(sentiment_counts))))

        # Add a title
        plt.title('Sentiment Distribution')

        # Display the pie chart
        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

class StemmedCountVectorizer(CountVectorizer):
    """
    A custom CountVectorizer that stems words using an English Snowball stemmer during tokenization.

    This class inherits from the CountVectorizer class provided by scikit-learn and overrides the
    `build_analyzer` method to apply stemming to the tokenization process.

    Parameters:
    -----------
    CountVectorizer : Class
        The base class from scikit-learn that provides text vectorization capabilities.

    Attributes:
    -----------
    None

    Methods:
    --------
    build_analyzer(self)
        Override the build_analyzer method of the parent class to create a custom analyzer
        that tokenizes text and stems each word using an English Snowball stemmer.
    """

    def build_analyzer(self):
        """
        Build a custom analyzer for text tokenization and stemming.

        Returns:
        --------
        callable
            A function that takes a document (text) as input and returns a list of stemmed words.

        Example:
        --------
        vectorizer = StemmedCountVectorizer()
        analyzer = vectorizer.build_analyzer()
        tokens = analyzer("Running runs in the runner's shoes.")
        # Result: ['run', 'run', 'runner', 'shoe']
        """
        # Get the default analyzer from the parent class (CountVectorizer)
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        # Create a lambda function that tokenizes and stems words using an English Snowball stemmer
        custom_analyzer = lambda doc: [english_stemmer.stem(w) for w in analyzer(doc)]

        return custom_analyzer
    
def train_bert(docs, model_path):
    """
    Train a BERTopic model for topic modeling on a corpus of text data and save the model to a specified path.

    Parameters:
    -----------
    docs : list of str
        A list of text documents (comments or text data) for which topics need to be extracted.

    model_path : str
        The file path where the trained BERTopic model will be saved.

    Returns:
    --------
    topic_model : BERTopic
        The trained BERTopic model capable of extracting topics from text data.

    Example:
    --------
    # Train a BERTopic model on a list of text documents
    documents = ["This is a sample document.", "Another example document.", ...]
    model_path = "bertopic_model.pkl"
    trained_model = train_bert(documents, model_path)
    """

    # Load the pre-trained sentence embedding model (MiniLM)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create a clustering model for topic extraction using HDBSCAN
    cluster_model = HDBSCAN(
        min_cluster_size=15,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # Create a ClassTFIDF transformer with BM25 weighting
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)

    # Create a custom vectorizer model (StemmedCountVectorizer) for text preprocessing
    vectorizer_model = StemmedCountVectorizer(
        analyzer="word",
        stop_words=stopwords,
        ngram_range=(1, 2)
    )

    # Create a BERTopic model by combining the embedding, clustering, and vectorizer models
    topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model,
        ctfidf_model=ctfidf_model,
        vectorizer_model=vectorizer_model,
        language="english"
    )

    # Fit the BERTopic model on the corpus of text documents
    topics, probs = topic_model.fit_transform(docs)

    # Save the trained BERTopic model to the specified file path
    topic_model.save(model_path)

    return topic_model, topics

def visualize_topic_model(topic_model, documents, top_n_topics=10):
    """
    Visualize various aspects of a trained topic model.

    Parameters:
    -----------
    topic_model : BERTopic instance
        A trained BERTopic model.

    documents : list of str
        The documents used to train the topic model.

    top_n_topics : int, optional
        The number of top topics to visualize in the barchart. Default is 10.
    """

    # Visualize the topics in a 2D space
    fig1 = topic_model.visualize_topics()
    fig1.show()

    # Visualize the top topics as a barchart
    fig2 = topic_model.visualize_barchart(top_n_topics=top_n_topics)
    fig2.show()

    # Visualize the documents
    fig3 = topic_model.visualize_documents(documents)
    fig3.show()

    # Visualize the topic hierarchy
    #fig4 = topic_model.visualize_hierarchy()
    #fig4.show()