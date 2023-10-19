import pickle
from config import Base
from pyLDAvis import gensim as genset
import pyLDAvis
import os
import gensim.corpora as corpora
from nltk.corpus import stopwords
import pandas as pd
from data_processing.preprocessing import process_dataframe
import nltk
import gensim
from gensim.utils import simple_preprocess
import nltk
from utils import sent_to_words, remove_stopwords, replace_side_effect
nltk.download('stopwords')


config = Base
config.ROOT_DATA_PATH = 'data/'
df = pd.read_csv(os.path(config.ROOT_DATA_PATH, 'raw_data_healthcare.csv'))

papers = process_dataframe(df)

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'since'])


# Apply the function to the 'cleaned_comment' column
papers['cleaned_comment'] = papers['cleaned_comment'].apply(
    replace_side_effect)

data = papers.cleaned_comment.values.tolist()
data_words = list(sent_to_words(data))

# remove stop words
data_words = remove_stopwords(data_words)

id2word = corpora.Dictionary(data_words)

words_to_remove = ['treatment_name',
                   'im', 'years', 'would', 'get',
                   'year', 'two', 'try', 'dont', 'still',
                   'like', 'weeks', 'start',
                   'take', 'months', 'month', 'back', 'first', 'ago']

# Filter out the words to be removed from the dictionary
id2word.filter_tokens(bad_ids=[id2word.token2id[word]
                      for word in words_to_remove])

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# number of topics
num_topics = 15

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
doc_lda = lda_model[corpus]

# import pyLDAvis.gensim_models as gensimvis

path_output = config.ROOT_OUPUT
LDAvis_data_filepath = os.path.join(path_output, 'LDA', str(num_topics))

if 1 == 1:
    LDAvis_prepared = genset.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)

pyLDAvis.save_html(LDAvis_prepared, path_output +
                   'LDA' + str(num_topics) + '.html')
