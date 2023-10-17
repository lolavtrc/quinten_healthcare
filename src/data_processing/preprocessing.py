import re
import nltk
import string
import pandas as pd
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def clean_comment(row):
    # Lower comments
    comment = row['comment'].lower() 
    
    # Delete ponctuation
    comment = ''.join([char for char in comment if char not in string.punctuation]) 
    words = comment.split() 
    
    # Delete stop words (english)
    stop_words = set(stopwords.words('english')) 
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]

    cleaned_comment = ' '.join(words) 
    return cleaned_comment

def process_dataframe(df):

    df = df.copy()

    if 'comment' not in df.columns:
        raise ValueError("DataFrame must contain a column named 'comment'")
    punctuations = string.punctuation
    df['medication'] = df['medication'].str.replace(f"[{re.escape(punctuations)}]", "", regex=True)
    df['medication'] = df['medication'].str.lower()

    df['cleaned_comment'] = df.apply(clean_comment, axis=1)

    # Extracting treatment name, treatment code, and disease name
    pattern = r'(?P<treatment_name>.+?) (?P<treatment_code>.+?) for (?P<disease_name>.+?)( Maintenance)?$'
    extracted_data = df['medication'].str.extract(pattern)
    df['Treatment name'] = extracted_data['treatment_name']
    df['Treatment code'] = extracted_data['treatment_code']
    df['Disease'] = extracted_data['disease_name']

    # Replace disease, treatment code and treatment name per generic formulas
    replacement_dict = {
        # Disease replacements 
        'crohns disease': 'disease',
        'rheumatoid arthritis': 'disease',
        'ulcerative colitis': 'disease',
        'crohns disease maintenance': 'disease',
        'psoriatic arthritis': 'disease',
        'ankylosing spondylitis': 'disease',
        'ankylose spondylitis': 'disease',
        'ulcerative colitis maintenance': 'disease',
        'psoriasis': 'disease',
        'Psoriasis': 'disease',
        'crohn': 'disease',
        "chron's": 'disease',
        'crohns': 'disease',
        # Treatment code replacements
        'infliximab': 'treatment_code',
        'adalimumab': 'treatment_code',
        'certolizumab': 'treatment_code',
        'golimumab': 'treatment_code',
        'aria golimumab': 'treatment_code',
        'vedolizumab': 'treatment_code',
        'ustekinumab': 'treatment_code',
        'natalizumab': 'treatment_code',
        # Treatment name replacements
        'inflectra': 'treatment_name',
        'remicade': 'treatment_name',
        'renflexis': 'treatment_name',
        'humira': 'treatment_name',
        'cimzia': 'treatment_name',
        'simponi': 'treatment_name',
        'entyvio': 'treatment_name',
        'stelara': 'treatment_name',
        'tysabri': 'treatment_name'
    }
    
    # Appliquez les remplacements dans la colonne 'cleaned_comment' en utilisant le dictionnaire
    df['cleaned_comment'] = df['cleaned_comment'].replace(replacement_dict, regex=True)

    return df