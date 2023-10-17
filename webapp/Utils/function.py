import string
import re

def process_dataframe(df):
    """
    Custom made function to extract the treatment and the deases
    """
    if 'comment' not in df.columns:
        raise ValueError("DataFrame must contain a column named 'comment'")

    # Remove punctuations from the 'comment' column
    punctuations = string.punctuation
    df['medication'] = df['medication'].str.replace(f"[{re.escape(punctuations)}]", "", regex=True)

    # Extracting treatment name, treatment code, and disease name
    # Adjust the pattern to match unpunctuated strings
    pattern = r'(?P<treatment_name>.+?) (?P<treatment_code>.+?) for (?P<disease_name>.+?)( Maintenance)?$'

    extracted_data = df['medication'].str.extract(pattern)

    # Adding new columns to the dataframe
    df['Treatment name'] = extracted_data['treatment_name']
    df['Treatment code'] = extracted_data['treatment_code']
    df['Disease'] = extracted_data['disease_name']

    # Replacing the values in the 'comment' column
    df['comment'] = df['comment'].replace(to_replace=extracted_data['treatment_name'].tolist(), value="Treatment", regex=True)
    df['comment'] = df['comment'].replace(to_replace=extracted_data['treatment_code'].tolist(), value="Treatment Code", regex=True)
    df['comment'] = df['comment'].replace(to_replace=extracted_data['disease_name'].tolist(), value="Disease", regex=True)

    return df