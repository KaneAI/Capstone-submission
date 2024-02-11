import pickle
import pandas as pd
import re
import nltk

# Ensure the necessary NLTK data packages are downloaded
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

def load_pickle(filename):
    # Open the file in read mode
    with open(filename, 'rb') as file:
        # Deserialize and load the object from the file
        loaded_object = pickle.load(file)
    return loaded_object

def standardise_text_series(text_series:pd.Series)->pd.Series:
    output_series = text_series
    for i in range(0, len(text_series)):
        output_series[i] = text_series[i].lower()
        output_series[i] = re.sub('[^a-zA-Z]', ' ', output_series[i])
        output_series[i] = re.sub(r'\s+', ' ', output_series[i])
        output_series[i] = output_series[i].rstrip(' ')
        output_series[i] = output_series[i].lstrip(' ')
    return pd.Series(output_series)

def get_wordnet_pos(treebank_tag):
    """Converts treebank tags to WordNet tags."""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN # Default to noun

def lemmatize_text(text:str)->str:
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]
        return ' '.join(lemmatized_tokens)

def lemmatize_text_series(text_series:pd.Series)->pd.Series:
    return text_series.apply(lemmatize_text)

def remove_stopwords(input_text:str)->str:
    output_list = [word for word in input_text.split() if not word in set(stopwords.words('english'))]
    output_text = ' '.join(output_list)
    return output_text

def remove_stopwords_series(text_series:pd.Series)->pd.Series:
    return text_series.apply(remove_stopwords)

user_pred = load_pickle('pickle//user_final_ratings.pkl')
product_lookup = load_pickle('pickle//product_lookup.pkl')
vectorizer = load_pickle('pickle//vectorizer.pkl')
sc = load_pickle('pickle//scaler.pkl')
model = load_pickle('pickle//sentiment_model.pkl')


def get_recommendations(username):
    # Get the top 20 recommendations from the previously calculated recommendations
    d = user_pred.loc[username].sort_values(ascending=False)[0:20]

    # Merge in the product names
    d = pd.merge(d,product_lookup,left_on='id',right_on='id', how = 'left')

    df = load_pickle('pickle//processed_reviews.pkl')
    
    # filter souce data for the 20 recommended products 
    # Creating a boolean series
    ids_in_d = df['id'].isin(d['id'])

    # Filtering df1
    df = df[ids_in_d]

    # reset the index after the deletion
    df.reset_index(drop=True, inplace=True)

    # Transform the 'review_s3' column
    tfidf_matrix = vectorizer.transform(df['review_s3'])

    # Convert the TF-IDF matrix to a DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Standardize the values
    tfidf_df = sc.transform(tfidf_df) 

    # Make the predictions
    preds = pd.DataFrame(model.predict(tfidf_df))

    # merge the predictions with the df
    df['sentiment'] = preds

    df = df[['id','sentiment']]

    sentiment_percent_df = df.groupby('id')['sentiment'].mean().reset_index()

    # select the top 5 and merge in the product names
    results = pd.merge(sentiment_percent_df,product_lookup,left_on='id',right_on='id', how = 'left')
    results = results.sort_values(by=['sentiment'],ascending=False)[0:5]

    return results
