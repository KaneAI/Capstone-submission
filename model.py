import pickle
import pandas as pd
import nltk

# Ensure the necessary NLTK data packages are downloaded
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize

def load_pickle(filename):
    # Open the file in read mode
    with open(filename, 'rb') as file:
        # Deserialize and load the object from the file
        loaded_object = pickle.load(file)
    return loaded_object

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
