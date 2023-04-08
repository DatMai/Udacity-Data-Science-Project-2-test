import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

import pickle
import warnings
warnings.filterwarnings('ignore') 

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("Disaster_Response", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y

def tokenize(text):   
    """
    Tokenize and preprocess a text message.
    Args:
    text: str. A text message.
    Returns:
    tokens: list of str. A list of tokens (words) after performing text preprocessing.
    """
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
     # Convert all text to lowercase
    text = text.lower()
    
    # Tokenize text into words
    tokens = word_tokenize(text)
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
        
    # specify parameters for grid search 
    parameters = {'clf__estimator__n_estimators' : [50, 100],
                'clf__estimator__max_depth': [None, 10, 20],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test):
    y_predict = model.predict(X_test)
    # Classification report
    print("CLASSIFICATION REPORT:")
    for i, column in enumerate(Y_test.columns):
        print(column)
        print(classification_report(Y_test[column], y_predict[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print("CHECKING MODEL", type(X_train),type(Y_train),X_train.dtype,Y_train.dtypes)
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()