# Import libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()


# Tokenization (a list of tokens), will be used as the analyzer
# 1.Punctuations are [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]
# 2.Stop words in natural language processing, are useless words (data).

def main():
    print("Hello World")
    train()
    process_user_CSV()


def train():
    # Load the data
    # from google.colab import files # Use to load data on Google Colab
    # uploaded = files.upload() # Use to load data on Google Colab

    print("Begin Training")
    df = pd.read_csv('emails.csv')
    print(df.head(5))

    # Print the shape (Get the number of rows and cols)
    print(df.shape)

    # Get the column names
    print(df.columns)

    # Show the new shape (number of rows & columns)
    print(df.shape)

    # Show the number of missing (NAN, NaN, na) data for each column
    print(df.isnull().sum())

    # Need to download stopwords
    nltk.download('stopwords')

    process_text(df)

    # Show the Tokenization (a list of tokens )
    df['Text'].head().apply(process_text)

    from sklearn.feature_extraction.text import CountVectorizer

    df['full text'] = df['Text'] + df['From'] + df['Subject']

    messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['full text'])

    # Split data into 80% training & 20% testing data sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['Spam'], test_size=0.20, random_state=0)

    # Get the shape of messages_bow
    print(messages_bow.shape)

    classifier.fit(X_train, y_train)

    # Print the predictions
    print(classifier.predict(X_train))
    # Print the actual values
    print(y_train.values)

    # Evaluate the model on the training data set
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    pred = classifier.predict(X_train)
    print(classification_report(y_train, pred))
    print('Confusion Matrix: \n', confusion_matrix(y_train, pred))
    print()
    print('Accuracy: ', accuracy_score(y_train, pred))

    # Print the predictions
    print('Predicted value: ', classifier.predict(X_test))
    # Print Actual Label
    print('Actual value: ', y_test.values)

    # Evaluate the model on the test data set
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    pred = classifier.predict(X_test)
    print(classification_report(y_test, pred))
    print('Confusion Matrix: \n', confusion_matrix(y_test, pred))
    print()
    print('Accuracy: ', accuracy_score(y_test, pred))


def process_text(text):
    # 1 Remove Punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    # 2 Remove Stop Words
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    # 3 Return a list of clean words
    return clean_words


def process_user_CSV():
    # output for ease of testing
    print("Begin User Data Processing")

    # proecess User CSV file into dataframe
    df = pd.read_csv('emails.csv')

    print(df.Spam.value_counts())

    print(df.head(5))

    # Print the shape (Get the number of rows and cols)
    print(df.shape)

    df['full text'] = df['Text'] + df['From'] + df['Subject']

    messages = CountVectorizer(analyzer=process_text).fit_transform(df['full text'])

    spam_check = classifier.predict(messages)

    df["spam_new"] = spam_check

    print(df)

    df.to_csv('output.csv')


main()