from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import alpaca_api
import nltk
from nltk.corpus import stopwords
from preprocessing import preprocess_text, preprocess_series

class Sentiment:
    def __init__(self, test_size=0.2):
        nltk.download('stopwords')
        self.stopwords = stopwords.words('english')
        self.pipeline = make_pipeline(TfidfVectorizer(stop_words=self.stopwords), MultinomialNB())
        alpaca = alpaca_api.Alpaca()
        self.valid_tickers = alpaca.get_refined_ticker_list()
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.accuracy = None

    def train_model(self, data, threshold=0.5):
        self.threshold = threshold

        df = data[data['link_flair_text'].isin(['Gain', 'Loss'])]
        df['sentiment'] = df['link_flair_text'].map({'Gain': 1, 'Loss': 0})

        # Splitting the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df['processed'], df['sentiment'], test_size=self.test_size, random_state=42, stratify=df['sentiment'], shuffle = False
        )

        # # Splitting the dataset into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(
        #     df['processed'], df['sentiment'], test_size=0.2, random_state=42, shuffle = False
        # )

        # Compute class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_prior = class_weights / class_weights.sum()
        
        # Update MultinomialNB with class weights
        self.pipeline = make_pipeline(TfidfVectorizer(stop_words=self.stopwords), MultinomialNB(class_prior=class_prior))

        # Fit the pipeline to the training data
        self.pipeline.fit(self.X_train, self.y_train)
        y_probs, y_pred = self.predict(self.X_test)

        # Evaluation
        self.accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {self.accuracy}")
        print(classification_report(self.y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(self.y_test, y_probs)}")

    def predict(self, sen, preprocess = False):
        
        if not isinstance(sen, pd.Series):
            sen = pd.Series(sen)
        if preprocess:
            sen = preprocess_series(sen, valid_tickers=self.valid_tickers)

        # Get probabilities for each class
        y_probs = self.pipeline.predict_proba(sen)[:, 1]  # Probabilities of the positive class

        # Apply threshold to determine class predictions
        y_pred = (y_probs >= self.threshold).astype(int)

        return y_probs, y_pred

    def score(self, sen, preprocess = True):
        return self.predict(sen, preprocess = preprocess)[0]
        #return self.pipeline.predict_proba([preprocess_text(sen, valid_tickers=self.valid_tickers)])[0]
