import re
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


# Download necessary NLTK data files
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Initialize resources
s = set(stopwords.words('english'))
s.update(['nan', 'deleted', 'removed', "edit"])
stemmer = PorterStemmer()

# Emoji replacements
emoji_replacements = {
    '🚀': ' moon ',
    '💎🙌': ' hold ',
    '🧻🙌': ' sell ',
    '🐂': ' bull ',
    '🐻': ' bear ',
    '📈': ' bull ',
    '📉': ' bear ',
    '🦍': ' hold ',
    '💰': ' profit ',
    '🤡': ' fool ',
    '🔥': ' hot ',
    '😭': ' cry ',
    '😎': ' confident ',
    '😂': ' laugh ',
    '🍿': ' watch ',
    '🙈': ' ignore ',
    '👀': ' watch ',
    '💀': ' loss ',
    '🎢': ' volatile ',
    '🏴‍☠️': ' risk '
}

# Compile regex patterns once
emoji_pattern = re.compile('|'.join(re.escape(key) for key in emoji_replacements.keys()))
url_pattern = re.compile(r'https?://\S+|www\.\S+')
html_pattern = re.compile(r'&[^;]+;')
email_pattern = re.compile(r'\S*@\S*')
bracket_pattern = re.compile(r'\[.*?\]')
symbol_pattern = re.compile(r'[^a-z\s]')

def replace_emoji(match):
    return emoji_replacements[match.group(0)]

def preprocess_text(text, valid_tickers=[]):
    # Replace emojis
    text = re.sub(emoji_pattern, replace_emoji, text)

    # Convert to lower case
    text = text.lower()
    
    # Remove URLs, HTML entities, emails, and bracketed content
    text = re.sub(url_pattern, ' ', text)
    text = re.sub(html_pattern, ' ', text)
    text = re.sub(email_pattern, ' ', text)
    text = re.sub(bracket_pattern, ' ', text)

    # Remove ticker symbols
    if valid_tickers:
        ticker_pattern = re.compile(r'(?<!\w)(\$)?(' + '|'.join(re.escape(ticker.lower()) for ticker in valid_tickers) + r')(?!\w)')
        text = re.sub(ticker_pattern, ' ', text)
    
    # Remove symbols, keeping letters, numbers, and spaces
    text = re.sub(symbol_pattern, ' ', text)
    
    # Removing extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in s)

    # Tokenize and stem each word
    text = word_tokenize(text)
    text = ' '.join(stemmer.stem(word) for word in text)
    #text = ' '.join(word for word in text)

    return text

def preprocess_series(series, valid_tickers=[]):
    return series.apply(lambda text: preprocess_text(text, valid_tickers))

def get_top_n_words(corpus, n=None):
        '''
        Function to return a list of most frequent unigrams in documents
        @param corpus: (str or pandas.dataframe) documents 
        @param n: (int) number of most frequent unigrams
        @return: (list) most frequent unigrams
        '''
        vec = CountVectorizer().fit(corpus)             # bag of words
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)  
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]