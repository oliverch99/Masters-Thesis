import pandas as pd
import numpy as np
#import sentiment
import sentiment_test
import matplotlib.pyplot as plt
import pickle
import alpaca_api
import alpaca_trade_api as tradeapi
alpaca = alpaca_api.Alpaca()
from statsmodels.tsa.stattools import grangercausalitytests
from savenload import save_and_load_data
from datetime import timedelta
from wsb_reddit import get_tickers_from_title
from collections import Counter
from statsmodels.tsa.stattools import adfuller
import re
from preprocessing import preprocess_series
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from matplotlib.ticker import FuncFormatter


valid_tickers = alpaca.get_refined_ticker_list()

@save_and_load_data
def get_reddit_data(name):
    data = pd.read_csv(f"{name}.csv").sort_values('created')
    data['title'] = data['title'].fillna('')
    data['selftext'] = data['selftext'].fillna('')
    data['fulltext'] = data['title'] + " " + data['selftext']
    data['processed'] = preprocess_series(data.fulltext, valid_tickers = valid_tickers)
    return data

data = get_reddit_data("wallstreetbets_submissions")

S = sentiment_test.Sentiment(test_size=0.2)
#S.train_model(data, threshold=0.6722)
S.train_model(data)
with open("sentiment.pkl", "wb") as file:
    pickle.dump(S, file)


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


def evaluate_TextBlob(data, labels):
    # Splitting the dataset into training and testing sets

    X_test, y_test = data['processed'], labels
    # Predictions
    y_probs = pd.Series([TextBlob(x).sentiment.polarity for x in X_test])
    y_probs = (y_probs + 1)/2 #Normalize
    y_pred = y_probs >= 0.5

    # Evaluation
    #roc = roc_auc_score(y_test, y_probs)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    #print(f"ROC AUC Score: {roc}")
    print(classification_report(y_test, y_pred))
    

def evaluate_sentiment(data, labels):
    # Splitting the dataset into training and testing sets
    X_test, y_test = data['processed'], labels
    # Predictions
    y_probs, y_pred = S.predict(X_test, preprocess=False)

    # Evaluation
    #roc = roc_auc_score(y_test, y_probs)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    #print(f"ROC AUC Score: {roc}")
    print(classification_report(y_test, y_pred))
  
test_data = data.sample(20)
test_data2 = data.sample(10)

test_data.processed

test_data['sentiment'] = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0]
test_data2['sentiment'][20:] = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0]
evaluate_TextBlob(test_data2, test_data2['sentiment'])
evaluate_sentiment(test_data2,test_data2['sentiment'])



s = set(stopwords.words('english'))
s.update(['nan', 'deleted', 'removed', "edit"])
stp = lambda text: ' '.join(word for word in text.lower().split() if word not in s)

text = get_top_n_words(data.fulltext.apply(stp), n=10)
#text = get_top_n_words(data.processed.apply(stp), n=10)


words, counts = zip(*text)
plt.figure(figsize=(12, 8))
plt.bar(words, counts, color='blue', edgecolor='k')
plt.ylabel('Frequency', fontsize=14)
plt.title("Word Frequency (preprocessed)", fontsize=14)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, axis='y', linestyle='--', alpha=1)
plt.savefig('..\\..\\billeder\\Histogram processed.png')


@save_and_load_data
def get_ticker_mentions():
    tickers_series = (data['title'] + data['selftext']).fillna("").apply(lambda x: get_tickers_from_title(x,valid_tickers))
    return tickers_series

def find_most_common_tickers(num_tickers, include_count=False):
    ticker_series = get_ticker_mentions()
    all_tickers = [ticker for sublist in ticker_series for ticker in sublist]
    ticker_counts = Counter(all_tickers)
    top_tickers = ticker_counts.most_common(num_tickers)
    if not include_count:
        top_tickers = [ticker for ticker, _ in top_tickers]
    return top_tickers

tickers = find_most_common_tickers(9) #Excluding 'IT' from top 10






import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Assuming the required functions and data loading decorator are defined
def get_mentions(data):
    tickers_series = (data['title'] + data['selftext']).fillna("").apply(lambda x: get_tickers_from_title(x, valid_tickers))
    return tickers_series

def find_most_common_tickers(ticker_series, num_tickers, include_count=False):
    all_tickers = [ticker for sublist in ticker_series for ticker in sublist]
    ticker_counts = Counter(all_tickers)
    top_tickers = ticker_counts.most_common(num_tickers)
    if not include_count:
        top_tickers = [ticker for ticker, _ in top_tickers]
    return top_tickers

def get_ticker_counts_over_time(data, tickers, ticker_series, increment=50):
    ticker_counts_over_time = []
    for start in range(0, len(data), increment):
        end = start + increment
        subset = data.iloc[start:end]
        ticker_series_sub = ticker_series[start:end]
        all_tickers = [ticker for sublist in ticker_series_sub for ticker in sublist]
        ticker_counts = Counter(all_tickers)
        ticker_counts_over_time.append({ticker: ticker_counts[ticker] for ticker in tickers})
    return pd.DataFrame(ticker_counts_over_time)

def plot_ticker_counts_over_time(ticker_counts_df):
    ticker_counts_df.plot(kind='line', figsize=(10, 6))
    plt.title('Ticker Mentions Over Time')
    plt.xlabel('Increments of 50 posts')
    plt.ylabel('Number of Mentions')
    plt.legend(title='Tickers')
    plt.show()

# Get ticker mentions
ticker_series = get_ticker_mentions()

# Find the top 9 most common tickers excluding 'IT'
#top_tickers = find_most_common_tickers(ticker_series, 9)

# Get ticker counts over time
ticker_counts_df = get_ticker_counts_over_time(data, tickers, ticker_series)

# Plot the ticker counts over time
plot_ticker_counts_over_time(ticker_counts_df)




import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Assuming the required functions and data loading decorator are defined
@save_and_load_data
def get_ticker_mentions(data):
    tickers_series = (data['title'] + data['selftext']).fillna("").apply(lambda x: get_tickers_from_title(x, valid_tickers))
    return tickers_series

def find_most_common_tickers(ticker_series, num_tickers, include_count=False):
    all_tickers = [ticker for sublist in ticker_series for ticker in sublist]
    ticker_counts = Counter(all_tickers)
    top_tickers = ticker_counts.most_common(num_tickers)
    if not include_count:
        top_tickers = [ticker for ticker, _ in top_tickers]
    return top_tickers

def get_ticker_counts_over_time(data, tickers, ticker_series, increment='50D'):
    # Ensure 'created' column is in datetime format
    df = data.copy()
    df['created'] = pd.to_datetime(df['created'])
    df.set_index('created', inplace=True)
    ts = ticker_series.copy()
    ts.index = df.index

    # Resample the data based on the specified increment
    ticker_counts_over_time = []
    resampled_data = df.resample(increment)

    for _, subset in resampled_data:
        ts = ticker_series.loc[subset.index]
        all_tickers = [ticker for sublist in ts for ticker in sublist]
        ticker_counts = Counter(all_tickers)
        ticker_counts_over_time.append({ticker: ticker_counts[ticker] for ticker in tickers})
    
    ticker_counts_df = pd.DataFrame(ticker_counts_over_time, index=resampled_data.groups.keys())
    ticker_counts_df.fillna(0, inplace=True)
    return ticker_counts_df

def plot_ticker_counts_over_time(ticker_counts_df):
    ax = ticker_counts_df.plot(kind='line', figsize=(10, 6))
    plt.title('Ticker Mentions Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Mentions')
    # Define the formatter function for the y-axis
    def log_format(y, pos):
        return r'$10^{{{:.0f}}}$'.format(y)

    # Apply the formatter to the y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(log_format))
    plt.legend(title='Tickers')
    plt.show()



# Get ticker counts over time using 'created' column with increments of 50 days
ticker_counts_df = get_ticker_counts_over_time(data, tickers, ticker_series, increment='50D')

# Plot the ticker counts over time
plot_ticker_counts_over_time(np.log10(ticker_counts_df))





pattern = re.compile(r'\b(' + '|'.join(tickers) + r')\b', re.IGNORECASE)

# Filter the DataFrame using the compiled pattern
filtered_data = data[data['fulltext'].apply(lambda x: bool(pattern.search(x)))]


@save_and_load_data
def get_all_bars(ticker):
    # Convert start and end to datetime objects
    
    start = data.created.iloc[0]
    end = data.created.iloc[-1]
    # Initialize an empty DataFrame to hold all concatenated results
    all_bars = pd.DataFrame()
    # Loop through each year from start to end
    current_start = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    while current_start < end_date:
        # Set the end of the current period to the end of the current year, unless it exceeds the overall end date
        current_end = min(pd.Timestamp(year=current_start.year + 1, month=1, day=1) - timedelta(minutes=1), end_date)
        
        # Fetch data for the current period
        bars = alpaca.get_minute_bars_by_time(ticker, current_start.strftime('%Y-%m-%d %H:%M'), current_end.strftime('%Y-%m-%d %H:%M'))
        
        # Concatenate the fetched data with the accumulated DataFrame
        all_bars = pd.concat([all_bars, bars], ignore_index=True)
        
        # Update the start date for the next iteration to the beginning of the next year
        current_start = pd.Timestamp(year=current_start.year + 1, month=1, day=1)
    
    return all_bars

@save_and_load_data
def get_all_data(tickers):
    return {ticker: get_all_bars(ticker) for ticker in tickers}

@save_and_load_data
def get_reddit_posts(ticker):
    return data[data.fulltext.str.contains(ticker, case=False)]

@save_and_load_data
def get_all_merged_data(tickers):
    df = get_all_data(tickers)
    reddit_posts = {}
    for ticker in tickers:
        df[ticker]['log_return_close'] \
                = np.log(df[ticker]['close']).diff() 
        #reddit_posts[ticker] = get_reddit_posts(ticker)
        reddit_posts[ticker] = data[data.fulltext.str.contains(ticker, case=False)]
        reddit_posts[ticker]['sentiment'] = S.score(reddit_posts[ticker].processed)
        reddit_posts[ticker]['Time'] = pd.to_datetime(reddit_posts[ticker]['created']).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    

    merged = {ticker: pd.merge_asof(reddit_posts[ticker],
                                    df[ticker], 
                                    left_on='time', 
                                    right_on='Time', 
                                    direction='backward') for ticker in tickers}
    return merged

merged = get_all_merged_data(tickers)
for ticker in tickers:
    merged[ticker].set_index('Time', inplace=True)

t = max([merged[ticker].index.min() for ticker in tickers])
filtered_merged = {}
for ticker, df in merged.items():
    filtered_merged[ticker] = df[df.index >= t]


with open("filtered_merged.pkl", "wb") as file:
    pickle.dump(filtered_merged,file)

with open("filtered_merged.pkl", "rb") as file:
    filtered_merged = pickle.load(file)


bars = {}

for ticker in tickers:
    with open(f"data/get_all_bars_{ticker}_.pkl", "rb") as f:
        b = pickle.load(f)
    bars[ticker] = b



def piechart(ticker, data, threshold = 0.5):
    # Classify sentiment polarities as positive or negative
    positive_count = (data >= threshold).sum()
    negative_count = (data < threshold).sum()

    # Data for the pie chart
    labels = ['Positive', 'Negative']
    sizes = [positive_count, negative_count]
    colors = ['#1f77b4', '#ff7f0e']  # Blue for positive, orange for negative
    explode = (0.1, 0)  # "explode" the first slice (positive)

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title(f'Sentiment Polarity Distribution ({ticker})')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Show the plot
    plt.savefig(f'../../billeder/piechart ({ticker})')
    plt.close()

for ticker in tickers:
    piechart(ticker, filtered_merged[ticker].sentiment, threshold=0.5)


sentiments = np.array([[np.mean(filtered_merged[ticker].sentiment > 0.5), 
                        np.mean(filtered_merged[ticker].sentiment == 0.5),
                        np.mean(filtered_merged[ticker].sentiment < 0.5)] for ticker in tickers])




sentiments = np.array([sentiments, 1-sentiments]).transpose()

df = data[data['link_flair_text'].isin(['Gain', 'Loss'])]
df['sentiment'] = df['link_flair_text'].map({'Gain': 1, 'Loss': 0})
piechart(df.sentiment)



_,test = S.predict(df.processed)

piechart(test)



import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(12, 8))

# Loop through each ticker and plot its closing prices with time as the x-axis
for ticker in tickers:
    # Extract the closing prices and their corresponding time index for the current ticker
    close_prices = filtered_merged[ticker].close
    time_index = close_prices.index
    # Plot the closing prices against the time index
    plt.plot(time_index, close_prices, label=ticker)

# Add a vertical line to split train and test sets
split_time = pd.Timestamp('2021-04-09 14:46:00-04:00')
plt.axvline(x=split_time, color='r', linestyle='--')

# Add text labels for train and test sections
plt.text(split_time - pd.Timedelta(days=60), plt.ylim()[1]*(-0.15), 'Train', horizontalalignment='right', color='black', fontsize=16)
plt.text(split_time + pd.Timedelta(days=120), plt.ylim()[1]*(-0.15), 'Test', horizontalalignment='left', color='black', fontsize=16)

plt.title('Stock Prices Over Time')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
plt.show()




def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


# Define the resampling frequency and the rolling window size
resample_frequency = 'H'  # Daily resampling
window_size = '30D'  # 30-day rolling window

for ticker in tickers[0:1]:
    plt.figure(figsize=(12, 8))
    # Resample the data to a daily frequency
    resampled_data = filtered_merged[ticker]['sentiment'].resample(resample_frequency).mean()
    
    # Apply a rolling window with a 30-day window size
    rolling_mean = resampled_data.rolling(window=window_size).mean()
    stock_prices = filtered_merged[ticker]['log_return_close']
    
    # Resample the stock prices to a daily frequency
    resampled_stock_prices = stock_prices.resample(resample_frequency).mean()
    
    # Apply a rolling window with a 60-day window size to the stock prices
    rolling_stock_prices = resampled_stock_prices.rolling(window=window_size).mean()
    # Plot the resampled and smoothed data
    plt.plot(rolling_mean.index, normalize(rolling_mean), label=ticker)
    plt.plot(rolling_stock_prices.index, normalize(rolling_stock_prices), linestyle='--', label=f'{ticker} Stock Price (%)', alpha=0.7)
    plt.legend()
    #plt.savefig(f'..\\..\\billeder\\sentiment_vs_stock_prices_{ticker}.png')
    #plt.clf()
    plt.show()





import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Function to normalize the data
# def normalize(data):
#     scaler = StandardScaler()
#     normalized_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
#     return normalized_data


normalized_data = {}
from matplotlib.dates import YearLocator, DateFormatter
# Define the resampling frequency and the rolling window size
resample_frequency = 'min'  # Daily resampling
window_size = '30D'  # 30-day rolling window

normalized_data_list = []

for ticker in tickers:
    plt.figure(figsize=(12, 8))
    # Resample the data to a daily frequency
    resampled_data = filtered_merged[ticker]['sentiment'].resample(resample_frequency).mean()
    
    # Apply a rolling window with a 60-day window size
    rolling_mean = resampled_data.rolling(window=window_size).mean()
    stock_prices = filtered_merged[ticker]['log_return_close']
    
    # Resample the stock prices to a daily frequency
    resampled_stock_prices = stock_prices.resample(resample_frequency).mean()
    
    # Apply a rolling window with a 60-day window size to the stock prices
    rolling_stock_prices = resampled_stock_prices.rolling(window=window_size).mean()
    normalized_rolling_stock_prices = normalize(rolling_stock_prices)
    # Plot the resampled and smoothed data
    plt.plot(rolling_mean.index, rolling_mean, label='Sentiment')
    plt.plot(rolling_stock_prices.index, normalized_rolling_stock_prices, linestyle='--', label='Normalized log-returns', alpha=0.7)
    plt.title(f'Sentiment and Log-Returns Over Time for {ticker}', fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

    plt.gca().xaxis.set_major_locator(YearLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))

    plt.legend(fontsize = 20)
    
    # Construct the relative file path
    file_path = os.path.join('..', '..', 'billeder', f'sentiment_vs_stock_prices_{ticker}.png')
    
    # Save the plot to the specified directory
    plt.savefig(file_path)
    
    # Clear the plot for the next ticker
    plt.clf()
    # Construct the DataFrame with normalized data and indices
    df_normalized = pd.DataFrame({
        'Date': rolling_mean.index,
        'Ticker': ticker,
        'Sentiment': rolling_mean,
        'Normalized_Log_Returns': normalized_rolling_stock_prices
    }).dropna()  # Drop any rows with NaN values
    
    # Append the DataFrame to the list
    normalized_data_list.append(df_normalized)

# Concatenate all DataFrames into a single DataFrame
all_normalized_data = pd.concat(normalized_data_list, ignore_index=True)






#Cross-correlation:

from statsmodels.tsa.stattools import ccf

from scipy.stats import norm

# Function to plot cross-correlation function with confidence intervals
def plot_ccf_with_confidence(ticker, series1, series2, max_lag=20, alpha=0.05):
    lags = range(-max_lag, max_lag + 1)
    cross_corrs = [np.corrcoef(series1.shift(lag).dropna(), series2.loc[series1.shift(lag).dropna().index])[0, 1] for lag in lags]
    
    # Calculate confidence intervals
    conf_int = norm.ppf(1 - alpha / 2) / np.sqrt(len(series1.dropna()))
    
    plt.figure(figsize=(12, 8))
    plt.bar(lags, cross_corrs, color='blue', alpha=0.7)
    plt.axhline(y=conf_int, linestyle='--', color='red', linewidth=1.5, label='95% Confidence Interval')
    plt.axhline(y=-conf_int, linestyle='--', color='red', linewidth=1.5)
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation')
    plt.title(f'Cross-Correlation Function ({ticker})')
    plt.legend()
    plt.savefig(f'..\\..\\billeder\\Cross-Correlation ({ticker}).png')

# Example usage
for i in range(9):
    ticker = normalized_data_list[i].Ticker.iloc[0]
    plot_ccf_with_confidence(ticker, normalized_data_list[i].Sentiment, normalized_data_list[i].Normalized_Log_Returns, max_lag=120)

for ticker in tickers:
    plot_ccf_with_confidence(ticker, filtered_merged[ticker].sentiment, normalize(filtered_merged[ticker].log_return_close), max_lag=120)


filtered_merged[ticker].log_return_close





@save_and_load_data
def adftest(tickers):
    s = ""
    for ticker in tickers:
        s += ticker + "\n"
        s += "==========SCORES==========\n"
        s += str(adfuller(filtered_merged[ticker].sentiment)) + "\n" #stationary
        s += "==========PRICES==========\n"
        s += str(adfuller(filtered_merged[ticker]['close'].dropna())) + "\n" #not stationary
        s += "========LOG-PRICES========\n"
        s += str(adfuller(filtered_merged[ticker]['log_return_close'].dropna())) + "\n" #stationary
        s += "==========================\n"
    return s

print(adftest(tickers))

# df = get_all_data(tickers)
# merged_df = pd.merge_asof(df.sort_values('Time'), 
#                             posts.sort_values('time'), 
#                             left_on='Time', 
#                             right_on='time', 
#                             direction='backward')
#merged_df = merged_df.dropna(subset=['time'])
# 'direction' can be 'backward' (default), 'forward', or 'nearest'
# 'backward' finds the closest preceding match
# 'forward' finds the closest succeeding match
# 'nearest' finds the closest match regardless of direction


def granger(df,a,b, maxlag=4):
    with suppress_stdout():
        test = grangercausalitytests(df[[a, b]], maxlag=maxlag)
    # Assuming 'test' is your Granger causality test result and 'maxlag' is defined
    tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']
    p = {test_name: [test[i+1][0][test_name][1] for i in range(maxlag)] for test_name in tests}
    return p


tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']

p_values = pd.DataFrame(index=tickers, columns=tests)

for ticker in ticker_counts_df:
    #print(f"Granger test for {ticker}")
    g = granger(filtered_merged[ticker].dropna(), "log_return_close","sentiment",maxlag=50)
    p_values.loc[ticker] = g


ticker_below_05 = {ticker: any(p < 0.05 for p in p_values) for ticker, p_values in p_values.ssr_ftest.items()}
ticker_below_05





import contextlib
import os

# Function to suppress print statements
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.sys.stdout
        try:
            os.sys.stdout = devnull
            yield
        finally:
            os.sys.stdout = old_stdout

def granger(df, a, b, maxlag=4):
    with suppress_stdout():
        test = grangercausalitytests(df[[a, b]], maxlag=maxlag)
    tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']
    p = {test_name: [test[i+1][0][test_name][1] for i in range(maxlag)] for test_name in tests}
    return p

tests = ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']

# Assuming ticker_counts_df and filtered_merged are defined
p_values = pd.DataFrame(index=tickers, columns=tests)

for ticker in ticker_counts_df.index:
    g = granger(filtered_merged[ticker].dropna(), "log_return_close", "sentiment", maxlag=10)
    p_values.loc[ticker] = g

ticker_below_05 = {ticker: any(p < 0.05 for p in p_values.loc[ticker, 'ssr_ftest']) for ticker in p_values.index}
ticker_below_05






# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer()

# # Tokenize and build vocab
# tfidf_matrix = vectorizer.fit_transform(data.fulltext)

# # Summarize
# feature_names = vectorizer.get_feature_names_out()
# print("Feature names:", feature_names)

# # Display the TF-IDF matrix
# print("TF-IDF Matrix:")
# print(tfidf_matrix.toarray())



