from datetime import datetime
from dateutil import tz
import datetime as dt
import pytz
import re
import praw
import pandas as pd

import alpaca_api


alpaca = alpaca_api.Alpaca()

# def get_wsb_posts_for_day(date : str) -> list:
#     """
#     Used to get a list of all submissions from wallstreetbets
#     :param date: datetime formatted by dd/mm/yyyy
#     :type date: str
#     :return: A list of praw submissions
#     :rtype: list
#     """
#     day = int(date[0:2])
#     month = int(date[3:5]) 
#     year = int(date[6:])
#     epoch = int(dt.datetime(year, month, day).timestamp())
#     res = list(api.search_submissions(after=epoch,
#                                     before=epoch + (60 * 60 * 24),
#                                     subreddit='wallstreetbets',
#                                     filter=['url','author', 'title', 'subreddit'],
#                                     limit=10000))
#     return res


def get_wsb_posts_for_day(df : pd.DataFrame,  date : str):
    # Create a datetime object from the string
    # Note: the datetime object is 'naive' at this point, meaning it doesn't have timezone information
    epoch = datetime.strptime(date, "%d/%m/%Y")
    # If you need the datetime object to be 'aware', you can assign a timezone to it
    # For example, if the date string represents a time in UTC
    epoch = pytz.timezone('UTC').localize(epoch)
    dates = pd.to_datetime(df['created'], utc=True)
    res = df[(dates >= epoch) & (dates < epoch + pd.Timedelta(days=1))]
    
    return res

def get_tickers_from_title(title: str, valid_tickers: set) -> list:
    """
    Extracts a list of ticker symbols from a WSB post title.
    
    :param title: Title of the WSB post to search for ticker symbols within.
    :param valid_tickers: A set of valid ticker symbols for validation.
    :type title: str
    :type valid_tickers: set
    :return: A list of extracted and validated ticker symbols.
    :rtype: list[str]
    """
    # Regular expression to match ticker symbols, assuming they start with $ or are in uppercase letters
    ticker_pattern = re.compile(r'\$[A-Za-z]+|[A-Z]{2,}')
    matches = ticker_pattern.findall(title)
    
    # Validate and format tickers
    tickers = [match.strip('$').upper() for match in matches if match.strip('$').upper() in valid_tickers]
    
    return list(set(tickers))  # Remove duplicates by converting to a set and back to a list


def get_tradeable_periods(timestamp : int) -> dict:
    """
    Used to get a list of valid trading timeframes for this post
    Returns a dict like javascript which is terrible but don't judge
    :param timestamp: timestamp unix int
    :type timestamp: int
    :return: A dict of tradeable timeframes with boolean od whether or not they are tradeable
    :rtype: dict
    """
    adjustedDatetime = datetime.fromtimestamp(int(timestamp//10))
    adjustedDatetime.replace(tzinfo=tz.tzutc())
    adjustedDatetime = adjustedDatetime.astimezone(tz.gettz('America/New_York'))
    
    ret_dict = {"10min": False, "30min": False, "1hour": False}
    # check to see if the date is in a good range so we can check for winners
    if (int(adjustedDatetime.strftime('%H')) * 100) + int(adjustedDatetime.strftime('%M')) < 930:
        return ret_dict
    if (int(adjustedDatetime.strftime('%H')) * 100) + int(adjustedDatetime.strftime('%M')) < 1455:
        ret_dict["1hour"] = True
    if (int(adjustedDatetime.strftime('%H')) * 100) + int(adjustedDatetime.strftime('%M')) < 1525:
        ret_dict["30min"] = True
    if (int(adjustedDatetime.strftime('%H')) * 100) + int(adjustedDatetime.strftime('%M')) < 1545:
        ret_dict["10min"] = True 
        
    return ret_dict


