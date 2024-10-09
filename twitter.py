import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import matplotlib.ticker as mtick
plt.style.use('ggplot')

# Load sentiment data
sentiment_df = pd.read_csv('sentiment_data.csv')

# Convert 'date' column to datetime
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

# Set multi-index with 'date' and 'symbol'
sentiment_df = sentiment_df.set_index(['date', 'symbol'])

# Calculate engagement ratio
sentiment_df['engagement_ratio'] = sentiment_df['twitterComments'] / sentiment_df['twitterLikes']

# Filter DataFrame based on criteria
sentiment_df = sentiment_df[(sentiment_df['twitterLikes'] > 20) & (sentiment_df['twitterComments'] > 10)]

# Aggregate engagement ratio by month and symbol
aggragated_df = (sentiment_df.reset_index('symbol').groupby([pd.Grouper(freq='ME'), 'symbol'])
                 [['engagement_ratio']].mean())

# Rank engagement ratio within each month
aggragated_df['rank'] = (aggragated_df.groupby(level=0)['engagement_ratio']
                         .transform(lambda x: x.rank(ascending=False)))

# Filter top 5 engagement ratios
filtered_df = aggragated_df[aggragated_df['rank'] < 6].copy()

# Reset index to work with dates
filtered_df = filtered_df.reset_index(level=1)

# Offset index for clarity (might cause misalignment in future use, double-check)
filtered_df.index = filtered_df.index + pd.DateOffset(1)

# Reset index and set multi-index again
filtered_df = filtered_df.reset_index().set_index(['date', 'symbol'])

# Print filtered data (optional for debugging)
print(filtered_df.head(20))

# Get unique dates from filtered DataFrame
dates = filtered_df.index.get_level_values('date').unique().tolist()

# Prepare fixed_dates for aggregation
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()

# Get list of unique stocks
stocks_list = sentiment_df.index.get_level_values('symbol').unique().tolist()

# Function to check if a ticker is valid
def is_valid_ticker(ticker):
    try:
        # Attempt to download the data for the ticker
        data = yf.download(tickers=ticker, start='2021-01-01', end='2023-03-01', progress=False)
        # Check if the DataFrame is empty, meaning the ticker is likely invalid
        return not data.empty
    except (yf.YFinanceError, ValueError) as e:
        print(f"Error retrieving {ticker}: {e}")
        return False  # Return False if there's an error

# Filter stocks_list to only include valid tickers
valid_stocks_list = [ticker for ticker in stocks_list if is_valid_ticker(ticker)]

# Print valid tickers for debugging
print("Valid Tickers:", valid_stocks_list)

# Download historical stock prices using yfinance for valid tickers
prices_df = yf.download(tickers=valid_stocks_list,
                        start='2021-01-01',
                        end='2023-03-01',
                        progress=False)

# Calculate log returns of the adjusted close prices
returns_df = np.log(prices_df['Adj Close']).diff().dropna()

# Initialize portfolio DataFrame for returns
portfolio_df = pd.DataFrame()

# Calculate mean returns for each time period
for start_date in fixed_dates.keys():
    end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd()).strftime('%Y-%m-%d')
    cols = fixed_dates[start_date]
    
    temp_df = returns_df[start_date:end_date][cols].mean(axis=1).to_frame('portfolio_return')
    portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

# Download QQQ (NASDAQ) returns
qqq_df = yf.download(tickers='QQQ',
                     start='2021-01-01',
                     end='2023-03-01',
                     progress=False)

# Calculate log returns for QQQ
qqq_ret = np.log(qqq_df['Adj Close']).diff().to_frame('nasdaq_return')

# Check and align timezones before merging
if portfolio_df.index.tz is None and qqq_ret.index.tz is not None:
    portfolio_df.index = portfolio_df.index.tz_localize('UTC')
elif portfolio_df.index.tz is not None and qqq_ret.index.tz is None:
    qqq_ret.index = qqq_ret.index.tz_localize('UTC')

# Merge portfolio and QQQ returns
portfolio_df = portfolio_df.merge(qqq_ret, left_index=True, right_index=True)

# Calculate cumulative returns for the portfolio
portfolios_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()).sub(1)

# Plot the results
portfolios_cumulative_return.plot(figsize=(16, 6))
plt.title('Twitter Engagement Ratio Strategy Return Over Time')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylabel('Return')
plt.show()
