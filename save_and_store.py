import pandas as pd
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from os.path import exists
import os
import yaml

# THIS FILE SAVES THE PAST 60 DAYS OF STOCK DATA AT A 5 MINUTE INTERVAL


with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


# data = stockDataFetch(['AAPL'], 'max', '1d')
# valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#
# fetch data by interval (including intraday if period < 60 days)
# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
def stock_data_fetch(tickers_list, time_duration, time_interval):
    return yf.download(tickers_list, period=time_duration, interval=time_interval, group_by='ticker', threads='True')


# plotStockData(['AAPL'], 'max', '1d')
# valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#
# fetch data by interval (including intraday if period < 60 days)
# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
def plot_stock_data_download(tickers_list, time_duration, time_interval):
    data = stock_data_fetch(tickers_list, time_duration, time_interval)
    # Plot all the close prices
    ((data['Close'].pct_change() + 1).cumprod()).plot(figsize=(10, 7))

    # Show the legend
    plt.legend()

    # Define the label for the title of the figure
    plt.title("Returns", fontsize=16)

    # Define the labels for x-axis and y-axis
    plt.ylabel('Cumulative Returns', fontsize=14)
    plt.xlabel('Year', fontsize=14)

    # Plot the grid lines
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.show()


def plot_stock_data_stored(tickers_list):
    for t in tickers_list:
        if exists(config['path'] + '/' + t + '.pkl'):
            data = pd.read_pickle(config['path'] + '/' + t + '.pkl')
            # Plot all the close prices
            ((data['Close'].pct_change() + 1).cumprod()).plot(figsize=(10, 7))

            # Show the legend
            plt.legend()

            # Define the label for the title of the figure
            plt.title(str(t) + " Returns", fontsize=16)

            # Define the labels for x-axis and y-axis
            plt.ylabel('Cumulative Returns', fontsize=14)
            plt.xlabel('Samples', fontsize=14)

            # Plot the grid lines
            plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
            plt.show()
        else:
            print('Ticker ' + str(t) + ' not stored locally in path.')


def get_sp500():
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return tickers.Symbol.to_list()


def stock_save(tickers_list):
    tod = datetime.datetime.now()
    d59 = datetime.timedelta(days=59)
    past = tod - d59
    day = datetime.timedelta(days=1)
    if exists(config['path'] + '/last_download.txt'):
        f = open(config['path'] + '/last_download.txt', 'r')
        last_download = f.read()
        f.close()
        last_date = datetime.datetime.strptime(last_download, '%Y-%m-%d %H:%M:%S.%f')
    else:
        last_date = tod - day
    if tod - last_date >= day:
        for t in tickers_list:
            # GRAB DATA FROM YAHOO FINANCE
            new_data = yf.download([t],
                               start=past.date(),
                               end=tod.date(),
                               interval='5m',
                               group_by='ticker',
                               threads='True')
            if not new_data.empty:
                # EXPORT DATA TO BETTER MANIPULATE
                if exists(config['path'] + '/' + t + '.pkl'):
                    original_data = pd.read_pickle(config['path'] + '/' + t + '.pkl')
                    merge_data = pd.merge(new_data, original_data, how='outer')
                    merge_data = merge_data[merge_data.Datetime != 'NaN']
                    merge_data.to_pickle(config['path'] + '/' + t + '.pkl', index=False)
                else:
                    new_data.to_pickle(config['path'] + '/' + t + '.pkl')

        with open(config['path'] + '/last_download.txt', 'w') as f:
            write_str = tod.strftime('%Y-%m-%d %H:%M:%S.%f')
            f.write(write_str)


def delete_last_download():
    if exists(config['path'] + '/last_download.txt'):
        os.remove(config['path'] + '/last_download.txt')


def delete_all_data():
    files_in_path = os.listdir(config['path'])
    for f in files_in_path:
        os.remove(config['path'] + '/' + f)


if __name__ == '__main__':
    tick_list = []
    if config['delete_data']:
        delete_all_data()
    if config['delete_last_download']:
        delete_last_download()
    if config['S&P500']:
        for t in get_sp500():
            tick_list.append(t)
    for t in config['tickers']:
        tick_list.append(t)
    stock_save(tick_list)
    # plot_stock_data_stored(tick_list)
