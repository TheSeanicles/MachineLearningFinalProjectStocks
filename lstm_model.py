import numpy as np
import pandas as pd
from os.path import exists
import yaml
import tensorflow as tf
import matplotlib.pyplot as plt

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


def get_sp500():
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return tickers.Symbol.to_list()


def dataset_from_tickers(tickers_list):
    # Use Smallest Stock Dataset Size for Model Inputs
    dataset_size_list = []
    for t in tickers_list:
        if exists(config['path'] + '/' + t + '.csv'):
            dataset_size_list.append(len(pd.read_csv(config['path'] + '/' + t + '.csv')))
    frame_size = min(dataset_size_list)

    arrays_to_stack = []
    for t in tickers_list:
            if exists(config['path'] + '/' + t + '.csv'):
                df = pd.read_csv(config['path'] + '/' + t + '.csv')
                date_time = pd.to_datetime(df.pop('Datetime'), format='%Y-%m-%d %H:%M:%S%z')
                timestamp_s = date_time.map(pd.Timestamp.timestamp)
                day = 24 * 60 * 60
                year = (365.2425) * day
                # Normalize df before adding time back
                df = (df - df.mean()) / df.std()
                df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
                df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
                df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
                df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
                arrays_to_stack.append(df.iloc[-frame_size:-1].to_numpy())
    return np.stack(arrays_to_stack)


def plot_prediction(val_data, prediction):
    for i in range(val_data.shape[0]):
        # Plot all the close prices
        plt.plot(val_data[i, :, 3], label='data')
        plt.plot(prediction[i, :, 3], label='prediction')

        # Show the legend
        plt.legend()

        # Define the label for the title of the figure
        plt.title('Close', fontsize=16)

        # Define the labels for x-axis and y-axis
        plt.ylabel('Normalized Close Value', fontsize=14)
        plt.xlabel('Samples', fontsize=14)

        # Plot the grid lines
        plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
        plt.show()


def train_lstm_model(data):
    max_epochs = 200
    samples = data.shape[0]
    frame_size = data.shape[1]
    features = data.shape[2]

    samples_for_evaluation = 3

    in_size = frame_size // 2
    out_size = frame_size - in_size

    x = data[:samples - samples_for_evaluation, :in_size, :]
    y = data[:samples - samples_for_evaluation, in_size:, :]

    validation_data_x = data[samples - samples_for_evaluation:, :in_size, :]
    validation_data_y = data[samples - samples_for_evaluation:, in_size:, :]

    # Model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(out_size * features,
                              kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([out_size, features])
    ])

    model.compile(optimizer='adam', loss='MSE')

    model.fit(x=x, y=y, epochs=max_epochs)

    p = model.predict(validation_data_x)

    plot_prediction(validation_data_y, p)


if __name__ == '__main__':
    tick_list = []
    if config['S&P500']:
        for t in get_sp500():
            tick_list.append(t)
    for t in config['tickers']:
        tick_list.append(t)
    model_dataset = dataset_from_tickers(tick_list)
    train_lstm_model(model_dataset)
