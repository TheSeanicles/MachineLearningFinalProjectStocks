import numpy as np
import pandas as pd
from os.path import exists
import yaml
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
import tensorflow as tf

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


def get_sp500():
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return tickers.Symbol.to_list()


def kmeans_dataset_from_tickers(tickers_list):
    # Use Smallest Stock Dataset Size for Model Inputs
    dataset_size_list = []
    for t in tickers_list:
        if exists(config['path'] + '/' + t + '.pkl'):
            dataset_size_list.append(len(pd.read_pickle(config['path'] + '/' + t + '.pkl')))
    frame_size = min(dataset_size_list)

    arrays_to_stack = []
    new_t_list = []
    for t in tickers_list:
            if exists(config['path'] + '/' + t + '.pkl'):
                new_t_list.append(t)
                df = pd.read_pickle(config['path'] + '/' + t + '.pkl')
                df = (df - df.mean()) / df.std()
                arrays_to_stack.append(df.iloc[-frame_size:-1].to_numpy())
    return np.stack(arrays_to_stack), new_t_list


def train_kmeans(data):
    for f in range(data.shape[2]):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[:, :, f])
        sse = []
        for k in range(1, 11):
            model = KMeans(init='random', n_clusters=k, n_init=10, max_iter=300)
            model.fit(scaled_features)
            sse.append(model.inertia_)
        knee = KneeLocator(range(1, 11), sse, curve='convex', direction='decreasing')
        model = KMeans(n_clusters=knee.elbow)
        model.fit(scaled_features)
        fte_colors = {0: "#008fd5", 1: "#fc4f30", 2: "#d66cb2", 3: "#772a98", 4: "#e69666"}
        km_colors = [fte_colors[label] for label in model.labels_]
        plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)
        plt.savefig('plots/feature'+str(f)+'kmeans.png')


def train_tseries_kmeans(data, tickers_list):
    map_stock_to_label = {}
    for f in range(data.shape[2]):
        print('Feature: ' + str(f))
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[:, :, f])
        # sse = []
        # for k in range(1, 11):
        #     print('Number of Clusters: ' + str(k))
        #     model = TimeSeriesKMeans(n_clusters=k, metric="softdtw", max_iter=10, n_jobs=-1, verbose=1)
        #     model.fit(scaled_features)
        #     sse.append(model.inertia_)
        # knee = KneeLocator(range(1, 11), sse, curve='convex', direction='decreasing')
        # print('Elbow: ' + str(knee.elbow))
        # model = TimeSeriesKMeans(n_clusters=knee.elbow, metric="softdtw", max_iter=10, n_jobs=-1, verbose=1)
        # model.fit(scaled_features)
        #
        # Grouped to an elbow of 2 on testing
        model = TimeSeriesKMeans(n_clusters=2, metric="softdtw", max_iter=10, n_jobs=-1, verbose=1)
        model.fit(scaled_features)
        model.to_pickle('saved_models/kmeans'+str(f))
        counter = 0
        for _ in model.labels_:
            if f in map_stock_to_label:
                if model.labels_[counter] in map_stock_to_label[f]:
                    map_stock_to_label[f][model.labels_[counter]].append(tickers_list[counter])
                else:
                    map_stock_to_label[f][model.labels_[counter]] = [tickers_list[counter]]
            else:
                map_stock_to_label[f] = {}
                map_stock_to_label[f][model.labels_[counter]] = [tickers_list[counter]]
            counter += 1
    return map_stock_to_label


def plot_groups(label_dict):
    for feature in label_dict:
        for group in label_dict[feature]:
            plt.clf()
            tickers_list = label_dict[feature][group]
            for t in tickers_list:
                if exists(config['path'] + '/' + t + '.pkl'):
                    df = pd.read_pickle(config['path'] + '/' + t + '.pkl')
                    df = (df - df.mean()) / df.std()
                    plt.plot(df.to_numpy()[-50:-1, feature])
                    plt.title('Group ' + str(group) + ' for feature ' + str(feature))
            plt.savefig('plots/feature'+str(feature)+'Group'+str(group)+'Plot.png')


def plot_prediction(val_data, prediction, filename, feature):
    fig, axs = plt.subplots(val_data.shape[0])
    for i in range(val_data.shape[0]):
        # Plot all the close prices
        axs[i].plot(val_data[i, :, feature], label='data')
        axs[i].plot(prediction[i, :, feature], label='prediction')

        # Show the legend
        axs[i].legend()

        # Define the label for the title of the figure
        axs[i].set_title('Feature: ' + str(feature))

        # Define the labels for x-axis and y-axis
        axs[i].set_ylabel('Normalized Close Value')
        axs[i].set_xlabel('Samples')

    plt.savefig('plots/'+filename)


def lstm_dataset_from_mapped_labels(label_dict):
    data = {}
    for feature in label_dict:
        data[feature] = {}
        for group in label_dict[feature]:
            tickers_list = label_dict[feature][group]
            # Use Smallest Stock Dataset Size for Model Inputs
            dataset_size_list = []
            for t in tickers_list:
                if exists(config['path'] + '/' + t + '.pkl'):
                    dataset_size_list.append(len(pd.read_pickle(config['path'] + '/' + t + '.pkl')))
            frame_size = min(dataset_size_list)

            arrays_to_stack = []
            for t in tickers_list:
                if exists(config['path'] + '/' + t + '.pkl'):
                    df = pd.read_pickle(config['path'] + '/' + t + '.pkl')
                    date_time = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S%z')
                    timestamp_s = date_time.map(pd.Timestamp.timestamp)
                    day = 24 * 60 * 60
                    year = 365.2425 * day
                    # Normalize df before adding time back
                    df = (df - df.mean()) / df.std()
                    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
                    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
                    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
                    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
                    arrays_to_stack.append(df.iloc[-frame_size:-1].to_numpy())
            data[feature][group] = np.stack(arrays_to_stack)
    return data


def train_lstm_model(label_dict):
    for feature in label_dict:
        for group in label_dict[feature]:
            data = label_dict[feature][group]

            max_epochs = 20
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
                tf.keras.layers.Dense(out_size * features, kernel_initializer=tf.initializers.zeros()),
                tf.keras.layers.Reshape([out_size, features])
            ])

            model.compile(optimizer='adam', loss='MSE')

            model.fit(x=x, y=y, epochs=max_epochs)

            model.save('saved_models/lstm'+str(feature)+str(group))

            p = model.predict(validation_data_x)

            filename = 'lstmFeature' + str(feature) + 'Group' + str(group) + '.png'

            plot_prediction(validation_data_y, p, filename, feature)


if __name__ == '__main__':
    tick_list = []
    if config['S&P500']:
        for t in get_sp500():
            tick_list.append(t)
    for t in config['tickers']:
        tick_list.append(t)
    model_dataset, tick_list = kmeans_dataset_from_tickers(tick_list)
    # train_kmeans(model_dataset)
    mapped_labels = train_tseries_kmeans(model_dataset, tick_list)
    plot_groups(mapped_labels)
    model_dataset = lstm_dataset_from_mapped_labels(mapped_labels)
    train_lstm_model(model_dataset)
