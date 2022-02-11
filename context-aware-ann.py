import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.losses import cosine_similarity
from sklearn.metrics import mean_absolute_error as metric_mae
from sklearn.metrics import mean_squared_error as metric_mse
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model

import os
import csv

SAVE_MODEL_PATH = "../"

DICT_LOSS_FUNCTION = {"mae": mean_absolute_error, "mse": mean_squared_error, "cosine_similarity": cosine_similarity}



class Context_Aware_Ann:

    # ----------
    # Initialize
    # ----------
    def __init__(self, model_type):
        self.model_type = model_type  # LSTM model type
        self.model = None  # LSTMs model
        self.series_df = None  # raw dataframe
        self.series = None  # processed series (list of dicts, each divided in train, val and test)
        self.h = None  # forecasting horizon
        self.lag = None  # lag variables used to forecast
        self.output_vars = None  # list with name(s) of output variables
        self.batch_size = None  # batch size
        self.interval_minute = None  # granularity in minutes
        self.window_size = None  # size of sub-datasets
        self.moving_step = None  # step size (between datasets)
        self.num_features = None  # number of features used in each timestep
        self.num_datasets = None  # number of datasets
        self.activation = None  # activation function
        self.num_units = ''  # num units if mlp
        # X and Y
        self.x_train = None
        self.x2_train = None
        self.y_train = None
        self.x_test = None
        self.x2_test = None
        self.y_test = None
        self.x_val = None
        self.x2_val = None
        self.y_val = None
        # Error array
        self.score_mae = []
        self.score_rmse = []
        self.scores_mae_horizon = []
        # History
        self.history = []
        # Linechart
        self.true_vs_pred = []  # num_datasets lists of [y_test, y_pred], y_test and y_pred are lists

    # ---------------------------------------------
    # Create datasets, train and evaluate the model
    # ---------------------------------------------
    def training(self, series_df, output_vars, interval_minute=None, window_size=72, moving_step=1,
                 dataset_dir_name=None,
                 h=1, lag=2, batch_size=8, l_r=10e-3, reg_value=0.1, reg_type=0, load_dataset=None, epochs=400,
                 loss_function='mse', instance_moving_step=1, stateful=False, in_days=False, mask='univariate'):

        self.series_df = series_df
        self.h = h
        self.lag = lag
        self.output_vars = output_vars
        self.interval_minute = interval_minute
        self.window_size = window_size
        self.moving_step = moving_step
        self.batch_size = batch_size

        self.series = self.split_dataset(in_days=in_days, window_size=window_size, moving_step=moving_step, lag=lag,
                                         h=h, train_val_size=0.8, instance_moving_step=instance_moving_step)

        self.num_features = self.series[0]["x_train"].shape[2]
        self.num_datasets = len(self.series)
        self.scores_mae_horizon = [[] for x in range(self.h)]

        print('\n\n\n--------------------------------------------')
        print("\n### Number of datasets: ", self.num_datasets)

        y_pred = []
        first_sub_dataset = True
        # For each sub-dataset
        for dataset in self.series:
            regularizer = self.set_regularization(reg_type, reg_value)
            self.set_model(n_timesteps=self.lag, n_features=self.num_features, n_outputs=self.h,
                           batch_size=self.batch_size, recurrent_regularizer=regularizer,
                           bias_regularizer=regularizer, l_r=l_r, loss_function=loss_function, dropout=0)

            self.x_train = dataset["x_train"]
            self.x2_train = dataset["x2_train"]
            self.y_train = dataset["y_train"]
            self.x_test = dataset["x_test"]
            self.x2_test = dataset["x2_test"]
            self.y_test = dataset["y_test"]
            self.x_val = dataset["x_val"]
            self.x2_val = dataset["x2_val"]
            self.y_val = dataset["y_val"]

            # Scale data
            self.x_train, self.x_val, self.x_test = self.scale_x(train=self.x_train,
                                                                 val=self.x_val,
                                                                 test=self.x_test,
                                                                 scaler=MinMaxScaler())
            self.x2_train, self.x2_val, self.x2_test = self.scale_x(train=self.x2_train,
                                                                 val=self.x2_val,
                                                                 test=self.x2_test,
                                                                 scaler=MinMaxScaler())
            self.y_train, self.y_val, self.y_test, y_scaler = self.scale_y(train=self.y_train,
                                                                           val=self.y_val,
                                                                           test=self.y_test,
                                                                           scaler=MinMaxScaler())

            if first_sub_dataset:
                print("\n### Number of training instances: ", self.x_train.shape[0])
                first_sub_dataset = False

            # Fit model
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40,
                                                  restore_best_weights=True)
            history = self.fit(x_train=self.x_train, x2_train=self.x2_train, y_train=self.y_train,
                               epochs=epochs, batch_size=self.batch_size,
                               x_val=self.x_val, x2_val=self.x2_val, y_val=self.y_val, callback=[es],
                               stateful=stateful)
            self.history.append(history)

            # Predict
            y_pred = self.model.predict([self.x_test, self.x2_test], verbose=0)
            y_pred = y_scaler.inverse_transform(y_pred)
            self.y_test = y_scaler.inverse_transform(self.y_test)

            # Evaluate model
            self.get_scores(y_pred=y_pred, y_test=self.y_test)
            self.true_vs_pred.append([self.y_test, y_pred])

            print("\n.\n")

        # Plots
        self.save_results(dataset_dir_name=dataset_dir_name, mask=mask, save_files=True, display=False)

        return self.mean_scores()

    # ------------
    # Save results
    # ------------
    def save_results(self, dataset_dir_name=None, mask='univariate', save_files=True, display=False):
        file_name = f'{mask}_{model_type}_lag{self.lag}_h{self.h}_batch{self.batch_size}'
        print('### File name: ', file_name)

        # self.plot_predictions(file_name, dataset_dir_name=dataset_dir_name, save_files=save_files, display=display)
        self.plot_history(file_name, dataset_dir_name=dataset_dir_name, save_files=save_files, display=display)
        self.save_scores(file_name, dataset_dir_name=dataset_dir_name, save_files=save_files, display=display)

    # -----------
    # Save scores
    # -----------
    def save_scores(self, file_name, dir_path='./', dataset_dir_name=None, save_files=True, display=False):
        if save_files:
            # Create dirs
            '''if os.path.exists(dir_path + dir_name):
                shutil.rmtree(dir_path + dir_name)'''
            if not os.path.exists(dir_path + dataset_dir_name):
                os.mkdir(dir_path + dataset_dir_name)
            if not os.path.exists(dir_path + dataset_dir_name + '/scores'):
                os.mkdir(dir_path + dataset_dir_name + '/scores')
            new_dir = dir_path + dataset_dir_name + '/scores/'

            # mean_scores = self.mean_scores()

            with open(new_dir + file_name, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                for dataset_mae, dataset_rmse in zip(self.score_mae, self.score_rmse):
                    writer.writerow(['mae', dataset_mae])
                    writer.writerow(['rmse', dataset_rmse])

            ########## SCORE PER TIMESTEP ##########
            # Create dirs
            '''if os.path.exists(dir_path + dir_name):
                shutil.rmtree(dir_path + dir_name)'''
            if not os.path.exists(dir_path + dataset_dir_name):
                os.mkdir(dir_path + dataset_dir_name)
            if not os.path.exists(dir_path + dataset_dir_name + '/scores_per_timestep'):
                os.mkdir(dir_path + dataset_dir_name + '/scores_per_timestep')
            new_dir = dir_path + dataset_dir_name + '/scores_per_timestep/'

            with open(new_dir + file_name, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                for t, t_maes in enumerate(self.scores_mae_horizon):
                    for mae in t_maes:
                        writer.writerow(['mae', t, mae])

    # ------------
    # Plot history
    # ------------
    def plot_history(self, file_name, dir_path='./', dataset_dir_name=None, save_files=False, display=False):
        history = self.history[0].history

        if save_files:
            # Create dirs
            '''if os.path.exists(dir_path + dir_name):
                shutil.rmtree(dir_path + dir_name)'''
            if not os.path.exists(dir_path + dataset_dir_name):
                os.mkdir(dir_path + dataset_dir_name)
            if not os.path.exists(dir_path + dataset_dir_name + '/history'):
                os.mkdir(dir_path + dataset_dir_name + '/history')
            new_dir = dir_path + dataset_dir_name + '/history/'

            # Save
            # with open(new_dir + file_name, 'w', encoding='UTF8') as f:
            #    writer = csv.writer(f)
            #    writer.writerow(history)
            with open(new_dir + file_name, 'wb') as file_pi:
                pickle.dump(history, file_pi)

        if display:
            plt.show()

    # ----------------
    # Plot predictions
    # ----------------
    def plot_predictions(self, file_name, dir_path='./', dataset_dir_name=None, save_files=False, display=False):
        fig, ax = plt.subplots(self.num_datasets, 1)
        fig.suptitle(f'True vs Predicted - {file_name}')
        plt.ylabel('count')

        for index, data in enumerate(self.true_vs_pred):
            obs = [value for sublist in data[0] for value in sublist]
            pred = [value for sublist in data[1] for value in sublist]
            if self.num_datasets == 1:
                sns.lineplot(ax=ax, data=obs, label='True')
                sns.lineplot(ax=ax, data=pred, linestyle='--', label='Predicted')
            else:
                sns.lineplot(ax=ax[index], data=obs, label='True')
                sns.lineplot(ax=ax[index], data=pred, linestyle='--', label='Predicted')
                ax[index].set_title(f'Dataset {index}')

        if save_files:
            # Create dirs
            '''if os.path.exists(dir_path + dir_name):
                shutil.rmtree(dir_path + dir_name)'''
            if not os.path.exists(dir_path + dataset_dir_name):
                os.mkdir(dir_path + dataset_dir_name)
            if not os.path.exists(dir_path + dataset_dir_name + '/pred'):
                os.mkdir(dir_path + dataset_dir_name + '/pred')
            new_dir = dir_path + dataset_dir_name + '/pred/'
            fig.savefig(os.path.join(new_dir, file_name))

        if display:
            plt.show()

    # ------------------------------
    # Fit the model to training data
    # ------------------------------
    def fit(self, x_train, x2_train, y_train, epochs, batch_size, x_val, x2_val, y_val, callback, stateful):
        history = None

        if stateful:
            for i in range(0, epochs):
                history = self.model.fit(x=[x_train, x2_train], y=y_train, epochs=batch_size, batch_size=1,
                                         validation_data=([x_val, x2_val], y_val), verbose=0, callbacks=callback)
                self.model.reset_states()
        else:
            history = self.model.fit(x=[x_train, x2_train], y=y_train, epochs=epochs, batch_size=batch_size,
                                     validation_data=([x_val, x2_val], y_val), verbose=0, callbacks=callback)

        return history

    # ------------------
    # Compute the scores
    # ------------------
    def get_scores(self, y_pred, y_test):
        self.score_rmse.append(metric_mse(y_test, y_pred, squared=False))
        self.score_mae.append(metric_mae(y_test, y_pred))
        for t in range(self.h):
            self.scores_mae_horizon[t].append(metric_mae(y_test[:, t], y_pred[:, t]))

    # -------
    # Scale x
    # -------
    def scale_x(self, train, val, test, scaler):
        scaler.fit(train.reshape(-1, train.shape[1]))
        scaled_train = scaler.transform(train.reshape(-1, train.shape[1])).reshape(train.shape[0],
                                                                                   train.shape[1],
                                                                                   train.shape[2])
        scaled_val = scaler.transform(val.reshape(-1, train.shape[1])).reshape(val.shape[0],
                                                                               val.shape[1],
                                                                               val.shape[2])
        scaled_test = scaler.transform(test.reshape(-1, train.shape[1])).reshape(test.shape[0],
                                                                                 test.shape[1],
                                                                                 test.shape[2])

        return scaled_train, scaled_val, scaled_test

    # -------
    # Scale y
    # -------
    def scale_y(self, train, val, test, scaler):
        scaler.fit(train)
        scaled_train = scaler.transform(train)
        scaled_val = scaler.transform(val)
        scaled_test = scaler.transform(test)

        return scaled_train, scaled_val, scaled_test, scaler

    # -----------------------------
    # Split dataset in sub-datasets
    # -----------------------------
    def split_dataset(self, in_days, window_size, moving_step, lag, h,
                      train_val_size, instance_moving_step):
        series = []
        dataset_len = len(self.series_df.iloc[:, 0])  # number of data points
        for end_index in range(window_size, dataset_len + 1, moving_step):
            begin_index = end_index - window_size
            sub_dataset = self.series_df.iloc[begin_index:end_index]
            series.append(self.create_instance_sub_dataset(sub_dataset=sub_dataset, lag=lag,
                                                           h=h, moving_step=instance_moving_step,
                                                           train_val_size=train_val_size, train_size=0.8))
        return series

    # -----------------------------------------------
    # Split sub-dataset in train, validation and test
    # -----------------------------------------------
    def create_instance_sub_dataset(self, sub_dataset, lag, h, moving_step, train_val_size,
                                    train_size):
        x_data = []
        y_data = []
        x2_data = []

        for end_index in range(lag + h, len(sub_dataset) + 1, moving_step):
            begin_index = end_index - (lag + h)
            x_data.append(sub_dataset.iloc[begin_index:begin_index + lag].to_numpy())
            y_data.append(
                sub_dataset.iloc[begin_index + lag:end_index][self.output_vars].to_numpy().flatten())
            x2_data.append(
                sub_dataset.iloc[begin_index + lag:end_index].drop(self.output_vars, axis=1).to_numpy())

        train_val_len = int(len(x_data) * train_val_size)

        x_train_val = x_data[:train_val_len]
        y_train_val = y_data[:train_val_len]
        x2_train_val = x2_data[:train_val_len]

        train_len = int(len(x_train_val) * train_size)

        x_validation = x_train_val[train_len:train_val_len]
        y_validation = y_train_val[train_len:train_val_len]
        x2_validation = x2_train_val[train_len:train_val_len]

        x_train = x_train_val[:train_len]
        y_train = y_train_val[:train_len]
        x2_train = x2_train_val[:train_len]

        x_test = x_data[train_val_len:len(x_data)]
        y_test = y_data[train_val_len:len(y_data)]
        x2_test = x2_data[train_val_len:len(x2_data)]

        serie = {"x_train": np.array(x_train), "y_train": np.array(y_train), "x2_train": np.array(x2_train),
                 "x_test": np.array(x_test), "y_test": np.array(y_test), "x2_test": np.array(x2_test),
                 "x_val": np.array(x_validation), "y_val": np.array(y_validation), "x2_val": np.array(x2_validation)}

        return serie

    # ---------------------------------
    # Set regularization type and value
    # ---------------------------------
    def set_regularization(self, reg_type, reg_value):
        if reg_type == 0:
            return tf.keras.regularizers.l1(reg_value)
        if reg_type == 1:
            return tf.keras.regularizers.l2(reg_value)
        if reg_type == 2:
            return tf.keras.regularizers.l1_l2(l1=reg_value, l2=reg_value)

    # ------------------------------
    # Average scores across datasets
    # ------------------------------
    def mean_scores(self):
        sum_mae = 0
        sum_rmse = 0
        for mae, rmse in zip(self.score_mae, self.score_rmse):
            sum_mae += mae
            sum_rmse += rmse

        mean_scores = {'mae': sum_mae / self.num_datasets, 'rmse': sum_rmse / self.num_datasets}
        return mean_scores

    # -----------------
    # Define LSTM model
    # -----------------
    def set_model(self, n_timesteps, n_features, n_outputs, batch_size, recurrent_regularizer, bias_regularizer,
                  l_r, loss_function, dropout=0, stateful=False):

        # Activation function
        self.activation = 'relu'

        ########## LSTMS ##########
        if self.model_type == 'context_lstms':

            # LSTM 1
            input1 = Input(shape=(n_timesteps, n_features))
            lstm1 = LSTM(16,
                         activation=self.activation,
                         # recurrent_activation="relu",
                         stateful=stateful,
                         recurrent_regularizer=recurrent_regularizer,
                         bias_regularizer=bias_regularizer, dropout=dropout, return_sequences=False)(input1)
            dense1 = Dense(n_outputs)(lstm1)
            reshape1 = Reshape((-1, 1))(dense1)

            # Add prospective information
            input2 = Input(shape=(n_outputs, n_features - 1))

            # Merge
            merge = concatenate([reshape1, input2], axis=2)

            # LSTM 2
            lstm2 = LSTM(8,
                         activation=self.activation,
                         # recurrent_activation="relu",
                         stateful=stateful,
                         recurrent_regularizer=recurrent_regularizer,
                         bias_regularizer=bias_regularizer, dropout=dropout, return_sequences=False)(merge)
            dense2 = Dense(n_outputs)(lstm2)

            self.model = Model(inputs=[input1, input2], outputs=dense2)

            optimizer = tf.keras.optimizers.Adam(learning_rate=l_r, clipnorm=0.8, clipvalue=0.4)
            self.model.compile(loss=DICT_LOSS_FUNCTION.get(loss_function), optimizer=optimizer)

            # Plot graph
            # plot_model(self.model, to_file='./model_plots/plot_context_lstms.png', show_shapes=True)

        ########## LSTM + MLP ##########
        if self.model_type == 'context_lstm_mlp':
            # LSTM 1
            input1 = Input(shape=(n_timesteps, n_features))
            lstm1 = LSTM(16,
                         activation=self.activation,
                         # recurrent_activation="relu",
                         stateful=stateful,
                         recurrent_regularizer=recurrent_regularizer,
                         bias_regularizer=bias_regularizer, dropout=dropout, return_sequences=False)(input1)
            dense1 = Dense(n_outputs)(lstm1)
            reshape1 = Reshape((-1, 1))(dense1)

            # Add prospective information
            input2 = Input(shape=(n_outputs, n_features - 1))

            # Merge
            merge = concatenate([reshape1, input2], axis=2)

            # MLP 1
            flatten = Flatten()(merge)
            mlp1 = Dense(16,
                         activation=self.activation,
                         bias_regularizer=bias_regularizer)(flatten)
            dense2 = Dense(n_outputs)(mlp1)

            self.model = Model(inputs=[input1, input2], outputs=dense2)

            optimizer = tf.keras.optimizers.Adam(learning_rate=l_r, clipnorm=0.8, clipvalue=0.4)
            self.model.compile(loss=DICT_LOSS_FUNCTION.get(loss_function), optimizer=optimizer)

            # Plot graph
            # plot_model(self.model, to_file='./model_plots/plot_context_lstm_mlp.png', show_shapes=True)

        ########## MLPS ##########
        if self.model_type == 'context_mlps':
            # MLP 1
            input1 = Input(shape=(n_timesteps, n_features))
            flatten1 = Flatten()(input1)
            dense1 = Dense(16,
                         activation=self.activation,
                         bias_regularizer=bias_regularizer)(flatten1)
            dense2 = Dense(16,
                           activation=self.activation,
                           bias_regularizer=bias_regularizer)(dense1)
            dense3 = Dense(n_outputs)(dense2)
            reshape1 = Reshape((-1, 1))(dense3)

            # Add prospective information
            input2 = Input(shape=(n_outputs, n_features - 1))

            # Merge
            merge = concatenate([reshape1, input2], axis=2)

            # MLP 2
            flatten2 = Flatten()(merge)
            dense4 = Dense(16,
                         activation=self.activation,
                         bias_regularizer=bias_regularizer)(flatten2)
            dense5 = Dense(16,
                           activation=self.activation,
                           bias_regularizer=bias_regularizer)(dense4)
            dense6 = Dense(n_outputs)(dense5)

            self.model = Model(inputs=[input1, input2], outputs=dense6)

            optimizer = tf.keras.optimizers.Adam(learning_rate=l_r, clipnorm=0.8, clipvalue=0.4)
            self.model.compile(loss=DICT_LOSS_FUNCTION.get(loss_function), optimizer=optimizer)

            # Plot graph
            plot_model(self.model, to_file='model_plots/plot_context_mlps.png', show_shapes=True)

        # self.model.summary()


# ----
# Main
# ----
if __name__ == '__main__':

    # --------------------------------------------------------------------------------------------

    i_date = datetime(2016, 1, 1)  # initial date
    f_date = datetime(2018, 1, 1)  # final date

    dataset_dir_name = f'dataset_{i_date.strftime("%Y%m%d")}_{f_date.strftime("%Y%m%d")}'

    hs = [24]  # lengths of forecasting horizons to consider
    lags = [24 * 7]  # lengths of historical context data to consider
    instance_moving_steps = [1]  # step sizes between input-output instances to consider
    batches = [8]   # batch sizes to consider

    model_types = ['context_lstms']  # model types to consider

    # --------------------------------------------------------------------------------------------

    # Retrieve dataframe containing target variable
    events = None

    # Retrieve dataframe containing weather
    meteo = None

    # Test
    for instance_moving_step in instance_moving_steps:

        masks = [f'hour_step{instance_moving_step}',
                 f'weekday_step{instance_moving_step}',
                 f'weekend_step{instance_moving_step}',
                 f'hour_weekday_weekend_step{instance_moving_step}',
                 f'temp_step{instance_moving_step}',
                 f'pp_step{instance_moving_step}',
                 f'rh_step{instance_moving_step}',
                 f'temp_pp_rh_step{instance_moving_step}',
                 f'hour_weekday_weekend_temp_step{instance_moving_step}',
                 f'hour_weekday_weekend_pp_step{instance_moving_step}',
                 f'hour_weekday_weekend_rh_step{instance_moving_step}',
                 f'hour_weekday_weekend_temp_pp_rh_step{instance_moving_step}']

        for mask in masks:
            df = events.copy()

            time_range = pd.date_range(i_date, f_date, freq='H')
            df = df.reindex(time_range, fill_value=0)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'time'}, inplace=True)

            # In case there are missing values, fill using the mean of all the data points at the corresponding hour
            df['hour'] = df['time'].dt.hour.astype('int')
            df = df.fillna(df.groupby('hour').transform('mean'))
            df = df.drop(['hour'], axis=1)

            # Create masks
            if 'weekday' in mask:
                df['weekday'] = [record.weekday() for record in df['interval_start']]
            if 'weekend' in mask:
                df['weekend'] = [0 if record.weekday() < 5 else 1 for record in df['interval_start']]
            if 'hour' in mask:
                df['hour'] = [record.hour for record in df['interval_start']]
            if 'temp' in mask:
                df['temp'] = meteo['temp']  # temperature
            if 'rh' in mask:
                df['rh'] = meteo['rh']  # relative humidity
            if 'pp' in mask:
                df['pp'] = meteo['pp']  # precipitation

            df = df.set_index('time')

            for model_type in model_types:
                for h in hs:
                    for lag in lags:
                        for batch in batches:
                            # Print series
                            print('\n### Dataframe: ', df.head())
                            print('\n### Number of data points: ', len(df))

                            # Train
                            forecaster = Context_Aware_Ann(model_type)
                            scores = forecaster.training(series_df=df, output_vars=['count'], h=h, lag=lag,
                                                  window_size=24 * 7 * 6,  # size of each dataset
                                                  moving_step=24 * 7,  # step size between datasets
                                                  instance_moving_step=instance_moving_step,  # step size between input-output instances within a dataset
                                                  dataset_dir_name=dataset_dir_name, mask=mask, batch_size=batch)

                            # Print scores
                            print(f'\n### Mask: {mask}')
                            print(f'\n### Horizon: {h}')
                            print(f'\n### Lag: {lag}')
                            print(f'\n### Batch size: {batch}')
                            print('\n### Scores:')
                            print(scores)
                            print('\n\n\n\n\n')
