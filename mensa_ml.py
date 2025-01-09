# basic packages
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from pathlib import Path

# ml packages
from sklearn.model_selection import KFold
import keras
import matplotlib.dates as mdates

# read dataset
PATH_TO_DATA = Path(__file__).parent.joinpath("daily_data.csv")
df_data = pd.read_csv(PATH_TO_DATA, index_col=0)

# some parameters of a model
# keras.utils.set_random_seed(42)
LOSS = keras.losses.LogCosh() # keras.losses.MeanAbsoluteError()
N_DAYS = 14 # time frame of days
N_YEARS = 2 # amount of years in consideration
N_PADDING = 3 * N_DAYS + 2
BATCH_SIZE = 50
N_EPOCHS = 120

# create a model
def create_model(n_features: int, out_shape: int) -> keras.Sequential:
    # create model
    model = keras.Sequential()
    # add layers
    model.add(
        keras.layers.InputLayer(
            shape=(None, n_features)
        )
    )
    # encoder for features
    model.add(
        keras.layers.Dense(
            units=128,
            activation="tanh",
            kernel_regularizer=keras.regularizers.L2(l2=0.01),
            bias_regularizer=keras.regularizers.L2(l2=0.1),
            use_bias=True,
        )
    )
    # LSTM for memory
    model.add(
        keras.layers.LSTM(
            units=128, # amount of memory neurons
            return_sequences=False, # just one return for a sequence
            kernel_regularizer=keras.regularizers.L2(l2=0.01),
            recurrent_regularizer=keras.regularizers.L2(l2=0.01),
            use_bias=False,
        )
    )
    # decoder
    model.add(
        keras.layers.Dense(
            units=out_shape,
            activation="linear",
            kernel_regularizer=keras.regularizers.L2(l2=0.01),
            use_bias=False,
        )
    )
    # Leaky-ReLU
    model.add(keras.layers.LeakyReLU(negative_slope=0.2))

    # compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
        loss=LOSS, 
        metrics=["mae"]
    )

    return model

# converter string to date and back
def string_to_datetime(date_string: str) -> datetime:
    return datetime.strptime(date_string, "%Y-%m-%d")

def datetime_to_string(date: datetime) -> str:
    return datetime.strftime(date, "%Y-%m-%d")

# frames for date
def get_data_advance(date_string: str, n_days: int = 7) -> list[str]:
    """Get data several days in advance."""
    # convert do datetime
    target_date = string_to_datetime(date_string=date_string)

    # get dates of the last week
    last_week = [(target_date - timedelta(days=i)).date() for i in range(1, n_days + 1)][::-1]

    # get last year similar date
    year_ago = target_date - relativedelta(years=1)
    
    # get date for time period in the past
    past_before = [(year_ago - timedelta(days=i)).date() for i in range(1, n_days + 1)][::-1]
    past_after = [(year_ago + timedelta(days=i)).date() for i in range(0, n_days + 1)]

    # sum up and sort in asceding order
    frame = sorted(past_before + past_after + last_week)

    return [datetime_to_string(date) for date in frame]

def get_frame_features(frame: list[str], df: pd.DataFrame) -> np.ndarray:
    # feaures of all dates in one frame
    features = list()

    # for each date in frame get data from dataframe if exists
    for date in frame:
        if date in df.index:
            day_features = list(df.loc[date])
            features.append(day_features)
    
    # return features as matrix [day x feature]
    return np.array(features) if len(features) > 0 else np.zeros((0, df.shape[1]))

def get_fold_for_date(date_str: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # get target data
    y_data = df.loc[date_str, "Target"]
    raw = df.loc[date_str].copy()
    raw["Target"] = 0.0
    y_features = np.array(list(raw))

    # get frame for this data
    frame = get_data_advance(date_string=date_str, n_days=N_DAYS)
    X_data = get_frame_features(frame=frame, df=df)
    X_data = np.vstack([X_data, y_features])

    return X_data, np.array([y_data])

def padding(n_max: int, arr: np.ndarray) -> np.ndarray:
    # add zeros to the sequences for cardinality alignment
    rows_to_add = n_max - arr.shape[0]

    # create matrix
    matrix = np.zeros(shape=(rows_to_add, arr.shape[1]))
    return np.vstack([matrix, arr])

def get_data_from_dates(dates: list[str], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # convert to features for train data
    X_list, y_list = list(), list()
    for date_str in dates:
        X, y = get_fold_for_date(date_str=date_str, df=df)
        # apply padding to X
        X = padding(N_PADDING, X)
        X_list.append(X)
        y_list.append(y)

    return np.array(X_list), np.array(y_list)

def plot_history(history):
    # plot learning
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Training Loss", color="blue")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="orange")
    plt.title("Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()

# split to train, test, validation
BORDER_DATE = "2023-06-01"
test_data = df_data.index[df_data.index >= BORDER_DATE] # dates that will not be shown to models

# other stuff must be splitted
rest_data = list(set(df_data.index).difference(set(test_data))) # all other dates for training


# K-Fold cross validation to check performance
if False:
    k = 5  # Number of folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # do K-Fold cross-validation
    for idx, (train_index, val_index) in enumerate(kf.split(rest_data)):

        print(f"Fold {idx + 1}/{k}")

        # get train and test dates
        dates_train = [rest_data[i] for i in train_index]
        dates_val = [rest_data[i] for i in val_index]

        # convert to features for train data
        X_train , y_train = get_data_from_dates(dates=dates_train, df=df_data)

        # convert to features for validation data
        X_val, y_val = get_data_from_dates(dates=dates_val, df=df_data)

        # create a model
        model = create_model(n_features=df_data.shape[1], out_shape=1)

        # fit model
        history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(X_val, y_val), verbose=0)

        # plot learning
        plot_history(history=history)


# real learning and testing
if True:
    print("Test Run")
    # create data
    X_train, y_train = get_data_from_dates(dates=rest_data, df=df_data)
    X_test, y_test = get_data_from_dates(dates=test_data, df=df_data)

    # create model
    model = create_model(n_features=df_data.shape[1], out_shape=1)
    model.summary()
    # fit model
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(X_test, y_test), verbose=0)

    # plot learning
    plot_history(history=history)


# for iterative learning
def iterative_learning(old_data: pd.DataFrame, new_data: pd.DataFrame, model: keras.Sequential):
    """Small incremental learning with new data."""
    # all dates for new learning
    all_dates = list(old_data.index) + list(new_data.index)

    # add new data to old data to use it later
    old_data = pd.concat([old_data, new_data])
    
    # get data and fit model with new data
    X_data, y_data = get_data_from_dates(dates=all_dates, df=old_data)
    n_epochs = 20
    model.fit(X_data, y_data, batch_size=BATCH_SIZE, epochs=n_epochs, validation_data=(X_test, y_test), verbose=0)


# for long-distance continuous prediction
def continuous_prediction(df: pd.DataFrame, new_data: pd.DataFrame, model: keras.Sequential) -> np.ndarray:
    """Predict for a continuous time period."""
    # copy the DataFrame as we are going to change it
    df_copy = df.copy()

    # start prediction
    results: list[float] = list()
    # for each date make a prediction and add it in the df_copy
    for date in new_data.index:
        # read line with new features
        new_line = new_data.loc[date].copy()

        # add this line into the df_copy
        df_copy.loc[date] = new_line

        # get fold for this date
        X_data, _ = get_data_from_dates(dates=[date], df=df_copy)

        # call model for this one date
        y_data = model.predict(X_data, verbose=0).squeeze()

        # save this result in df_copy and for return
        df_copy.loc[date, "Target"] = y_data
        results.append(y_data)

    return np.array(results)


# a bit smarter continuous prediction
def smart_continuous_prediction(df: pd.DataFrame, new_data: pd.DataFrame, model: keras.Sequential) -> np.ndarray:
    """Continuous prediction with incremental learning."""
    # get dates and group it by months
    df_dates = pd.DataFrame({'Date String': list(new_data.index)})
    df_dates['Date'] = pd.to_datetime(df_dates['Date String'])
    df_dates['Month'] = df_dates['Date'].dt.to_period('M')
    df_dates.sort_values('Month', inplace=True)
    df_dates = df_dates.groupby('Month')

    # new do some iterative learning
    results = np.array(list())
    for _, month_df in df_dates:
        # get temporal data 
        temporal_data = new_data.loc[month_df["Date String"]]
        # get predict for quarter
        predict = continuous_prediction(df=df, new_data=temporal_data, model=model)
        # save this prediction
        results = np.hstack([results, predict])
        # incremental learning
        iterative_learning(old_data=df, new_data=temporal_data, model=model)

    return results

# simple prediction using daily data
y_predict_simple = model.predict(X_test, verbose=0)

# for continuous prediction using your own predicitions
new_data = df_data.loc[test_data].copy()
df_data.drop(index=test_data, inplace=True)

print("Continuous Prediction")
y_predict_continuous = smart_continuous_prediction(df=df_data, new_data=new_data, model=model)
# draw prediction results
plt.figure(figsize=(6, 10))
plt.plot(test_data, y_test, color="orange", label="Real Sales")
plt.plot(test_data, y_predict_simple, color="blue", label="Simple Predicted Sales")
plt.plot(test_data, y_predict_continuous, color="green", label="Continuous Predicted Sales")

# configurate plots
plt.title("Long-Time Prediction")
plt.xlabel("Days")
plt.ylabel("Sales")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()