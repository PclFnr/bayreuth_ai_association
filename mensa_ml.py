# Basic packages
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

# Machine learning packages
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import keras

# Constants
PATH_TO_DATA = Path(__file__).parent.joinpath("data/extended_data.csv")

LOSS = keras.losses.LogCosh()  # Loss function for the model

N_DAYS = 21  # Time frame of days
N_FUTURES = 30 #  Days to predict for future

N_YEARS = 3  # Number of years to consider
N_PADDING = 2 * (N_YEARS - 1) * N_DAYS + N_YEARS * (N_FUTURES + 1) + N_DAYS  # Padding size

BATCH_SIZE = 15  # Batch size for training
N_EPOCHS = 20  # Number of training epochs

metrics = ["loss", "mae"]
targets = ["Target Coffee" ,"Target Milk Coffee", "Target Cocoa", "Target Tee" ,"Target Coffee Time"]

# Load dataset
df_data = pd.read_csv(PATH_TO_DATA, index_col=0)
df_data.drop(columns=["Student Tax", "Worker Tax", "Guest Tax"], inplace=True)

def create_model(n_features: int, out_shape: int) -> keras.Sequential:
    """
    Creates and compiles a Keras sequential model for time series prediction.

    Parameters:
    -----------
        n_features: int
            The number of input features for the model.
        out_shape: int
            The shape of the output layer.

    Returns:
    --------
        keras.Sequential: A compiled Keras model.
    """
    model = keras.Sequential()

    # Input layer
    model.add(
        keras.layers.InputLayer(shape=(None, n_features))
    )

    # Dense encoder layer
    model.add(
        keras.layers.Dense(
            units=128,
            activation="tanh",
            kernel_regularizer=keras.regularizers.L2(l2=0.01),
            bias_regularizer=keras.regularizers.L2(l2=0.1),
            use_bias=True,
        )
    )

    # LSTM layer for sequence learning
    model.add(
        keras.layers.LSTM(
            units=256,  # Number of memory neurons
            return_sequences=True,  # Sequences for continuous prediction
            kernel_regularizer=keras.regularizers.L2(l2=0.01),
            recurrent_regularizer=keras.regularizers.L2(l2=0.01),
            use_bias=False,
        )
    )

    # Dense decoder layer
    model.add(
        keras.layers.Dense(
            units=out_shape,
            activation="linear",
            kernel_regularizer=keras.regularizers.L2(l2=0.01),
            use_bias=False,
        )
    )

    # Leaky ReLU activation
    model.add(keras.layers.LeakyReLU(negative_slope=0.02))

    # Cut the output so that we need only a limited portion of the resulting sequence
    model.add(keras.layers.Lambda(lambda x: x[:, -N_FUTURES:, :]))

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=LOSS,
        metrics=["mae"]
    )

    return model


def string_to_datetime(date_string: str) -> datetime:
    """
    Converts a date string to a datetime object.

    Parameters:
    -----------
        date_string: str
            The date string in the format '%Y-%m-%d'.

    Returns:
    --------
        datetime: The corresponding datetime object.
    """
    return datetime.strptime(date_string, "%Y-%m-%d")


def datetime_to_string(date: datetime) -> str:
    """
    Converts a datetime object to a string.

    Parameters:
    -----------
        date (datetime): The datetime object to convert.

    Returns:
    --------
        str: The date string in the format '%Y-%m-%d'.
    """
    return datetime.strftime(date, "%Y-%m-%d")


def get_data_advance(date_string: str, n_days: int, n_future: int) -> list[str]:
    """
    Generate a list of date strings for the target date, including days in the past year 
    and the preceding week.

    Parameters:
    -----------
        date_string: str 
            The target date as a string in the format '%Y-%m-%d'.
        n_days: int
            The number of days before and after the past year's date to include.
        n_future: int
            The number of days to predict for future.

    Returns:
    --------
        list[str]: A sorted list of date strings in ascending order.
    """
    # Convert date string to datetime
    target_date = string_to_datetime(date_string=date_string)

    # Dates from the last week
    past_this_year = [(target_date - timedelta(days=i)).date() for i in range(1, n_future + n_days + 1)][::-1]

    # Dates one year ago
    frame = list()
    for i in range(1, N_YEARS):
        year_ago = target_date - relativedelta(years=i)
        # days before target a year ago
        past_before = [(year_ago - timedelta(days=i)).date() for i in range(1, n_future + n_days + 1)][::-1]
        # days after required period a year ago
        past_after = [(year_ago + timedelta(days=i)).date() for i in range(0, n_days + 1)]
        # add it
        frame += (past_before + past_after + past_this_year)

    # Combine and sort dates
    frame = sorted(frame)

    return [datetime_to_string(date) for date in frame]


def get_data_future(date_string: str, n_future: int) -> list[str]:
    """
    Generate a list of date strings for the target date, including days in the future

    Parameters:
    -----------
        date_string: str 
            The target date as a string in the format '%Y-%m-%d'.
        n_future: int
            The number of days to predict for future.

    Returns:
    --------
        list[str]: A sorted list of date strings in ascending order.
    """
    # Convert date string to datetime
    target_date = string_to_datetime(date_string=date_string)

    # days for the whole requested period
    dates_range = sorted([(target_date - timedelta(days=i)).date() for i in range(0, n_future)])

    return [datetime_to_string(date) for date in dates_range]


def get_frame_features(frame: list[str], df: pd.DataFrame) -> np.ndarray:
    """
    Extract features for the given list of dates from the DataFrame.

    Parameters:
    -----------
        frame: list[str]
            List of date strings to fetch data for.
        df: pd.DataFrame
            The DataFrame containing the data indexed by date.

    Returns:
    --------
        np.ndarray: A matrix where each row corresponds to features for a date.
    """
    features = list()
    for date in frame:
        if date in df.index:
            day_features = list(df.loc[date])
            features.append(day_features)

    return np.array(features) if features else np.zeros((0, df.shape[1]))


def get_fold_for_date(date_str: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and target data for a specific date.

    Parameters:
    -----------
        date_str: str
            The date string in the format '%Y-%m-%d'.
        df: pd.DataFrame
            The DataFrame containing the data.

    Returns:
    --------
        tuple[np.ndarray, np.ndarray]: A tuple containing feature matrix `X_data` 
                                       and target array `y_data`.
    """
    # Extract target data for future
    interesting_dates = get_data_future(date_str, n_future=N_FUTURES)
    y_data = df.loc[interesting_dates, targets].to_numpy()
    # copy interesting part from the dataframe and make them zeros because we do not know them
    rows = df.loc[interesting_dates].copy()
    rows[targets] = 0.0
    y_features = rows.to_numpy()

    # Extract frame features from the past
    frame = get_data_advance(date_string=date_str, n_days=N_DAYS, n_future=N_FUTURES)
    # split frame in known and unknown parts
    frame = sorted(list(set(frame).difference(set(interesting_dates))))
    X_data = get_frame_features(frame=frame, df=df)
    X_data = np.vstack([X_data, y_features])

    return X_data, y_data


def padding(n_max: int, arr: np.ndarray) -> np.ndarray:
    """
    Apply zero-padding to an array to align its number of rows with `n_max`.

    Parameters:
    -----------
        n_max:int
            The target number of rows after padding.
        arr: np.ndarray
            The array to pad.

    Returns:
    --------
        np.ndarray: The padded array.
    """
    rows_to_add = n_max - arr.shape[0]
    matrix = np.zeros(shape=(rows_to_add, arr.shape[1]))
    return np.vstack([matrix, arr])


def get_data_from_dates(dates: list[str], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of dates into padded feature and target matrices.

    Parameters:
    -----------
        dates: list[str]
            List of date strings in the format '%Y-%m-%d'.
        df: pd.DataFrame
            The DataFrame containing the data.

    Returns:
    --------
        tuple[np.ndarray, np.ndarray]: A tuple containing the feature matrix `X_list`
                                       and the target matrix `y_list`.
    """
    X_list, y_list = [], []
    for date_str in dates:
        X, y = get_fold_for_date(date_str=date_str, df=df)
        X = padding(N_PADDING, X)
        X_list.append(X)
        y_list.append(y)

    return np.array(X_list), np.array(y_list)


def plot_history(history):
    """
    Plot the training and validation loss curves from the model's training history.

    Parameters:
    -----------
        history: The training history object returned by `model.fit`.

    Returns:
    --------
        None
    """

    _, axs = plt.subplots(nrows=len(metrics), ncols=1, figsize=(18, 10))

    for metric, ax in zip(metrics, axs):
        val_key = "val_" + metric

        ax.plot(history.history[metric], label=f"Training {metric}", color="blue")
        ax.plot(history.history[val_key], label=f"Validation {metric}", color="orange")
        ax.set_title(f"Learning Curves for {metric}")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.show()

# cut off the earliest date possbile for training
EARLIEST_DATE_POSSIBLE = "2017-10-01"

# split to train, test, validation
BORDER_DATE = "2023-01-01"
test_data = df_data.index[df_data.index >= BORDER_DATE].to_list() # dates that will not be shown to models

# select test_samples that have no zero results
zero_mask = (df_data.loc[test_data, targets] == 0).all(axis=1)
non_zero_dates = [date for mask, date in zip(zero_mask, test_data, strict=True) if not mask]

# other stuff must be splitted
rest_data = list(set(df_data.index).difference(set(test_data))) # all other dates for training
rest_data = [date for date in rest_data if date >= EARLIEST_DATE_POSSIBLE]


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
        model = create_model(n_features=df_data.shape[1], out_shape=len(targets))

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
    model = create_model(n_features=df_data.shape[1], out_shape=len(targets))
    model.summary()
    # fit model
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(X_test, y_test), verbose=1)

    # plot learning
    plot_history(history=history)


# for iterative learning
def iterative_learning(
        old_data: pd.DataFrame, 
        new_data: pd.DataFrame, 
        model: keras.Sequential
) -> tuple[pd.DataFrame, keras.Sequential]:
    """
    Perform small incremental learning for a model using new data.

    Parameters:
    -----------
        old_data: pd.DataFrame
            The original dataset used for training.
        new_data: pd.DataFrame
            The new dataset for incremental learning.
        model: keras.Sequential
            The pre-trained Keras model..

    Returns:
    --------
        pd.DataFrame
            The concatenation of original and new DataFrames.
        keras.Sequential
            The relearned model.
    """
    # all dates for new learning
    new_dates = list(new_data.index)

    # add new data to old data to use it later
    updated_df = pd.concat([old_data, new_data], ignore_index=False)
    
    # get data
    X_data, y_data = get_data_from_dates(dates=new_dates, df=updated_df)

    # create new model and fit it
    # model = create_model(n_features=updated_df.shape[1], out_shape=1)
    model.fit(X_data, y_data, batch_size=BATCH_SIZE, epochs=N_EPOCHS, verbose=1)

    return updated_df, model


def continuous_prediction(
    df: pd.DataFrame, 
    new_data: pd.DataFrame, 
    model: keras.Sequential
) -> np.ndarray:
    """
    Predict values for a continuous time period using a pre-trained model.

    Parameters:
    -----------
        df: pd.DataFrame
            Historical data used for generating features.
        new_data: pd.DataFrame
            New data for prediction, indexed by date.
        model: keras.Sequential
            The pre-trained Keras model.

    Returns:
    --------
        np.ndarray: Array of predicted target values for the new data.
    """
    # Get range of dates that must be predicted
    date_range = sorted(list(new_data.index))
    next_start = date_range[0]

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # List to store prediction results
    results = np.zeros(shape=(0, len(targets)))

    # make step-by-step predictions
    while len(new_data) > 0:
        # calculate data borders
        start = string_to_datetime(next_start)
        end = start + timedelta(days=(N_FUTURES - 1))
        next_date = end + timedelta(days=1)

        # convert to strings
        next_start = datetime_to_string(start)
        next_end = min(datetime_to_string(end), new_data.index.max()) # because we can get more than max available date
        next_date_str = datetime_to_string(next_date)

        # select subset of appropriate dates
        indexes = sorted([day for day in new_data.index if (next_start <= day and day <= next_end)])
        tmp_df = new_data.loc[indexes].copy()
        new_data.drop(index=indexes, inplace=True)

        # memorize amount of this new data
        n_days = len(tmp_df)

        # add this data into the copy df
        df_copy = pd.concat([df_copy, tmp_df], ignore_index=False) 

        # Generate input features for the model based on the updated DataFrame
        X_data, _ = get_data_from_dates(dates=[next_end], df=df_copy)
        # Make predictions using the model
        y_data = model.predict(X_data, verbose=0)[0, -n_days:, :]
        # Update the DataFrame with the predicted target value
        df_copy.loc[indexes, targets] = y_data

        # Store the result for the return value
        results = np.vstack([results, y_data])

        # Switch to the next date
        next_start = next_date_str

    return results


def smart_continuous_prediction(
    df: pd.DataFrame, 
    new_data: pd.DataFrame, 
    model: keras.Sequential,
    time_unit: str = 'M'
) -> np.ndarray:
    """
    Perform smarter continuous prediction with incremental learning.

    Parameters:
    -----------
        df: pd.DataFrame
            Historical data used for feature generation.
        new_data: pd.DataFrame
            New data for prediction, indexed by date.
        model: keras.Sequential
            The pre-trained Keras model.
        time_unit: str = "M"
            Time grouping unit for incremental updates (e.g., 'M' for month, 'Q' for quarter).

    Returns:
    --------
        np.ndarray: Array of predicted target values for the new data.
    """
    # Prepare dates for grouping
    df_dates = pd.DataFrame({'Date String': list(new_data.index)})
    df_dates['Date'] = pd.to_datetime(df_dates['Date String'])
    df_dates['Group'] = df_dates['Date'].dt.to_period(time_unit)
    df_dates.sort_values(by=['Group', 'Date'], inplace=True)
    grouped_dates = df_dates.groupby('Group')

    # Initialize an empty results array
    results = np.zeros(shape=(0, len(targets)))

    # Perform predictions and incremental learning by group
    for _, group_df in grouped_dates:
        # Get data for the current time group
        temporal_data = new_data.loc[group_df["Date String"]].copy()

        # Predict for the current group
        predictions = continuous_prediction(df=df, new_data=temporal_data.copy(), model=model)
        results = np.vstack([results, predictions])

        # Perform incremental learning with the current group
        df, model = iterative_learning(old_data=df, new_data=temporal_data, model=model)

    return results

# Perform simple prediction using the model
y_predict_simple = model.predict(X_test, verbose=0)

# Prepare new data for continuous prediction
new_data = df_data.loc[test_data].copy()

# Remove test data from the original DataFrame to avoid overlap
truth_df = df_data.loc[test_data, targets].copy()
df_data.drop(index=test_data, inplace=True)

# Try just a simple continuous prediciton witho
y_simple_continuous_predict = continuous_prediction(df=df_data, new_data=new_data.copy(), model=model)

# Smart Continuous prediction with relearning
if True:
    print("Continuous Prediction")
    y_predict_continuous = smart_continuous_prediction(df=df_data, new_data=new_data, model=model, time_unit="M")

# create data for plotting from all predictions
simple_predicted_df = pd.DataFrame(y_predict_simple[:, -1, :].squeeze(), columns=targets, index=test_data)
smart_predicted_df = pd.DataFrame(y_predict_continuous, columns=targets, index=test_data)
continuous_predicted_df = pd.DataFrame(y_simple_continuous_predict, columns=targets, index=test_data)

# Calculate MAE 
mae = mean_absolute_error(y_true=truth_df.values, y_pred=simple_predicted_df.values)
print("Simple Prediction MAE: ", round(mae, 3))
# Calculate MAE for simple continuous prediction
mae = mean_absolute_error(y_true=truth_df.values, y_pred=continuous_predicted_df.values)
print("Continuous Prediction MAE: ", round(mae, 3))
# Calculate MAE for adaptive continuous prediction
mae = mean_absolute_error(y_true=truth_df.values, y_pred=smart_predicted_df.values)
print("Adaptive Learning MAE: ", round(mae, 3))

def group_by_date(df: pd.DataFrame, time_unit: str = "M") -> pd.DataFrame:
    """Group solution by time unit. Possible units: D - Day, M - Month, Q - Quarter, Y - Year."""
    # Prepare dates for grouping
    df_dates = pd.DataFrame({'Date String': list(df.index)})
    df_dates['Date'] = pd.to_datetime(df_dates['Date String'])
    df_dates['Group'] = df_dates['Date'].dt.to_period(time_unit)
    df_dates[targets] = df.loc[df_dates['Date String'], targets].values.copy()
    df_dates.sort_values(by=['Group', 'Date'], inplace=True)
    grouped_dates = df_dates.groupby('Group')[targets].sum()

    return grouped_dates


TIME_UNIT = "W"
INTERVAL = 1 if TIME_UNIT != "D" else 7

# group all the results by unit
truth_df = group_by_date(df=truth_df, time_unit=TIME_UNIT)
simple_predicted_df = group_by_date(df=simple_predicted_df, time_unit=TIME_UNIT)
smart_predicted_df = group_by_date(df=smart_predicted_df, time_unit=TIME_UNIT)
continuous_predicted_df = group_by_date(df=continuous_predicted_df, time_unit=TIME_UNIT)
# get axis values for plotting
x_values = truth_df.index.map(lambda value: str(value))

for target in targets:
    # Plot the prediction results
    plt.figure(figsize=(6, 10))

    # Plot real sales data
    plt.plot(x_values, truth_df[target], color="orange", label="Real Sales")

    # Plot simple predictions
    plt.plot(x_values, simple_predicted_df[target], color="blue", label="Simple Predicted Sales")

    # Plot smart continuous prediction
    plt.plot(x_values, smart_predicted_df[target], color="green", label="Smart Continuous Predicted Sales")

    # Plot smart continuous prediction
    plt.plot(x_values, continuous_predicted_df[target], color="red", label="Simple Continuous Predicted Sales")

    # Reporting
    print(100 * "=")
    # Calculate MAE for continuous prediction of one target.
    mae = mean_absolute_error(y_true=truth_df[target].values, y_pred=continuous_predicted_df[target].values)
    print(f"Continuous MAE for {target}: ", round(mae, 3))
    # Calculate MAE for smart continuous prediction of one target.
    mae = mean_absolute_error(y_true=truth_df[target].values, y_pred=smart_predicted_df[target].values)
    print(f"Adaptive Learning MAE for {target}: ", round(mae, 3))
    print(100 * "=")

    # Configure the plot
    plt.title(f"Long-Time Prediction {target}")
    plt.xlabel("Days")
    plt.ylabel("Sales")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=INTERVAL))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()
