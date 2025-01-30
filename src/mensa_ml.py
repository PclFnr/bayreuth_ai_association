# Basic packages
import pandas as pd
from matplotlib import pyplot as plt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from pathlib import Path
from src.utils import string_to_datetime, datetime_to_string

# Machine learning packages
import keras
import keras_tuner as kt
from keras_tuner.src.engine.hyperparameters.hyperparameters import HyperParameters

# Constants
PATH_TO_DATA = Path(__file__).parent.parent.joinpath("data/extended_data.csv")

LOSS = keras.losses.LogCosh()  # Loss function for the model

N_DAYS = 21  # Time frame of days
N_FUTURES = 30 #  Days to predict for future

N_YEARS = 2  # Number of years to consider
N_PADDING = 2 * (N_YEARS - 1) * N_DAYS + N_YEARS * N_FUTURES + N_DAYS  # Padding size

BATCH_SIZE = 25  # Batch size for training
N_EPOCHS = 12 # Number of training epochs

metrics = ["loss", "mae"]
targets = ["Target Coffee", "Target Milk Coffee", "Target Cocoa", "Target Tee", "Target Coffee Time"]

def create_model(n_features: int, out_shape: int, n_futures: int, hp: HyperParameters = None) -> keras.Sequential:
    """
    Creates and compiles a Keras sequential model with tunable hyperparameters.

    Parameters:
    -----------
        hp: HyperParameters
            Hyperparameter object for tuning.
        n_features: int
            The number of input features for the model.
        out_shape: int
            The shape of the output layer.
        n_futures: int
            Amount of days to predict in the future.

    Returns:
    --------
        keras.Sequential: A compiled Keras model.
    """
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.InputLayer(shape=(None, n_features)))

    # Encoder Layer
    model.add(
        keras.layers.Dense(
            units=128, #hp.Int("dense_units", min_value=64, max_value=192, step=64),
            activation="tanh", # hp.Choice("enc_actv", values=["tanh", "relu"])
            kernel_regularizer=keras.regularizers.L2(l2=1e-3), #keras.regularizers.L2(l2=hp.Float("dense_enc_l2", min_value=1e-3, max_value=1e-2, sampling="log"))
            use_bias=False,
        )
    )

    # Mask useless values to skip zero frames
    keras.layers.Masking(mask_value=0.0)

    # LSTM layer for sequence learning
    model.add(
        keras.layers.LSTM(
            units=128, # hp.Int("lstm_units", min_value=128, max_value=256, step=128),
            return_sequences=True,
            kernel_regularizer=keras.regularizers.L2(l2=1e-5), # keras.regularizers.L2(l2=hp.Float("lstm_l2", min_value=1e-7, max_value=1e-5, sampling="log")),
            recurrent_regularizer=keras.regularizers.L2(l2=1e-5), # keras.regularizers.L2(l2=hp.Float("lstm_rec_l2", min_value=1e-7, max_value=1e-5, sampling="log")),
            use_bias=False,
        )
    )

    # Dense decoder layer
    model.add(
        keras.layers.Dense(
            units=out_shape,
            activation="linear",
            kernel_regularizer=keras.regularizers.L2(l2=1e-2), # keras.regularizers.L2(hp.Float("dense_out_l2", 1e-4, 1e-2, sampling="log")),
            use_bias=False,
        )
    )

    # Leaky ReLU activation
    model.add(keras.layers.LeakyReLU(negative_slope=0.2))

    # Output slicing
    model.add(keras.layers.Lambda(lambda x: x[:, -n_futures:, :]))

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=LOSS,
        metrics=["mae"],
    )

    return model


def tune_model(
        n_features: int, 
        out_shape: int, 
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        max_trials: int = 1, 
        executions_per_trial: int = 1):
    """
    Tunes hyperparameters using Keras Tuner.
    
    Parameters:
    -----------
        n_features: int
            Number of input features.
        out_shape: int
            Shape of the output layer.
        max_trials: int
            Maximum number of trials for tuning.
        executions_per_trial: int
            Number of executions per trial for reliability.
        X_train: np.ndarray,
            Train features.
        y_train: np.ndarray,
            Train targets.
        X_test: np.ndarray,
            Test features.
        y_test: np.ndarray
            Test targets.
    
    Returns:
    --------
        tuner: Hyperband
            The tuner object with results.
    """
    tuner = kt.RandomSearch(
        lambda hp: create_model(n_features, out_shape, N_FUTURES, hp=hp),
        objective="val_mae",
        max_trials=max_trials,
        directory="hyperparameter_tuning",
        project_name="time_series_model",
    )

    # Search for the best hyperparameters
    tuner.search(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)],
    )

    return tuner


def get_data_advance(date_string: str, n_days: int, n_years: int, n_future: int) -> list[str]:
    """
    Generate a list of date strings for the target date, including days in the past year 
    and the preceding week.

    Parameters:
    -----------
        date_string: str 
            The target date as a string in the format '%Y-%m-%d'.
        n_days: int
            The number of days before and after the past year's date to include.
        n_years: int
            The number of years to be considered in the past.
        n_future: int
            The number of days to predict for future.

    Returns:
    --------
        list[str]: A sorted list of date strings in ascending order.
    """
    # Convert date string to datetime
    target_date = string_to_datetime(date_string=date_string)

    # Dates from the last week
    past_this_year = [(target_date - timedelta(days=i)).date() for i in range(0, n_future + n_days)][::-1]

    # Dates one year ago
    frame = list()
    for i in range(1, n_years):
        year_ago = target_date - relativedelta(years=i)
        # days before target a year ago
        past_before = [(year_ago - timedelta(days=i)).date() for i in range(1, n_future + n_days)][::-1]
        # days after required period a year ago
        past_after = [(year_ago + timedelta(days=i)).date() for i in range(0, n_days)]
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
    frame = get_data_advance(date_string=date_str, n_days=N_DAYS, n_years=N_YEARS, n_future=N_FUTURES)
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

    # Add time marker for each step.
    matrix = np.vstack([matrix, arr])

    return matrix


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


def train_val_test_split(
    dates: list[str], 
    n_val_days: int, 
    n_test_days: int
) -> tuple[list[str], list[str], list[str]]:
    """
    Split available date in train, validation and test subsets without intersections.

    Parameters
    ----------
    dates: list[str]
        All available dates in ISO format "YYYY-mm-dd".
    n_val_days: int
        Amount of days in the validation set.
    n_test_days: int
        Amount of days in the test set.

    Returns
    -------
    tuple[list[str], list[str], list[str]]:
        Indices of sets in the following order:
            - list[str]: List of training dates.
            - list[str]: List of validation dates.
            - list[str]: List of test dates.
    """
    # check for correctness
    assert len(dates) > n_val_days + n_test_days, "Amount of dates is less than amount of validation and test sets."

    # sort dates
    sorted_dates = sorted(dates)

    # get subsets
    train_data = sorted_dates[0:-(n_val_days + n_test_days)]
    val_data = sorted_dates[-(n_val_days + n_test_days):-n_test_days]
    test_data = sorted_dates[-n_test_days:]

    return train_data, val_data, test_data

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
    # some preprocessing for correct memory selection
    min_date = string_to_datetime(old_data.index.min())
    min_date = (min_date + timedelta(days=N_FUTURES)).date()
    min_date_str = datetime_to_string(min_date)
    old_dates = [date for date in old_data.index if date >= min_date_str]

    # all dates for new learning
    memory_size = int(len(old_dates) * 0.1)
    memory_dates = list(np.random.choice(old_dates, size=memory_size, replace=False))
    all_dates = list(new_data.index) + memory_dates

    # add new data to old data to use it later
    updated_df = pd.concat([old_data, new_data], ignore_index=False)
    
    # get data
    X_data, y_data = get_data_from_dates(dates=all_dates, df=updated_df)

    # create new model and fit it
    # model = create_model(n_features=updated_df.shape[1], out_shape=1)
    model.fit(
        X_data, 
        y_data, 
        batch_size=BATCH_SIZE, 
        epochs=N_EPOCHS, 
        verbose=0,
    )

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

    # Create a copy of the DataFrame to avoid modifying the original one
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


def educated_guess(date_str: str, df: pd.DataFrame) -> np.ndarray:
    """
    Get an average amount of beverages for a date considering previous similar period as a base.

    Parameters
    ----------
    date_str: str
        Date to make a predict for.
    df: pd.DataFrame
        DataFrame as source of data. 

    Returns
    -------
    np.ndarray
        Prediction about beverages ["Target Coffee" ,"Target Milk Coffee", "Target Cocoa", "Target Tee" ,"Target Coffee Time"].
    """
    # check if it is a holiday or weekends then 0.0 for everything
    if df.loc[date_str, "is_holiday"] or (df.loc[date_str, ["x0_Friday", "x0_Monday", "x0_Thursday", "x0_Tuesday", "x0_Wednesday"]] == 0).all():
        return np.zeros(shape=len(targets))

    # get basic data
    is_summer, is_lecture_free = bool(df.loc[date_str, "is_summer"]), bool(df.loc[date_str, "is_lecture_free"])
    
    # split the date
    year_str, month_str, _ = tuple(date_str.split("-"))

    # define start and end of a semester lectures
    winter_lectures_end, summer_lecures_end = "02-10", "07-31"

    # define borders of a semester
    summer_start, summer_end = "04-01", "09-31"
    winter_start, winter_end = "10-01", "03-31"

    # summer and lecture free
    if is_summer and is_lecture_free:
        # get one year before
        year_before = str(int(year_str) - 1)
        # create bordering dates
        period_start = year_before + "-" + summer_lecures_end
        period_end = year_before + "-" + summer_end
    
    # summer with lectures
    elif is_summer and not is_lecture_free:
        # get one year before
        year_before = str(int(year_str) - 1)
        # create bordering dates
        period_start = year_before + "-" + summer_start
        period_end = year_before + "-" + summer_lecures_end
    
    # winter and lecture free
    elif not is_summer and is_lecture_free:
        # get one year before
        year_before = str(int(year_str) - 1)
        # create bordering dates
        period_start = year_before + "-" + winter_lectures_end
        period_end = year_before + "-" + winter_end

    # winter with lectures
    else:
        # if we are in march or before, then we need to go 2 years back
        if month_str < "04":
            # get one and two years before
            year_before = str(int(year_str) - 1)
            two_years_before = str(int(year_str) - 2)
            # create bordering dates
            period_start = two_years_before + "-" + winter_start
            period_end = year_before + "-" + winter_lectures_end

        # if we are in october and after, then only one year
        elif month_str > "09":
            # get one year before
            year_before = str(int(year_str) - 1)
            # create bordering dates
            period_start = year_before + "-" + winter_start
            period_end = year_str + "-" + winter_lectures_end
    
    # get dates for prediction
    time_period = [date for date in df.index if (period_start <= date <= period_end)]

    # get rid of holidays and dates where all sales are zero
    zero_mask = (df.loc[time_period, targets] == 0).all(axis=1)
    data = df.loc[time_period, targets][~zero_mask].values

    return np.mean(data, axis=0)


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
