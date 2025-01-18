import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error

from mensa_ml import (
    create_model,
    get_data_from_dates,
    group_by_date,
    plot_history,
    educated_guess,
    continuous_prediction,
    smart_continuous_prediction,
    BATCH_SIZE,
    N_EPOCHS,
    PATH_TO_DATA,
    targets,
    N_FEATURES
)

# Load dataset
df_data = pd.read_csv(PATH_TO_DATA, index_col=0)
df_data.drop(columns=["Year", "Student Tax", "Worker Tax", "Guest Tax"], inplace=True)

# train-val-test-split
# train_data, val_data, test_data = train_val_test_split(dates=list(df_data.index), n_val_days=91, n_test_days=109)

# split to train, test, validation
BORDER_DATE = "2023-01-01"
test_data = df_data.index[df_data.index >= BORDER_DATE].to_list() # dates that will not be shown to models

# select test_samples that have no zero results
zero_mask = (df_data.loc[test_data, targets] == 0).all(axis=1)
non_zero_dates = [date for mask, date in zip(zero_mask, test_data, strict=True) if not mask]

# other stuff must be splitted
train_data = list(set(df_data.index).difference(set(test_data))) # all other dates for training

# cut off the earliest date possbile for training
EARLIEST_DATE_POSSIBLE = "2017-10-01"
train_data = sorted([date for date in train_data if date >= EARLIEST_DATE_POSSIBLE])

if True:
    print("Test Run")
    # create data
    X_train, y_train = get_data_from_dates(dates=train_data, df=df_data)
    X_test, y_test = get_data_from_dates(dates=test_data, df=df_data)

    # create model
    model = create_model(n_features=N_FEATURES, out_shape=len(targets))
    model.summary()
    # fit model
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(X_test, y_test), verbose=1)

    # plot learning
    plot_history(history=history)

# Perform simple prediction using the model
y_predict_simple = model.predict(X_test, verbose=0)

# Remove test data from the original DataFrame to avoid overlap
truth_df = df_data.loc[test_data, targets].copy()

# Make a simple AVG guess
educated_guess_predict = np.array(
    [
        educated_guess(date, df_data) for date in test_data
    ]
)

# Prepare new data for continuous prediction
new_data = df_data.loc[test_data].copy()
df_data.drop(index=test_data, inplace=True)

# Try just a simple continuous prediciton witho
y_simple_continuous_predict = continuous_prediction(df=df_data, new_data=new_data.copy(), model=model)

# Smart Continuous prediction with relearning
if True:
    print("Continuous Prediction")
    y_predict_continuous = smart_continuous_prediction(df=df_data, new_data=new_data, model=model, time_unit="Q")

# create data for plotting from all predictions
simple_predicted_df = pd.DataFrame(y_predict_simple[:, -1, :].squeeze(), columns=targets, index=test_data)
smart_predicted_df = pd.DataFrame(y_predict_continuous, columns=targets, index=test_data)
continuous_predicted_df = pd.DataFrame(y_simple_continuous_predict, columns=targets, index=test_data)
educated_guess_df = pd.DataFrame(educated_guess_predict, columns=targets, index=test_data)

# Calculate MAE 
mae = mean_absolute_error(y_true=truth_df.values, y_pred=simple_predicted_df.values)
print("Simple Prediction MAE: ", round(mae, 3))
# Calculate MAE for simple continuous prediction
mae = mean_absolute_error(y_true=truth_df.values, y_pred=continuous_predicted_df.values)
print("Continuous Prediction MAE: ", round(mae, 3))
# Calculate MAE for adaptive continuous prediction
mae = mean_absolute_error(y_true=truth_df.values, y_pred=smart_predicted_df.values)
print("Adaptive Learning MAE: ", round(mae, 3))

# target_date
target_date = "2024-31-12"

new_df = pd.DataFrame

# for plotting
TIME_UNIT = "W" # possible units: D - Day, W - week, M - month, Q - quarter.
match TIME_UNIT:
    case "D":
        INTERVAL = 7
    case "W":
        INTERVAL = 3
    case _:
        INTERVAL = 1

# group all the results by unit
truth_df = group_by_date(df=truth_df, time_unit=TIME_UNIT)
simple_predicted_df = group_by_date(df=simple_predicted_df, time_unit=TIME_UNIT)
smart_predicted_df = group_by_date(df=smart_predicted_df, time_unit=TIME_UNIT)
continuous_predicted_df = group_by_date(df=continuous_predicted_df, time_unit=TIME_UNIT)
educated_guess_df = group_by_date(df=educated_guess_df, time_unit=TIME_UNIT)
# get axis values for plotting
x_values = truth_df.index.map(lambda value: str(value))

for target in targets:
    # Plot the prediction results
    plt.figure(figsize=(6, 10))

    # Plot real sales data
    plt.plot(x_values, truth_df[target], color="orange", label="Real Sales")

    # Plot smart continuous prediction
    plt.plot(x_values, smart_predicted_df[target], color="green", label="Smart Continuous Predicted Sales")

    # Plot smart continuous prediction
    plt.plot(x_values, continuous_predicted_df[target], color="red", label="Simple Continuous Predicted Sales")

    # Plot educated guess prediction
    plt.plot(x_values, educated_guess_df[target], color="black", label="Educated Guess")

    # Reporting
    print(100 * "=")
    # Calculate MAE for continuous prediction of one target.
    mae = mean_absolute_error(y_true=truth_df[target].values, y_pred=continuous_predicted_df[target].values)
    print(f"Continuous MAE for {target}: ", round(mae, 3))
    # Calculate MAE for smart continuous prediction of one target.
    mae = mean_absolute_error(y_true=truth_df[target].values, y_pred=smart_predicted_df[target].values)
    print(f"Adaptive Learning MAE for {target}: ", round(mae, 3))
    # Calculate MAE for educated guess
    mae = mean_absolute_error(y_true=truth_df[target].values, y_pred=educated_guess_df[target].values)
    print(f"Educated Guess MAE for {target}: ", round(mae, 3))
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