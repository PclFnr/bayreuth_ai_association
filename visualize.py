import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from src.mensa_ml import group_by_date, targets
from sklearn.metrics import mean_absolute_error
import numpy as np

# read old data
continuous_predicted_df = pd.read_csv("predictions/2024-test-predict.csv", index_col=0)
smart_predicted_df = pd.read_csv("predictions/2024-test-predict-iterative.csv", index_col=0)
truth_df = pd.read_csv("predictions/2024-true-data.csv", index_col=0)
educated_guess_df = pd.read_csv("predictions/2024-educated-guess.csv", index_col=0)
future_df = pd.read_csv("predictions/2025-predict.csv", index_col=0)

def plot_error_distribution(error_data: np.ndarray, target: str, case: str):
    """Plot histogram of the error to check bias and variance."""
    # Calculate mean and standard deviation
    mean = np.mean(error_data)
    std = np.std(error_data)

    # Create histogram
    plt.hist(error_data, bins=25, density=True, alpha=0.6, color='b')

    # Plot mean line
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label='Mean')

    # Plot standard deviation lines
    plt.axvline(mean + std, color='green', linestyle='dashed', linewidth=2, label='Mean + 1 Std Dev')
    plt.axvline(mean - std, color='green', linestyle='dashed', linewidth=2)

    # Add labels and title
    plt.xlabel('Error Value')
    plt.ylabel('Frequency')
    plt.title(f'Error histogram in {case} for {target}')
    plt.legend()

    plt.show()

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
smart_predicted_df = group_by_date(df=smart_predicted_df, time_unit=TIME_UNIT)
continuous_predicted_df = group_by_date(df=continuous_predicted_df, time_unit=TIME_UNIT)
educated_guess_df = group_by_date(df=educated_guess_df, time_unit=TIME_UNIT)
future_df = group_by_date(df=future_df, time_unit=TIME_UNIT)
# get axis values for plotting
x_values = truth_df.index.map(lambda value: str(value))

x_values_future = future_df.index.map(lambda value: str(value))
# get axis values for plotting

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
    # 

    # Configure the plot
    plt.title(f"Long-Time Prediction {target}")
    plt.xlabel("Days")
    plt.ylabel("Sales")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=INTERVAL))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # plot BIAS - VARIENCE Tradeoff
    continuous_error = truth_df[target].values - continuous_predicted_df[target].values
    smart_continuous_error = truth_df[target].values - smart_predicted_df[target].values

    # plot the results
    plot_error_distribution(continuous_error, target=target, case="Continuous Precition")
    plot_error_distribution(smart_continuous_error, target=target, case="Smart Continuous Precition")
    

for target in targets:
    # Plot the prediction results
    plt.figure(figsize=(6, 10))

    # Plot smart continuous prediction
    plt.plot(x_values_future, future_df[target], color="red", label="Simple Continuous Predicted Sales")

    # Configure the plot
    plt.title(f"Long-Time Prediction {target}")
    plt.xlabel("Days")
    plt.ylabel("Sales")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=INTERVAL))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()