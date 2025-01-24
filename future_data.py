import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

from src.utils import adjust_week
from sklearn.preprocessing import StandardScaler

from src.mensa_ml import (
    create_model,
    get_data_from_dates,
    group_by_date,
    continuous_prediction,
    BATCH_SIZE,
    N_EPOCHS,
    targets,
)

# split to train, test, validation
df_data = pd.read_csv("data/extended_data.csv", index_col=0)
df_data.drop(columns=["Year", "Student Tax", "Worker Tax", "Guest Tax"], inplace=True)
df_data["Week"] = df_data["Week"].apply(adjust_week)
N_FEATURES = df_data.shape[1]

scaler = StandardScaler()
df_data[["pressure", "humidity", "wind_deg"]] = scaler.fit_transform(df_data[["pressure", "humidity", "wind_deg"]].values)

# read future data
future_data = pd.read_csv("data/future_data.csv", index_col=0)
future_data.drop(columns=["Year", "Student Tax", "Worker Tax", "Guest Tax"], inplace=True)
# for new data
future_data["Week"] = future_data["Week"].apply(adjust_week)
future_data[["pressure", "humidity", "wind_deg"]] = scaler.transform(future_data[["pressure", "humidity", "wind_deg"]].values)

# other stuff must be splitted
train_data = list(df_data.index)
# cut off the earliest date possbile for training
EARLIEST_DATE_POSSIBLE = "2017-10-01"
train_data = sorted([date for date in train_data if date >= EARLIEST_DATE_POSSIBLE])

if True:
    print("Learning")
    # create data
    X_train, y_train = get_data_from_dates(dates=train_data, df=df_data)

    # create model
    model = create_model(n_features=N_FEATURES, out_shape=len(targets))
    model.summary()
    # fit model
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, verbose=1)

# predict
y_simple_continuous_predict = continuous_prediction(df=df_data, new_data=future_data.copy(), model=model)

# create data for plotting from all predictions
continuous_predicted_df = pd.DataFrame(y_simple_continuous_predict, columns=targets, index=future_data.index)

# for plotting
TIME_UNIT = "D" # possible units: D - Day, W - week, M - month, Q - quarter.
match TIME_UNIT:
    case "D":
        INTERVAL = 7
    case "W":
        INTERVAL = 3
    case _:
        INTERVAL = 1

# group all the results by unit
continuous_predicted_df = group_by_date(df=continuous_predicted_df, time_unit=TIME_UNIT)
x_values = continuous_predicted_df.index.map(lambda value: str(value))
# get axis values for plotting

for target in targets:
    # Plot the prediction results
    plt.figure(figsize=(6, 10))

    # Plot smart continuous prediction
    plt.plot(x_values, continuous_predicted_df[target], color="red", label="Simple Continuous Predicted Sales")

    # Configure the plot
    plt.title(f"Long-Time Prediction {target}")
    plt.xlabel("Days")
    plt.ylabel("Sales")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=INTERVAL))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()