import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from src.utils import adjust_week
from sklearn.preprocessing import StandardScaler
import keras

from src.mensa_ml import (
    get_data_from_dates,
    educated_guess,
    continuous_prediction,
    smart_continuous_prediction,
    BATCH_SIZE,
    N_EPOCHS,
    N_FUTURES,
    PATH_TO_DATA,
    targets,
    tune_model,
    create_model,
)

# Load dataset
df_data = pd.read_csv(PATH_TO_DATA, index_col=0)
df_data.drop(columns=["Year", "Student Tax", "Worker Tax", "Guest Tax"], inplace=True)
N_FEATURES = df_data.shape[1]

# apply changes
df_data["Week"] = df_data["Week"].apply(adjust_week)
scaler = StandardScaler()
df_data[["pressure", "humidity", "wind_deg"]] = scaler.fit_transform(df_data[["pressure", "humidity", "wind_deg"]].values)

# train-val-test-split
# train_data, val_data, test_data = train_val_test_split(dates=list(df_data.index), n_val_days=91, n_test_days=109)

# split to train, test, validation
BORDER_DATE = "2024-01-01"
test_data = df_data.index[df_data.index >= BORDER_DATE].to_list() # dates that will not be shown to models

# select test_samples that have no zero results
zero_mask = (df_data.loc[test_data, targets] == 0).all(axis=1)
non_zero_dates = [date for mask, date in zip(zero_mask, test_data, strict=True) if not mask]

# other stuff must be splitted
train_data = list(set(df_data.index).difference(set(test_data))) # all other dates for training

# cut off the earliest date possbile for training
EARLIEST_DATE_POSSIBLE = "2017-10-01"
train_data = sorted([date for date in train_data if date >= EARLIEST_DATE_POSSIBLE])

# real learning and testing
if True:
    print("Test Run")
    # create data
    X_train, y_train = get_data_from_dates(dates=train_data, df=df_data)
    X_test, y_test = get_data_from_dates(dates=test_data, df=df_data)

    # create model
    N_FEATURES=df_data.shape[1]
    OUT_SHAPE=len(targets)

    # Execute the tuning process
    """
    tuner = tune_model(
        n_features=N_FEATURES,
    	out_shape=OUT_SHAPE,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        max_trials=36,
    )
    """

    # Retrieve the best hyperparameters and build the best model
    # best_hps = tuner.get_best_hyperparameters()[0]
    model = create_model(n_features=N_FEATURES, out_shape=OUT_SHAPE, n_futures=N_FUTURES) # tuner.hypermodel.build(best_hps)
    model.summary()
    
    # fit model
    history = model.fit(X_train, 
                        y_train, 
                        batch_size=BATCH_SIZE, 
                        epochs=N_EPOCHS, 
                        validation_data=(X_test, y_test), 
                        verbose=1, 
                        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", 
                                                                 patience=5, 
                                                                 restore_best_weights=True)]
    )

    # plot learning
    # plot_history(history=history)
    

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
    y_predict_continuous = smart_continuous_prediction(df=df_data, new_data=new_data, model=model, time_unit="M")

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

# save predictions to the DataFrame
continuous_predicted_df.to_csv("predictions/2024-test-predict.csv")
smart_predicted_df.to_csv("predictions/2024-test-predict-iterative.csv")
truth_df.to_csv("predictions/2024-true-data.csv")
educated_guess_df.to_csv("predictions/2024-educated-guess.csv")


# save model
model.save("mensa_model.keras")