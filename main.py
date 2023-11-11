import h5py
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


# Load the data from the HDF5 file
files = [
    "Austria.hdf5",
    "Bahrain.hdf5",
    "Hungary.hdf5",
    "Portugal.hdf5",
    "Spain.hdf5",
    "Turkey.hdf5",
]

for file in files:
    with h5py.File(file, "r") as f:
        data = f["data"][:]
        channel_names = list(f.attrs["channel_names"])
        all_data.append(data)
        all_channel_names.append(channel_names)

# Extract channels
for data, channel_names in zip(all_data, all_channel_names):
    car_speed = data[:, channel_names.index("vCar")]
    air_temp = data[:, channel_names.index("TAir")]
    track_temp = data[:, channel_names.index("TTrack")]
    brakeL_temp = data[:, channel_names.index("TBrakeL")]
    brakeR_temp = data[:, channel_names.index("TBrakeR")]


# Prepare data for LSTM
brakeL_temps = brakeL_temp.reshape((brakeL_temp.shape[0], 1, 1))
brakeR_temps = brakeR_temp.reshape((brakeR_temp.shape[0], 1, 1))
air_temps = air_temp.reshape((air_temp.shape[0], 1, 1))
track_temps = track_temp.reshape((track_temp.shape[0], 1, 1))
car_speeds = car_speed.reshape((car_speed.shape[0], 1, 1))

# Split data into training and testing sets
train_size = int(len(brakeL_temp) * 0.67)
test_size = len(brakeL_temp) - train_size

trainL, testL = (
    brakeL_temps[0:train_size, :],
    brakeL_temps[train_size : len(brakeL_temps), :],
)
trainR, testR = (
    brakeR_temps[0:train_size, :],
    brakeR_temps[train_size : len(brakeR_temps), :],
)

# Define the LSTM Model(s) - Given more time/resources, would add more LSTM layers while avoiding overfitting
modelL = Sequential()
modelL.add(
    LSTM(4, input_shape=(1, 1), return_sequences=True, kernel_regularizer=l2(0.01))
)
modelL.add(LSTM(4, kernel_regularizer=l2(0.01)))  # Add another LSTM layer
modelL.add(Dense(1, kernel_regularizer=l2(0.01)))

modelR = Sequential()
modelR.add(
    LSTM(4, input_shape=(1, 1), return_sequences=True, kernel_regularizer=l2(0.01))
)
modelR.add(LSTM(4, kernel_regularizer=l2(0.01)))
modelR.add(Dense(1, kernel_regularizer=l2(0.01)))

# Compile the model(s)
modelL.compile(loss="mean_squared_error", optimizer="adam")
modelR.compile(loss="mean_squared_error", optimizer="adam")

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Give a summary
modelL.summary()
modelR.summary()

# Train the model(s) - Given more time/resources, would increase epochs for more complex pattern learning and adjust batch size for performance
historyL = modelL.fit(
    trainL,
    trainL,
    epochs=200,  # 200, 16 worked best so far
    batch_size=16,
    verbose=2,
    validation_data=(testL, testL),
    callbacks=[early_stopping],
)

historyR = modelR.fit(
    trainR,
    trainR,
    epochs=200,
    batch_size=16,
    verbose=2,
    validation_data=(testR, testR),
    callbacks=[early_stopping],
)

# Make Prediction(s)
train_predictL = modelL.predict(trainL)
test_predictL = modelL.predict(testL)
train_predictR = modelR.predict(trainR)
test_predictR = modelR.predict(testR)

# Print training predictions
print("Left Brake Training Prediction:")
print(f"Prediction: {train_predictL[0]}, Actual: {trainL[0]}")

print("Right Brake Training Prediction:")
print(f"Prediction: {train_predictR[0]}, Actual: {trainR[0]}")

# Print testing predictions
print("Left Brake Testing Prediction:")
print(f"Prediction: {test_predictL[0]}, Actual: {testL[0]}")

print("Right Brake Testing Prediction:")
print(f"Prediction: {test_predictR[0]}, Actual: {testR[0]}")

# Other optimisations that could be included - Batch normalisation, tuning learning rate, hyperparameter tuning with grid search/random search/bayesian optimisation, more regularisation, data augmentation, expermenting with model architectures
