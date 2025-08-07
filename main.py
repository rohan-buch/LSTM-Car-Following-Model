import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("processed_ngsim.csv")

n_steps = 10

vehicle_id = 102
df_train = df[df["Follower_ID"] != vehicle_id].copy()
df_test = df[df["Follower_ID"] == vehicle_id].copy()

df_train.sort_values(["Follower_ID", "Frame_ID"], inplace = True)
df_test.sort_values(["Frame_ID"], inplace = True)

for df_ in [df_train, df_test]:
    df_["Time_Headway"] = df_["Spacing"] / df_["Follower_Speed"].replace(0, np.nan) #makes sure we don't divide by 0
    df_.replace([np.inf, -np.inf], np.nan, inplace = True)
    df_["Next_Speed"] = df_["Follower_Speed"].shift(-1)
    df_.dropna(inplace = True)

def filter_stable_leader(df_, n_steps):
    df_["Leader_ID_Next"] = df_["Leader_ID"].shift(-1)
    df_["Same_Leader"] = (df_["Leader_ID"] == df_["Leader_ID_Next"]).astype(int)
    df_["Leader_Stable"] = (
        df_["Same_Leader"]
        .rolling(window = n_steps, min_periods = n_steps)
        .apply(lambda x: all(x==1), raw = True)
        .shift(-(n_steps - 1))
    )
    df_ = df_[df_["Leader_Stable"] == 1]
    df_.dropna(inplace = True)
    return df_

df_train = filter_stable_leader(df_train, n_steps)
df_test = filter_stable_leader(df_test, n_steps)


def create_sequences(df, features, target, n_steps, x_scaler, y_scaler):
    Xs, ys = [], []

    for follower_id, group in df.groupby("Follower_ID"):
        group = group.sort_values("Frame_ID")
        X = group[features]
        y = group[target]

        if len(X) <= n_steps:
            continue

        X_scaled = X_scaler.transform(X)
        y_scaled = y_scaler.transform(y.values.reshape(-1, 1)).flatten()

        for i in range(len(X_scaled) - n_steps):
            Xs.append(X_scaled[i:i + n_steps])
            ys.append(y_scaled[i + n_steps])
    
    return np.array(Xs), np.array(ys)

features = ["Follower_Speed", "Spacing", "Time_Headway"]
target = "Next_Speed"

X_train_df = df_train[features]
y_train_df = df_train["Next_Speed"]

X_test_df = df_test[features]
y_test_df = df_test["Next_Speed"]

X_scaler = StandardScaler()
X_scaler.fit(df_train[features])

y_scaler = StandardScaler()
y_scaler.fit(df_train[[target]])

X_train_seq, y_train_seq = create_sequences(df_train, features, target, n_steps, X_scaler, y_scaler)
X_test_seq, y_test_seq = create_sequences(df_test, features, target, n_steps, X_scaler, y_scaler)

X_train = X_train_seq
y_train = y_train_seq
X_test = X_test_seq
y_test = y_test_seq

model = keras.Sequential([
    layers.LSTM(32, return_sequences=True, input_shape=(n_steps, X_train.shape[2])),
    layers.LSTM(16, return_sequences=False),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss = "mse",
    metrics = ["mae"]
)

history = model.fit(
    X_train, y_train,
    validation_split = 0.2,
    batch_size = 16,
    epochs = 5,
    verbose = 1
)

loss, mae = model.evaluate(X_test, y_test, batch_size = 16)
print(f"Test MAE: {mae:.4f} m/s")

y_pred_scaled = model.predict(X_test_seq)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_actual = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1))

plt.figure(figsize=(10,5))
plt.scatter(y_actual, y_pred, alpha = 0.3)
plt.plot(
    [y_actual.min(), y_actual.max()],
    [y_actual.min(), y_actual.max()],
    "r--",
    label = "(y = x)"
)
plt.xlabel("Actual Next Speed")
plt.ylabel("Predicted Next Speed")
plt.title("Predicted vs Actual Speeds")
plt.legend()
plt.show()