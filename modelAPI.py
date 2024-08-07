import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.stats import skew, kurtosis
from scipy.fft import fft

app = Flask(__name__)
TIMESTEPS = 10  # the number of sample to be fed to the NN
FEATURES = 6
LABELS = 3
N_RECORDS = 11
BATCH_SIZE = 250


def extract_features(df, window_size):
    features = []
    labels = []
    for i in range(0, len(df), window_size):
        window = df.iloc[i:i + window_size]
        if len(window) == window_size:
            acc_x = window['AccX'].values
            acc_y = window['AccY'].values
            acc_z = window['AccZ'].values
            gyro_x = window['GyroX'].values
            gyro_y = window['GyroY'].values
            gyro_z = window['GyroZ'].values

            # Time domain features
            feature_vector = [
                np.mean(acc_x), np.std(acc_x), skew(acc_x), kurtosis(acc_x),
                np.mean(acc_y), np.std(acc_y), skew(acc_y), kurtosis(acc_y),
                np.mean(acc_z), np.std(acc_z), skew(acc_z), kurtosis(acc_z),
                np.mean(gyro_x), np.std(gyro_x), skew(gyro_x), kurtosis(gyro_x),
                np.mean(gyro_y), np.std(gyro_y), skew(gyro_y), kurtosis(gyro_y),
                np.mean(gyro_z), np.std(gyro_z), skew(gyro_z), kurtosis(gyro_z)
            ]

            # Frequency domain features (FFT)
            fft_acc_x = np.abs(fft(acc_x))
            fft_acc_y = np.abs(fft(acc_y))
            fft_acc_z = np.abs(fft(acc_z))
            fft_gyro_x = np.abs(fft(gyro_x))
            fft_gyro_y = np.abs(fft(gyro_y))
            fft_gyro_z = np.abs(fft(gyro_z))
            feature_vector.extend([
                np.mean(fft_acc_x), np.std(fft_acc_x),
                np.mean(fft_acc_y), np.std(fft_acc_y),
                np.mean(fft_acc_z), np.std(fft_acc_z),
                np.mean(fft_gyro_x), np.std(fft_gyro_x),
                np.mean(fft_gyro_y), np.std(fft_gyro_y),
                np.mean(fft_gyro_z), np.std(fft_gyro_z)
            ])

            features.append(feature_vector)

    return np.array(features)


@app.route('/')
def home():
    return "Hello"


@app.route("/predict", methods=['POST'])
def predict():
    file_path = "model_drive_style.h5"
    try:
        with open("random_forest_model.pkl", "rb") as file:
            model = pickle.load(file)
        data = request.get_json(force=True)
        new_df = pd.DataFrame(data)
        # print("Received data:", data)
        # features = np.array([[
        #     entry["AccX"],
        #     entry["AccY"],
        #     entry["AccZ"],
        #     entry["GyroX"],
        #     entry["GyroY"],
        #     entry["GyroZ"]
        # ] for entry in data])

        # Drop any NaN values
        new_df = new_df.dropna()
        window_size = 8

        # Extract features and labels
        X = extract_features(new_df, window_size)

        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)

        prediction = model.predict(X)

        return jsonify(prediction.tolist())
    except OSError as e:
        raise OSError(f"Unable to open the file: {e}")


if __name__ == '__main__':
    app.run(debug=True)
