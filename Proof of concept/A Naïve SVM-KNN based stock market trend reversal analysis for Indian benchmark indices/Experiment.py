#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:18:54 2024

@author: justine
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Reading the CSV file
df = pd.read_csv('nifty_2000_to_2024.csv')

# Hyperparameter abd TI
normalization_flg = False
visualization_flg = False

# Define the parameters
sma_window_size = 3
ema_smoothing_param = 0.2
williams_r_window_size = 3
RSI_window_size = 14
ATR_window_size = 14
VR_window_size = 14
k_param = 10
svm_C = 2e-2
svm_gamma = 2e-10

# Function to print the parameters
def print_parameters():
    print("Parameters:")
    print(f"SMA Window Size: {sma_window_size}")
    print(f"EMA Smoothing Parameter: {ema_smoothing_param}")
    print(f"Williams %R Window Size: {williams_r_window_size}")
    print(f"RSI Window Size: {RSI_window_size}")
    print(f"ATR Window Size: {ATR_window_size}")
    print(f"VR Window Size: {VR_window_size}")
    print(f"K Parameter (for k-NN): {k_param}")
    print(f"SVM C: {svm_C}")
    print(f"SVM Gamma: {svm_gamma}")

# Call the function to print the parameters
# print_parameters()


def williams_r(highs, lows, closes, window):
    highest_high = highs.rolling(window=window).max()
    lowest_low = lows.rolling(window=window).min()
    r = ((highest_high - closes) / (highest_high - lowest_low)) * -100
    return r

def calculate_rsi(df, window):
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, window):
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def calculate_vr(df, window):
    tr = df['High'] - df['Low']
    tr_prev = tr.shift(1).rolling(window=window).mean()
    vr = tr / tr_prev
    return vr

def set_y(row):
    if row['returns'] > 0:
        return 1
    else:
        return -1

## Calculation of all the TI
# Calculate the Simple Moving Average
df['returns'] = -df['Close'].diff(-1)
df['SMA_Close'] = df['Close'].rolling(window=sma_window_size).mean()
df['EMA_Close'] = df['Close'].ewm(alpha=ema_smoothing_param).mean()
df['Williams %R'] = williams_r(df['High'], df['Low'], df['Close'], window=williams_r_window_size)
df['RSI'] = calculate_rsi(df, window=RSI_window_size)
df['ATR'] = calculate_atr(df, window=ATR_window_size)
df['VR'] = calculate_vr(df, window=VR_window_size)
df['y'] = df.apply(set_y, axis=1)
df.drop(columns=['Date'],inplace=True)

# Remove non-numeric values
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace = True)

# Min Max Normalization
if normalization_flg == True:
    df['Open'] = (df['Open'] - df['Open'].min())/(df['Open'].max() - df['Open'].min())
    df['High'] = (df['High'] - df['High'].min())/(df['High'].max() - df['High'].min())
    df['Close'] = (df['Close'] - df['Close'].min())/(df['Close'].max() - df['Close'].min())
    df['Low'] = (df['Low'] - df['Low'].min())/(df['Low'].max() - df['Low'].min())
    
    df['returns'] = (df['returns'] - df['returns'].min())/(df['returns'].max() - df['returns'].min())
    df['SMA_Close'] = (df['SMA_Close'] - df['SMA_Close'].min())/(df['SMA_Close'].max() - df['SMA_Close'].min())
    df['EMA_Close'] = (df['EMA_Close'] - df['EMA_Close'].min())/(df['EMA_Close'].max() - df['EMA_Close'].min())
    df['Williams %R'] = (df['Williams %R'] - df['Williams %R'].min())/(df['Williams %R'].max() - df['Williams %R'].min())
    df['RSI'] = (df['RSI'] - df['RSI'].min())/(df['RSI'].max() - df['RSI'].min())
    df['ATR'] = (df['ATR'] - df['ATR'].min())/(df['ATR'].max() - df['ATR'].min())
    df['VR'] = (df['VR'] - df['VR'].min())/(df['VR'].max() - df['VR'].min())

if visualization_flg == True:
    plt.figure()
    plt.plot(df['Close'])
    plt.plot(df['SMA_Close'])
    plt.plot(df['EMA_Close'])
    plt.legend(['Close','SMA','EMA'])

# Fitting the variables in SVC

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

training_size = int(df.shape[0]*0.9)
data_size = df.shape[0]
testing_size = data_size - training_size

rbf_svm = SVC(kernel='rbf', C=svm_C, gamma=svm_gamma)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def knn_search(X_train, X_vec, k):

    # Calculate distances from test point to all training points
    distances = np.linalg.norm(X_train - X_vec, axis=1)
    X_train['distances'] = distances
    
    X_train_sorted = X_train.sort_values(by='distances',ascending = True)
    
    # Get the indices of the k nearest neighbors
    if k < X_train_sorted.shape[0]:
        return X_train_sorted.head(k)
    else:
        return X_train_sorted.head(X_train_sorted.shape[0])



if __name__ == "__main__":

    print("..................................................")
    print(f"The Training size is {training_size}.")
    print(f"The Data size is {data_size}.")
    print(f"The Testing size is {testing_size}.")
    print()
    print_parameters()
    print("..................................................")
    
    svm_test_counter = 0
    error_price = np.zeros(testing_size)
    predicted_price = np.zeros(testing_size)
    for i in tqdm(range(1,testing_size-1),desc = "Processing : "):
        
        X_train = df.iloc[i:i+training_size,:-1]
        y_train_svm = df.iloc[i:i+training_size,-1]
        y_train_knn = df.iloc[i:i+training_size,6]
        
        X_test = df.iloc[i+training_size,:-1]
        y_test_svm = df.iloc[i+training_size,-1]
        y_test_knn = df.iloc[i+training_size,6]
        
        rbf_svm.fit(X_train, y_train_svm)
        
        y_pred_svm = rbf_svm.predict(X_test.values.reshape(1, -1))
        if y_pred_svm == y_test_svm:
            
            svm_test_counter += 1
            
            nearest_neighbours = knn_search(X_train, X_test, k_param)
            predicted_returns = nearest_neighbours['returns'].mean()
            predicted_price[i-1] = X_test['Close'] + predicted_returns
            
        error_price[i-1] = df.iloc[i+training_size+1,:-1]['Close'] - predicted_price[i-1]
        
        
    # Error Evaluation
    print(f"The accuracy of the SVM model is {svm_test_counter/testing_size}.")
    print(f"The rmse of the return estimation model (KNN) is {np.sqrt(np.mean((error_price) ** 2))}.")