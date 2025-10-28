# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 28.10.2025

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# --- Load the dataset ---
data = pd.read_csv("housing_price_dataset.csv")

# --- Prepare yearly average prices ---
data_yearly = data.groupby('YearBuilt')['Price'].mean().sort_index()

# Convert YearBuilt to datetime and set as index
data_yearly.index = pd.to_datetime(data_yearly.index, format='%Y')
data_yearly = pd.DataFrame(data_yearly)
data_yearly.columns = ['AveragePrice']

# --- Display info ---
print("Dataset Shape:", data_yearly.shape)
print("First 10 Rows:\n", data_yearly.head(10))

# --- Plot the data ---
plt.figure(figsize=(10, 5))
plt.plot(data_yearly.index, data_yearly['AveragePrice'], label='Average House Price')
plt.title('Average House Price by Year Built')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.grid(True)
plt.legend()
plt.show()

# --- Define ARIMA Model Function ---
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()
    
    # Forecast for test period
    forecast = fitted_model.forecast(steps=len(test_data))
    
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
    
    # --- Plot results ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='red', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel('Average House Price')
    plt.title(f'ARIMA Forecasting for {target_variable} (order={order})')
    plt.legend()
    plt.grid()
    plt.show()
    
    print("Root Mean Squared Error (RMSE):", rmse)
    return fitted_model, forecast

# --- Run the ARIMA model ---
# (p,d,q) = (5,1,0) as a starting point; can be tuned
model, forecast = arima_model(data_yearly, 'AveragePrice', order=(5,1,0))
```

### OUTPUT:

<img width="889" height="476" alt="image" src="https://github.com/user-attachments/assets/1b2b6bf9-fdfc-40d2-8a6e-c83581ba58e3" />


<img width="884" height="580" alt="image" src="https://github.com/user-attachments/assets/0f7e3577-58a2-469d-ad56-12305ff4a116" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
