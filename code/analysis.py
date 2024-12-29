import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Classical model
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Machine Learning
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load the CSV file
# Replace 'cad_rmb_data.csv' with the actual path/filename of your CSV.
df = pd.read_csv('inputs/CAD_CNY Historical Data_1994_to_2024.csv')

# 2. Parse the "Date" column and set it as the index
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.set_index('Date', inplace=True)

# 3. Clean up columns:
#    - Remove the trailing '%' in "Change %" and convert to float.
#    - Convert numeric columns to float.
if 'Change %' in df.columns:
    df['Change %'] = df['Change %'].str.replace('%', '', regex=True)
    df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce') / 100.0

# Convert other columns to numeric, ignoring errors.
for col in ['Price','Open','High','Low','Vol.']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. Preview the cleaned DataFrame
print("\n--- Data Sample ---\n")
print(df.head())

# 5. Optional: Quick Seasonal Decomposition on "Price"
#    (Requires a sufficiently long time series, ideally with constant frequency.)

# First, resample "Price" to a monthly average if your data is daily/weekly
if 'Price' in df.columns:
    # Resample to monthly
    df_monthly = df['Price'].resample('M').mean()

    # Seasonal decomposition (assuming 12-month seasonality)
    # Note: The 'period' parameter depends on your frequency and known seasonality.
    decomposition = seasonal_decompose(df_monthly, model='additive', period=12)

    # Plot the decomposition
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.show()

# Resample to weekly frequency by taking the mean Price for each week
df_weekly = df['Price'].resample('W').mean().dropna()

# Let's rename it to make it explicit
df_weekly = df_weekly.to_frame(name='Price')

split_date = '2015-12-31'  # or pick another date
train_data = df_weekly.loc[:split_date].copy()
test_data  = df_weekly.loc[split_date:].copy()

print(f"Train range: {train_data.index.min()} to {train_data.index.max()}")
print(f"Test range:  {test_data.index.min()} to {test_data.index.max()}")

# -------------------------------
# 2) DEFINE THE SEARCH SPACE
# -------------------------------
# We'll search over common ARIMA range for p,d,q and some seasonal parameters P,D,Q, plus different seasonal periods (m).
space = {
    'p':  hp.quniform('p', 0, 5, 1),   # p in [0..5]
    'd':  hp.quniform('d', 0, 2, 1),   # d in [0..2]
    'q':  hp.quniform('q', 0, 5, 1),   # q in [0..5]
    'P':  hp.quniform('P', 0, 2, 1),   # P in [0..2]
    'D':  hp.quniform('D', 0, 1, 1),   # D in [0..1]
    'Q':  hp.quniform('Q', 0, 2, 1),   # Q in [0..2]
    'm':  hp.choice('m', [7, 12, 26, 52])  # possible seasonal periods
}

# -------------------------------
# 3) DEFINE OBJECTIVE FUNCTION
# -------------------------------
def objective(params):
    """
    1. Parse params
    2. Fit SARIMA
    3. Forecast on test set
    4. Compute & return RMSE
    """
    # Cast to int since p, d, q, etc. must be integers
    p = int(params['p'])
    d = int(params['d'])
    q = int(params['q'])
    P = int(params['P'])
    D = int(params['D'])
    Q = int(params['Q'])
    m = params['m']

    # Avoid models that can't converge by using try/except
    try:
        # Build & fit the SARIMA model
        model = SARIMAX(
            train_data, 
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        
        # Forecast for the test period
        forecast = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])
        
        # Calculate RMSE (or MSE, MAE, etc.)
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        
        # Return the metric for hyperopt (it tries to MINIMIZE the loss)
        return {'loss': rmse, 'status': STATUS_OK}
    
    except Exception as e:
        # Return a large number if the model fails to converge
        return {'loss': 9999999999, 'status': STATUS_OK}

# -------------------------------
# 4) RUN HYPEROPT
# -------------------------------
trials = Trials()
best = fmin(
    fn=objective,        # objective function
    space=space,         # search space
    algo=tpe.suggest,    # Tree Parzen Estimator
    max_evals=50,        # number of iterations
    trials=trials,
    rstate=np.random.default_rng(42)
)

print("\nBest parameters found by hyperopt:")
print(best)

# >>> print(best)
# {'D': 1.0, 'P': 2.0, 'Q': 1.0, 'd': 0.0, 'm': 3, 'p': 1.0, 'q': 5.0}

# Example SARIMA config:
p, d, q = (1, 0, 1)
P, D, Q, m = (2, 1, 1, 26)  # 52-week seasonality if you suspect yearly cycle in weekly data

model_sarima = SARIMAX(
    train_data['Price'], 
    order=(p, d, q),
    seasonal_order=(P, D, Q, m), 
    enforce_stationarity=False, 
    enforce_invertibility=False
)
sarima_res = model_sarima.fit(disp=False)
print(sarima_res.summary())

# Forecast for the test period
sarima_pred = sarima_res.predict(
    start=test_data.index[0],
    end=test_data.index[-1],
    dynamic=False
)

# Evaluate
mae_sarima = mean_absolute_error(test_data['Price'], sarima_pred)
mse_sarima = mean_squared_error(test_data['Price'], sarima_pred)
print(f"\n[SARIMA] MAE: {mae_sarima:.4f}, MSE: {mse_sarima:.4f}")

# Plot
plt.figure(figsize=(10,5))
plt.plot(train_data.index, train_data['Price'], label='Train')
plt.plot(test_data.index, test_data['Price'], label='Test')
plt.plot(test_data.index, sarima_pred, label='SARIMA Forecast', linestyle='--')
plt.title('SARIMA Model - Actual vs Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

##################################
# Machine Learning Model: XGBoost
##################################

def create_lagged_features(df_in, n_lags=3):
    """
    For each row, create lagged versions of 'Price'.
    Example: n_lags=3 -> Price(t-1), Price(t-2), Price(t-3)
    """
    df_out = df_in.copy()
    for lag in range(1, n_lags + 1):
        df_out[f'lag_{lag}'] = df_out['Price'].shift(lag)
    # Example time-based feature: week of year
    df_out['week_of_year'] = df_out.index.isocalendar().week.astype(int)
    return df_out

# Create lagged dataset (e.g., 3 weeks of lag)
df_lagged = create_lagged_features(df_weekly, n_lags=3).dropna()

# New train-test split on df_lagged
train_lagged = df_lagged.loc[:split_date].copy()
test_lagged  = df_lagged.loc[split_date:].copy()

X_train = train_lagged.drop(columns=['Price'])
y_train = train_lagged['Price']
X_test  = test_lagged.drop(columns=['Price'])
y_test  = test_lagged['Price']

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

model_xgb = XGBRegressor(n_estimators=100, random_state=42)
model_xgb.fit(X_train, y_train)

# Forecast
xgb_pred = model_xgb.predict(X_test)

# Evaluate
mae_xgb = mean_absolute_error(y_test, xgb_pred)
mse_xgb = mean_squared_error(y_test, xgb_pred)
print(f"\n[XGBoost] MAE: {mae_xgb:.4f}, MSE: {mse_xgb:.4f}")

# Plot
plt.figure(figsize=(10,5))
plt.plot(train_lagged.index, train_lagged['Price'], label='Train')
plt.plot(test_lagged.index, test_lagged['Price'], label='Test')
plt.plot(test_lagged.index, xgb_pred, label='XGB Forecast', linestyle='--')
plt.title('XGBoost Model - Actual vs Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
