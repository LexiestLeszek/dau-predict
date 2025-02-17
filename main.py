import pandas as pd
import numpy as np
import timesfm

# Load data
df = pd.read_excel("board1_main.xlsx")
df["ds"] = pd.to_datetime(df["ds"], format="%d.%m.%Y")

# Split into historical and future data
historical = df[df["ds"] <= "2025-02-12"]  # Data with known DAU
future = df[df["ds"] > "2025-02-12"]  # Dates to forecast

# Extract series and covariates
active_users = historical["dau"].values
holiday = df["holiday"].values.astype(str).tolist()  # Full timeline
weekday = df["weekday"].values.astype(str).tolist()  # Full timeline

# Configuration
input_patch_len = 32
context_len = (len(active_users) // input_patch_len) * input_patch_len
horizon_len = len(future)  # Use actual number of days to forecast

# Truncate to match context length
active_users = active_users[-context_len:]
last_date = historical["ds"].iloc[-1]

# Get future covariates from existing data
future_holiday = future["holiday"].values.astype(str).tolist()
future_weekday = future["weekday"].values.astype(str).tolist()

# Combine historical + future covariates
full_holiday = holiday[-context_len:] + future_holiday
full_weekday = weekday[-context_len:] + future_weekday

# Initialize model (JAX version for covariates support)
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=32,
        horizon_len=horizon_len,
        context_len=context_len,
        input_patch_len=input_patch_len,
        num_layers=50,
        model_dims=1280
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    )
)

# Forecast with covariates
forecast, _ = tfm.forecast_with_covariates(
    inputs=[active_users],
    dynamic_categorical_covariates={
        "holiday": [full_holiday],
        "weekday": [full_weekday]
    },
    dynamic_numerical_covariates={},
    static_categorical_covariates={},
    static_numerical_covariates={},
    freq=[0]
)

# Map forecast to specific dates
forecast_dates = future["ds"].dt.strftime("%Y-%m-%d").tolist()
forecast_values = forecast[0].tolist()

print("DAU Forecast for February 2025:")
for date, value in zip(forecast_dates, forecast_values):
    print(f"{date}: {value:.1f} users")
    #print(f"{value:.1f}")
