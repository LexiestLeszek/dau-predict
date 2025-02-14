import pandas as pd
import numpy as np
import timesfm  # make sure you have installed the timesfm package from the GitHub repo

# Load the Excel file
excel_file = "data_1.xlsx"
df = pd.read_excel(excel_file)

# Convert the date column to datetime if necessary and sort by date
df["ds"] = pd.to_datetime(df["ds"], format="%d.%m.%Y")
df.sort_values("ds", inplace=True)

# Extract the active users series; adjust the column name if needed
active_users = df["dau"].values

# Define the input patch length (typically 32 for TimesFM-1.0-200m)
input_patch_len = 32

# Adjust context length to be the largest multiple of input_patch_len
context_len = (len(active_users) // input_patch_len) * input_patch_len

active_users = active_users[-context_len:]

# Create the TimesFM hyperparameters object
hparams = timesfm.TimesFmHparams(
    backend="gpu",                # or "cpu" depending on your hardware
    per_core_batch_size=32,       # if forecasting one series, you might consider 1; here we leave as 32
    horizon_len=30,               # forecasting one time step ahead
    context_len=context_len,
    input_patch_len=input_patch_len,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280
)

# For PyTorch, use the proper checkpoint repository (ensure the model files match backend expectations)
tfm = timesfm.TimesFm(
    hparams=hparams,
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )
)

# Prepare the forecast input (list with one time series)
forecast_input = [active_users]

# For daily data, the frequency indicator is 0
freq = [0]

# Run the forecast; the function returns a point forecast and additional outputs
point_forecast, _ = tfm.forecast(forecast_input, freq=freq)

# Print the forecasted value
#print("Forecasted Daily Active Users for next day:", point_forecast[0])

forecast_days = point_forecast[0]

for i, day in enumerate(forecast_days):
    #print(f"day {i+1}:    {day}")
    print(day)
