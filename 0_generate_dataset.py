# The Python code below generates a synthetic dataset based on your specifications. It uses `pandas` for DataFrame manipulation, `numpy` for numerical operations and random number generation, and `datetime` for date handling.

# Here's a breakdown of how each requirement is addressed:

# 1.  **Fixed Seed**: `numpy.random.seed(42)` and `random.seed(42)` are used for reproducibility.
# 2.  **1000 IDs**: A list of 1000 unique IDs (`ID_0001` to `ID_1000`) is generated.
# 3.  **Time Series Dates**: `pd.date_range` is used to create a series of Mondays from `01.01.2020` to `01.10.2025`.
# 4.  **Default Indicator (`default_ind`)**:
#     *   For each ID, there's a 5% chance of default.
#     *   If an ID defaults, the `first_default_date` is chosen randomly from after the initial 10% of its time series length.
#     *   `default_ind` is `0` before this date and `1` from this date onwards for that specific ID.
# 5.  **Regressors**: These are generated iteratively for each ID to handle time-series dependencies and ID-specific characteristics.
#     *   **`credit_limit`**: Starts with a random value between `100k` and `5M`. In each subsequent week, it has a `1/no_obs` probability of changing by `+/- 20-30%` of its current value. A minimum limit of `10k` is enforced.
#     *   **`sector`**: A categorical variable (`A`, `B`, `C`, `D`) is chosen randomly and remains constant for each ID.
#     *   **`used_amount`**:
#         *   Starts at `credit_limit / 2`.
#         *   Follows a unit root (random walk) process: `used_amount_t = used_amount_{t-1} + epsilon_t`.
#         *   `epsilon_t` is a normal random variable. Its standard deviation is calculated based on the initial credit limit and time series length to encourage approximately 10% of IDs to reach 0 and 10% to reach their maximum limit, while keeping the values sensible.
#         *   The `used_amount` is always clipped between `0` and the current `credit_limit`.
#     *   **`f_dummy1` - `f_dummy5`**: For each of these dummy features, a scaling factor `n` is chosen randomly from `uniform[0, 1000]`. All observations for a specific dummy feature are then `n * Normal(0, 1)`.
# 6.  **Nullification on Default**: If an ID defaults, `credit_limit`, `sector`, `used_amount`, and all `f_dummy` features are set to `NaN` from the `first_default_date` onwards for that ID. `default_ind` is explicitly excluded from nullification as it indicates the default status.
# 7.  **Output**: A single `pandas.DataFrame` containing all generated data.

# ```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from module_dataprep import build_default_flags


# --- 1) Fix seed for reproducibility ---
np.random.seed(42)
random.seed(42)

# --- 2) Generate time series dates ---
# from 01.01.2020 to 01.10.2025, weekly data, each Monday
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 10, 1)
date_range = pd.date_range(start=start_date, end=end_date, freq="W-MON")
no_obs_per_id = len(date_range)

print(f"Number of observations per ID: {no_obs_per_id}")

# --- 1) Generate IDs ---
num_ids = 1000
ids = [f"ID_{i+1:04d}" for i in range(num_ids)]

# Prepare a list to store data for each ID
all_id_data = []

# Pre-calculate `n` for dummy features once (as per interpretation: fixed per feature across all observations)
n_dummies = {f"f_dummy{i}": np.random.uniform(0, 1000) for i in range(1, 6)}

# Dictionary to store first default dates for nullification later
first_default_dates = {}

# --- Generate data for each ID ---
for _id in ids:
    id_df = pd.DataFrame({"id": _id, "date": date_range})

    # --- 3a & 3b) Binary default indicator (default_ind) ---
    # For each ID there is 5% probability of default
    if np.random.rand() < 0.05:
        # If default happens, the first default date is chosen randomly after initial 10% of time series length
        min_default_idx = int(no_obs_per_id * 0.10)

        # Ensure that min_default_idx doesn't exceed available indices
        if min_default_idx >= no_obs_per_id:
            min_default_idx = no_obs_per_id - 1 if no_obs_per_id > 0 else 0

        first_default_idx = np.random.randint(min_default_idx, no_obs_per_id)
        default_date = id_df.iloc[first_default_idx]["date"]
        first_default_dates[_id] = default_date

        # default_ind is 1 from the first default date onwards, 0 otherwise
        id_df["default_ind"] = (id_df["date"] >= default_date).astype(int)
    else:
        id_df["default_ind"] = 0
        first_default_dates[_id] = None  # No default for this ID

    # --- 4b) Categorical variable sector (uniform distribution from {A, B, C, D}) ---
    id_df["sector"] = np.random.choice(["A", "B", "C", "D"])

    # --- 4a) credit_limit ---
    # Random value from uniform distribution [100k, 5M] initially
    initial_credit_limit = np.random.uniform(100000, 5000000)
    credit_limits = [initial_credit_limit]

    for i in range(1, no_obs_per_id):
        current_limit = credit_limits[-1]
        # May change with probability 1/no_obs
        if np.random.rand() < (1 / no_obs_per_id):
            change_percent = np.random.uniform(0.20, 0.30)  # +/- uniform[20%, 30%]
            change_amount = current_limit * change_percent
            if np.random.rand() < 0.5:  # 50% chance to increase/decrease
                new_limit = current_limit + change_amount
            else:
                new_limit = current_limit - change_amount
            # Ensure limit doesn't go below a reasonable minimum (e.g., 10k)
            credit_limits.append(max(10000.0, new_limit))
        else:
            credit_limits.append(current_limit)
    id_df["credit_limit"] = credit_limits

    # --- 4c) used_amount ---
    # Always <= limit, starts at limit/2, then unit root (random walk)
    # with std dev making it sensible (10% IDs reach 0 and 10% ID reach max amount)
    used_amounts = []

    # Heuristic for std_epsilon to achieve the 10% goals:
    # For a random walk, spread is proportional to sqrt(time) * sigma_epsilon.
    # We want (initial_limit/2) to be roughly 1.28 * sqrt(no_obs_per_id) * sigma_epsilon
    # So, sigma_epsilon ~ (initial_limit/2) / (1.28 * sqrt(no_obs_per_id))

    # Use initial_credit_limit as a base for standard deviation
    # Add some randomness to std_epsilon for more varied ID behavior
    base_std_epsilon = (initial_credit_limit / 2) / (1.28 * np.sqrt(no_obs_per_id))
    std_epsilon = base_std_epsilon * np.random.uniform(0.5, 1.5)
    std_epsilon = max(100.0, std_epsilon)  # Ensure a minimum reasonable std_epsilon

    current_used_amount = initial_credit_limit / 2

    for i in range(no_obs_per_id):
        # First, ensure the current_used_amount is within the current credit_limit.
        # This clips the previous step's potential drift for the current observation.
        current_used_amount = max(
            0.0, min(id_df["credit_limit"].iloc[i], current_used_amount)
        )
        used_amounts.append(current_used_amount)

        # Then, apply the random walk for the *next* step
        epsilon = np.random.normal(0, std_epsilon)
        current_used_amount += epsilon

    id_df["used_amount"] = used_amounts

    # --- 4d) features f_dummy1--f_dummy5 ---
    for i in range(1, 6):
        col_name = f"f_dummy{i}"
        # 'n' is chosen from uniform distribution [0, 1000] for each dummy feature (once per feature)
        id_df[col_name] = n_dummies[col_name] * np.random.normal(0, 1, no_obs_per_id)

    # --- 4f) Nullification on Default ---
    # When default happens, other features are nullified (set to NaN)
    # from the first default date until the end of time series for that ID
    if first_default_dates[_id] is not None:
        default_date = first_default_dates[_id]

        # Columns to nullify (all except 'id', 'date', 'default_ind')
        cols_to_nullify = [
            col for col in id_df.columns if col not in ["id", "date", "default_ind"]
        ]

        id_df.loc[id_df["date"] >= default_date, cols_to_nullify] = np.nan

    all_id_data.append(id_df)

# --- 5) Output: pandas dataframe ---
final_df = pd.concat(all_id_data).reset_index(drop=True)

# Ensure 'date' column is datetime type
final_df["date"] = pd.to_datetime(final_df["date"])

print("\nDataset generated successfully!")
print(f"Shape of the dataset: {final_df.shape}")
print("\nFirst 5 rows:")
print(final_df.head())
print("\nLast 5 rows:")
print(final_df.tail())
print("\nInfo:")
final_df.info()

# --- Basic Validation Checks (Optional) ---
print("\n--- Validation Checks ---")

# Check default_ind logic:
defaulted_ids_list = [
    _id for _id, date in first_default_dates.items() if date is not None
]
if defaulted_ids_list:
    sample_id_default = np.random.choice(defaulted_ids_list)
    sample_default_date = first_default_dates[sample_id_default]
    print(
        f"\nChecking default_ind for a defaulted ID ({sample_id_default}) with first default date {sample_default_date}:"
    )
    default_rows = final_df[
        (final_df["id"] == sample_id_default)
        & (final_df["date"] >= sample_default_date)
    ]
    pre_default_rows = final_df[
        (final_df["id"] == sample_id_default) & (final_df["date"] < sample_default_date)
    ]

    if not default_rows.empty:
        print(f"Default indicator from default date onwards (first 5 rows):")
        print(default_rows["default_ind"].head())
        print(
            f"Expected all 1s. Actual unique values: {default_rows['default_ind'].unique()}"
        )
    if not pre_default_rows.empty:
        print(f"Default indicator before default date (last 5 rows):")
        print(pre_default_rows["default_ind"].tail())
        print(
            f"Expected all 0s. Actual unique values: {pre_default_rows['default_ind'].unique()}"
        )

non_defaulted_ids_list = [
    _id for _id, date in first_default_dates.items() if date is None
]
if non_defaulted_ids_list:
    sample_id_non_default = np.random.choice(non_defaulted_ids_list)
    print(f"\nChecking default_ind for a non-defaulted ID ({sample_id_non_default}):")
    print(
        final_df[final_df["id"] == sample_id_non_default]["default_ind"].value_counts()
    )
    print("Expected all 0s.")

# Check nullification for a defaulted ID
if defaulted_ids_list:
    sample_id_nullify = np.random.choice(defaulted_ids_list)
    sample_default_date_nullify = first_default_dates[sample_id_nullify]
    print(
        f"\nChecking nullification for defaulted ID ({sample_id_nullify}) from {sample_default_date_nullify}:"
    )
    nullified_cols = [
        col for col in final_df.columns if col not in ["id", "date", "default_ind"]
    ]

    pre_default_row = final_df[
        (final_df["id"] == sample_id_nullify)
        & (final_df["date"] < sample_default_date_nullify)
    ].tail(1)
    post_default_row = final_df[
        (final_df["id"] == sample_id_nullify)
        & (final_df["date"] >= sample_default_date_nullify)
    ].head(1)

    if not pre_default_row.empty:
        print(
            f"Number of NaNs in features just BEFORE default ({pre_default_row['date'].iloc[0]}):"
        )
        print(pre_default_row[nullified_cols].isna().sum().sum())  # Should be 0 NaNs
    if not post_default_row.empty:
        print(
            f"Number of NaNs in features at/AFTER default ({post_default_row['date'].iloc[0]}):"
        )
        print(post_default_row[nullified_cols].isna().sum().sum())
        print(
            f"Expected {len(nullified_cols)} NaNs. Actual: {post_default_row[nullified_cols].isna().sum().sum()}"
        )

# Check used_amount <= credit_limit (should be 0 cases where used_amount > credit_limit)
print(
    "\nChecking used_amount <= credit_limit (should ideally be 0 cases where used_amount > credit_limit after excluding NaNs):"
)
# Filter out NaNs as they are expected after default
issues = final_df[
    final_df["used_amount"].notna()
    & final_df["credit_limit"].notna()
    & (final_df["used_amount"] > final_df["credit_limit"])
]
print(f"Number of rows where used_amount > credit_limit: {len(issues)}")
# ```
final_df_prep = build_default_flags(final_df, "2025-02-01")
csv_filename = "generated_dataset.csv"
final_df_prep.to_csv(csv_filename, index=False)
