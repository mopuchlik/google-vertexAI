import pandas as pd


def build_default_flags(df: pd.DataFrame, data_cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    For each id, set binary flags (def_ind_1m/2m/3m) indicating whether there is at least
    one default_ind==1 in the window (data_cutoff, data_cutoff + N months], for N=1,2,3.

    Parameters
    ----------
    df : DataFrame with columns:
         - 'id' (string)
         - 'date' (date-like: 'yyyy-mm-dd')
         - 'default_ind' (0 or 1)
    data_cutoff : pandas.Timestamp or date-like ('yyyy-mm-dd')

    Returns
    -------
    DataFrame with columns:
      id, def_ind_1m, def_ind_2m, def_ind_3m  (all 0/1 at id level)
    """
    # Ensure proper dtypes
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    data_cutoff = pd.to_datetime(data_cutoff).normalize()

    # All ids to report on (every id present in df)
    ids = out["id"].drop_duplicates().sort_values()

    # flags = {}
    for n in (1, 2, 3):
        horizon_end = data_cutoff + pd.DateOffset(months=n)
        # Window: strictly AFTER cutoff up to and including horizon_end
        mask = (out["date"] > data_cutoff) & (out["date"] <= horizon_end)
        # Any default within window for each id → 0/1
        s = (
            out.loc[mask]
            .groupby("id")["default_ind"]
            .max()  # any 1 → 1
            .reindex(ids, fill_value=0)  # include ids with no rows in window
            .astype(int)
            .rename(f"def_ind_{n}m")
        )

        s = pd.DataFrame(s).reset_index()

        out = pd.merge(out, s, on="id", how="left")

        # remove obs before cutoff not to have data leakage
    out = out[out["date"] <= data_cutoff]

    return out


# %%
