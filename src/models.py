import logging
import sqlite3
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Machine Learning feature matrix definition
FEATURE_COLS_ML: List[str] = [
    "pred_xmin",
    "rolling_min_3",
    "rolling_starts_3",
    "rolling_xG_90_3",
    "rolling_xA_90_3",
    "rolling_xGC_90_3",
    "rolling_pts_3",
    "rolling_defcon_90_3",
    "rolling_saves_90_3",
    "rolling_min_5",
    "rolling_starts_5",
    "rolling_xG_90_5",
    "rolling_xA_90_5",
    "rolling_xGC_90_5",
    "rolling_pts_5",
    "rolling_defcon_90_5",
    "rolling_saves_90_5",
    "was_home",
    "pos_GK",
    "pos_DEF",
    "pos_MID",
    "pos_FWD",
]


def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes positional One-Hot features required by the XGBoost feature matrix."""
    df_encoded = df.copy()

    for pos_code in ["GK", "DEF", "MID", "FWD"]:
        df_encoded[f"pos_{pos_code}"] = (
            df_encoded["position"] == pos_code
        ).astype(int)

    return df_encoded


def predict_xmin(row: pd.Series) -> float:
    """Estimates expected playing minutes (xMin) based on recent rolling starts and minutes."""
    min_3 = row.get("rolling_min_3", 0.0)
    starts_3 = row.get("rolling_starts_3", 0.0)

    if starts_3 >= 2.5 or min_3 >= 240:
        return 85.0
    elif starts_3 >= 1.5 or min_3 >= 150:
        return 65.0
    elif min_3 >= 45:
        return 25.0
    else:
        return 0.0


def calculate_poisson_xp(row: pd.Series) -> float:
    """Calculates expected points (xP) using Poisson tactical probabilities."""
    position = str(row.get("position", "")).upper()
    pred_xmin = row.get("pred_xmin", 0.0)

    if pred_xmin == 0:
        return 0.0

    # Appearance points
    pts_appearance = 2.0 if pred_xmin >= 60 else 1.0

    # Goal & Assist expected points
    p_xg = row.get("pred_player_xG", 0.0)
    p_xa = row.get("pred_player_xA", 0.0)

    if position in ["GKP", "GK", "DEF"]:
        pts_goals = p_xg * 6.0
    elif position == "MID":
        pts_goals = p_xg * 5.0
    else:  # FWD
        pts_goals = p_xg * 4.0

    pts_assists = p_xa * 3.0

    # Clean sheet expected points
    prob_cs = row.get("prob_clean_sheet", 0.0)
    if pred_xmin >= 60:
        if position in ["GKP", "GK", "DEF"]:
            pts_cs = prob_cs * 4.0
        elif position == "MID":
            pts_cs = prob_cs * 1.0
        else:
            pts_cs = 0.0
    else:
        pts_cs = 0.0

    # Defensive contribution & Saves
    pts_defcon = row.get("exp_defcon", 0.0) * 0.1
    pts_saves = (
        (row.get("exp_saves", 0.0) / 3.0) * 1.0
        if position in ["GKP", "GK"]
        else 0.0
    )

    total_xp = (
        pts_appearance
        + pts_goals
        + pts_assists
        + pts_cs
        + pts_defcon
        + pts_saves
    )
    return round(float(total_xp), 2)


def compute_poisson_backtest(df_all: pd.DataFrame, target_gw: int) -> pd.Series:
    """Computes Poisson baseline xP predictions for backtesting on DataFrame records without data leakage."""
    df_past = df_all[df_all["gw"] < target_gw]
    df_curr = df_all[df_all["gw"] == target_gw].copy()

    if df_curr.empty:
        return pd.Series(dtype=float)

    team_stats = (
        df_past.groupby(["team_name", "was_home"])["xG"]
        .mean()
        .reset_index()
        .rename(columns={"xG": "avg_xG_scored"})
    )
    opp_stats = (
        df_past.groupby(["team_name", "was_home"])["xGC"]
        .mean()
        .reset_index()
        .rename(columns={"xGC": "avg_xGC_conceded"})
    )

    df_curr = df_curr.merge(team_stats, on=["team_name", "was_home"], how="left")
    df_curr = df_curr.merge(opp_stats, on=["team_name", "was_home"], how="left")

    df_curr["avg_xG_scored"] = df_curr["avg_xG_scored"].fillna(1.2)
    df_curr["avg_xGC_conceded"] = df_curr["avg_xGC_conceded"].fillna(1.2)

    df_curr["pred_team_xG"] = df_curr["avg_xG_scored"]
    df_curr["pred_opp_xG"] = df_curr["avg_xGC_conceded"]

    if "pred_xmin" not in df_curr.columns:
        df_curr["pred_xmin"] = df_curr.apply(predict_xmin, axis=1)
        df_curr["pred_xmin"] = np.clip(df_curr["pred_xmin"], 0, 90)

    team_xG_sum = df_curr.groupby("team_name")["rolling_xG_90_3"].transform("sum")
    df_curr["xG_share"] = np.where(
        team_xG_sum > 0, df_curr["rolling_xG_90_3"] / team_xG_sum, 0.0
    )

    df_curr["pred_player_xG"] = (
        df_curr["pred_team_xG"] * df_curr["xG_share"] * (df_curr["pred_xmin"] / 90.0)
    )
    df_curr["pred_player_xA"] = df_curr["rolling_xA_90_3"] * (
        df_curr["pred_xmin"] / 90.0
    )
    df_curr["prob_clean_sheet"] = np.exp(-df_curr["pred_opp_xG"])
    df_curr["exp_defcon"] = df_curr["rolling_defcon_90_3"] * (
        df_curr["pred_xmin"] / 90.0
    )
    df_curr["exp_saves"] = np.where(
        df_curr["position"].isin(["GKP", "GK"]),
        df_curr["rolling_saves_90_3"] * (df_curr["pred_xmin"] / 90.0),
        0.0,
    )

    return df_curr.apply(calculate_poisson_xp, axis=1)


def fit_predict_ml_backtest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    test_xmin: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trains XGBoost & Random Forest models for backtesting and applies domain guardrails."""
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, random_state=13
    )
    xgb_model.fit(X_train, y_train)
    raw_xgb = np.maximum(0.0, xgb_model.predict(X_test))
    pred_xgb = np.where(test_xmin == 0.0, 0.0, raw_xgb)

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100, random_state=13, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    raw_rf = np.maximum(0.0, rf_model.predict(X_test))
    pred_rf = np.where(test_xmin == 0.0, 0.0, raw_rf)

    return pred_xgb, pred_rf