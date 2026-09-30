import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Loads player gameweek data from Vaastava's dataset, standardizes column names,

    and computes rolling normalized features preserving strict data leakage protection.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found at path: {csv_path}")

    logging.info(f"Loading raw gameweek data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)

    # 1. Column mapping and standardization
    column_mapping = {
        "element": "player_id",
        "name": "player_name",
        "position": "position",
        "team": "team_name",
        "round": "gw",
        "fixture": "fixture_id",
        "opponent_team": "opponent_team_id",
        "was_home": "was_home",
        "minutes": "minutes",
        "starts": "starts",
        "total_points": "total_points",
        "goals_scored": "goals_scored",
        "assists": "assists",
        "expected_goals": "xG",
        "expected_assists": "xA",
        "expected_goal_involvements": "xGI",
        "expected_goals_conceded": "xGC",
        "bps": "bps",
        "saves": "saves",
        "defensive_contribution": "defcon",
    }

    # Handle column name variance for gameweek round (GW vs round)
    if "GW" in df.columns and "round" not in df.columns:
        df = df.rename(columns={"GW": "round"})

    # Filter and rename available columns
    cols_to_keep = [c for c in column_mapping.keys() if c in df.columns]
    df = df[cols_to_keep].rename(columns=column_mapping)

    # Fill missing values with zeros
    for col in ["saves", "defcon", "starts", "xG", "xA", "xGC"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Ensure correct data types
    df["was_home"] = df["was_home"].astype(int)
    df["gw"] = df["gw"].astype(int)

    # Map position IDs to standard positional codes
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    if df["position"].dtype in [int, np.int64]:
        df["position"] = df["position"].map(pos_map)


    # 2. Compute rolling window features (3-GW and 5-GW windows, shift=1)
    logging.info("Computing rolling window features (3-GW and 5-GW windows, shift=1)...")
    df = df.sort_values(["player_id", "gw"]).reset_index(drop=True)
    grouped = df.groupby("player_id")

    for window in [3, 5]:
        df[f"rolling_min_{window}"] = (
            grouped["minutes"].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_starts_{window}"] = (
            grouped["starts"].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_pts_{window}"] = (
            grouped["total_points"].shift(1).rolling(window=window, min_periods=1).mean()
        )

        roll_sum_min = (
            grouped["minutes"].shift(1).rolling(window=window, min_periods=1).sum()
        )

        for col, new_name in [
            ("xG", f"rolling_xG_90_{window}"),
            ("xA", f"rolling_xA_90_{window}"),
            ("xGC", f"rolling_xGC_90_{window}"),
            ("defcon", f"rolling_defcon_90_{window}"),
            ("saves", f"rolling_saves_90_{window}"),
        ]:
            roll_sum_stat = (
                grouped[col].shift(1).rolling(window=window, min_periods=1).sum()
            )
            df[new_name] = np.where(
                roll_sum_min > 0, (roll_sum_stat / roll_sum_min) * 90.0, 0.0
            )

    # Fill NaN values for debutants or unpopulated historical windows
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    df[rolling_cols] = df[rolling_cols].fillna(0.0)

    return df