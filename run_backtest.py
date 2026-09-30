import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.loader import load_and_prepare_data
from src.models import (
    FEATURE_COLS_ML,
    compute_poisson_backtest,
    fit_predict_ml_backtest,
    predict_xmin,
    prepare_ml_features,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# File paths
DATA_PATH = Path("data/merged_gw.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    logging.info("=== STARTING EXPANDING WINDOW BACKTEST EXPERIMENT ===")
    df = load_and_prepare_data(DATA_PATH)

    # 1. Precalculate xMin & positional One-Hot encodings
    df["pred_xmin"] = np.clip(df.apply(predict_xmin, axis=1), 0, 90)
    df = prepare_ml_features(df)

    metrics_per_gw = []

    # 2. Backtesting loop from Gameweek 2 through 38
    for gw in range(2, 39):
        logging.info(f"Evaluating Gameweek {gw}...")

        train_mask, test_mask = df["gw"] < gw, df["gw"] == gw
        X_train, y_train = (
            df.loc[train_mask, FEATURE_COLS_ML],
            df.loc[train_mask, "total_points"],
        )
        X_test, df_test = (
            df.loc[test_mask, FEATURE_COLS_ML],
            df.loc[test_mask].copy(),
        )

        # Generate model predictions via src.models helpers
        df_test["xP_Poisson"] = compute_poisson_backtest(
            df, target_gw=gw
        ).values
        df_test["xP_XGBoost"], df_test["xP_RF"] = fit_predict_ml_backtest(
            X_train, y_train, X_test, df_test["pred_xmin"]
        )

        # Evaluate metrics for all active players (>0 min) and regular starters (>=60 min)
        eval_all = df_test[df_test["minutes"] > 0]
        eval_starters = df_test[df_test["minutes"] >= 60]
        row_metrics = {"gw": gw}

        for model in ["Poisson", "XGBoost", "RF"]:
            row_metrics[f"MAE_{model}_all"] = mean_absolute_error(
                eval_all["total_points"], eval_all[f"xP_{model}"]
            )
            row_metrics[f"RMSE_{model}_all"] = np.sqrt(
                mean_squared_error(
                    eval_all["total_points"], eval_all[f"xP_{model}"]
                )
            )
            row_metrics[f"MAE_{model}_60m"] = mean_absolute_error(
                eval_starters["total_points"], eval_starters[f"xP_{model}"]
            )
            row_metrics[f"RMSE_{model}_60m"] = np.sqrt(
                mean_squared_error(
                    eval_starters["total_points"], eval_starters[f"xP_{model}"]
                )
            )

        metrics_per_gw.append(row_metrics)

    # 3. Save summary metrics to CSV
    metrics_df = pd.DataFrame(metrics_per_gw)
    summary_csv_path = OUTPUT_DIR / "backtest_metrics_summary.csv"
    metrics_df.to_csv(summary_csv_path, index=False)

    print("\n================ EXPERIMENT SUMMARY ================")
    print(
        f"Mean MAE  (min > 0)  - Poisson: {metrics_df['MAE_Poisson_all'].mean():.4f} | XGBoost: {metrics_df['MAE_XGBoost_all'].mean():.4f} | RF: {metrics_df['MAE_RF_all'].mean():.4f}"
    )
    print(
        f"Mean MAE  (min >= 60) - Poisson: {metrics_df['MAE_Poisson_60m'].mean():.4f} | XGBoost: {metrics_df['MAE_XGBoost_60m'].mean():.4f} | RF: {metrics_df['MAE_RF_60m'].mean():.4f}"
    )

    crossover_gw = metrics_df[
        metrics_df["MAE_XGBoost_60m"] < metrics_df["MAE_Poisson_60m"]
    ]["gw"].min()
    if pd.notna(crossover_gw):
        print(
            f"\nCROSSOVER POINT (min >= 60): XGBoost outperforms Poisson starting at GW{int(crossover_gw)}!"
        )
    else:
        print(
            "\nPoisson model maintained superior accuracy for regular starters across the entire season."
        )

    # 4. Visualization: 2x2 Grid (4 Plots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plots = [
        (axes[0, 0], "MAE_all", "All Active Players (>0 min) - MAE"),
        (axes[0, 1], "RMSE_all", "All Active Players (>0 min) - RMSE"),
        (axes[1, 0], "MAE_60m", "Regular Starters (>=60 min) - MAE"),
        (axes[1, 1], "RMSE_60m", "Regular Starters (>=60 min) - RMSE"),
    ]

    for ax, metric_key, title in plots:
        metric_name, filter_group = metric_key.split("_")
        for model, color, style in [
            ("Poisson", "blue", "-"),
            ("XGBoost", "green", "-"),
            ("RF", "orange", "--"),
        ]:
            ax.plot(
                metrics_df["gw"],
                metrics_df[f"{metric_name}_{model}_{filter_group}"],
                label=model,
                color=color,
                linestyle=style,
                linewidth=2,
            )
        if "60m" in filter_group and pd.notna(crossover_gw):
            ax.axvline(
                x=crossover_gw,
                color="red",
                linestyle=":",
                label=f"Crossover GW{int(crossover_gw)}",
            )
        ax.set_title(title)
        ax.set_xlabel("Gameweek (GW)")
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "FPL Predictive Engine Backtest: Model Error Evaluation (GW2 - GW38)",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    chart_path = OUTPUT_DIR / "backtest_performance_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    logging.info(f"Comparison chart saved to: {chart_path}")


if __name__ == "__main__":
    main()