"""Generate Plotly HTML visualisations from persisted evaluation results."""

import json
import pathlib as path

import pandas as pd
import plotly.graph_objects as go

from .transform import OUT_DIR
from .util import ensure_dir


def _load_json(filepath: path.Path) -> dict | list:
    with open(filepath) as f:
        return json.load(f)


# ------------------------------------------------------------------
# 1. RMSE / MAE bar chart
# ------------------------------------------------------------------
def plot_rmse_mae(pred_df: pd.DataFrame, viz_dir: path.Path) -> None:
    """Grouped bar chart comparing RMSE and MAE across strategies."""
    pred_df = pred_df.sort_values("rmse")
    min_rmse = pred_df["rmse"].min()
    fig = go.Figure(
        data=[
            go.Bar(name="RMSE", x=pred_df["strategy"], y=pred_df["rmse"]),
            go.Bar(name="MAE", x=pred_df["strategy"], y=pred_df["mae"]),
        ]
    )
    fig.add_hline(
        y=min_rmse,
        line_dash="dot",
        line_color="rgba(255,255,255,0.5)",
        line_width=5.0,
        annotation_text=f"min RMSE = {min_rmse:.4f}",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="group",
        title="RMSE & MAE by Strategy",
        xaxis_title="Strategy",
        yaxis_title="Error",
    )
    fig.write_html(viz_dir / "rmse_mae_comparison.html")
    print("  Saved rmse_mae_comparison.html")


# ------------------------------------------------------------------
# 2. Ranking metrics comparison
# ------------------------------------------------------------------


def plot_ranking_metrics(rank_df: pd.DataFrame, viz_dir: path.Path) -> None:
    """Grouped bar chart of Precision@K, Recall@K, NDCG@K."""
    rank_df = rank_df.sort_values("recall_at_k")
    k = int(rank_df["k"].iloc[0])

    fig = go.Figure(
        data=[
            go.Bar(name=f"Precision@{k}", x=rank_df["strategy"], y=rank_df["precision_at_k"]),
            go.Bar(name=f"Recall@{k}", x=rank_df["strategy"], y=rank_df["recall_at_k"]),
            go.Bar(name=f"NDCG@{k}", x=rank_df["strategy"], y=rank_df["ndcg_at_k"]),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=f"Ranking Metrics (top-{k}) by Strategy",
        xaxis_title="Strategy",
        yaxis_title="Score",
    )
    fig.write_html(viz_dir / "ranking_metrics_comparison.html")
    print("  Saved ranking_metrics_comparison.html")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    out = path.Path(OUT_DIR)
    viz_dir = ensure_dir(out, "viz")

    pred_path = out / "eval_prediction_metrics.csv"
    rank_path = out / "eval_ranking_metrics.csv"

    if not pred_path.exists():
        print(f"No evaluation results found at {pred_path}.")
        print("Run `hacreceval` first to generate evaluation data.")
        return

    pred_df = pd.read_csv(pred_path)
    rank_df = pd.read_csv(rank_path)

    print("Generating visualisations \u2026")

    plot_rmse_mae(pred_df, viz_dir)
    plot_ranking_metrics(rank_df, viz_dir)

    print("\nDone \u2014 all plots saved to", viz_dir)


if __name__ == "__main__":
    main()
