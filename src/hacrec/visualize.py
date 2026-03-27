"""Generate Plotly HTML visualisations from persisted evaluation results."""

import pathlib as path

import pandas as pd
import plotly.graph_objects as go

from .transform import OUT_DIR
from .util import ensure_dir


# ------------------------------------------------------------------
# 1. RMSE / MAE bar chart
# ------------------------------------------------------------------
def plot_rmse_mae(pred_df: pd.DataFrame, viz_dir: path.Path, split: str = "val") -> None:
    """Grouped bar chart comparing RMSE and MAE across strategies."""
    pred_df = pred_df.sort_values("rmse")
    min_rmse = pred_df["rmse"].min()
    split_label = "Validation" if split == "val" else "Test"
    file_prefix = "" if split == "val" else "test_"
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
        title=f"RMSE & MAE by Strategy — {split_label} (Lower is better)",
        xaxis_title="Strategy",
        yaxis_title="Error",
    )
    out_name = f"{file_prefix}rmse_mae_comparison.html"
    fig.write_html(viz_dir / out_name)
    print(f"  Saved {out_name}")


# ------------------------------------------------------------------
# 2. Ranking metrics comparison
# ------------------------------------------------------------------


def plot_ranking_metrics(rank_df: pd.DataFrame, viz_dir: path.Path, split: str = "val") -> None:
    """Grouped bar chart of Precision@K, Recall@K, NDCG@K."""
    rank_df = rank_df.sort_values("recall_at_k")
    k = int(rank_df["k"].iloc[0])
    split_label = "Validation" if split == "val" else "Test"
    file_prefix = "" if split == "val" else "test_"

    fig = go.Figure(
        data=[
            go.Bar(name=f"Precision@{k}", x=rank_df["strategy"], y=rank_df["precision_at_k"]),
            go.Bar(name=f"Recall@{k}", x=rank_df["strategy"], y=rank_df["recall_at_k"]),
            go.Bar(name=f"NDCG@{k}", x=rank_df["strategy"], y=rank_df["ndcg_at_k"]),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=f"Ranking Metrics (top-{k}) by Strategy — {split_label} (Higher is better)",
        xaxis_title="Strategy",
        yaxis_title="Score",
    )
    out_name = f"{file_prefix}ranking_metrics_comparison.html"
    fig.write_html(viz_dir / out_name)
    print(f"  Saved {out_name}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    out = path.Path(OUT_DIR)
    viz_dir = ensure_dir(out, "viz")

    val_pred_path = out / "val_prediction_metrics.csv"
    val_rank_path = out / "val_ranking_metrics.csv"

    if not val_pred_path.exists():
        print(f"No evaluation results found at {val_pred_path}.")
        print("Run `hacreceval` first to generate evaluation data.")
        return

    print("Generating visualisations \u2026")

    val_pred_df = pd.read_csv(val_pred_path)
    val_rank_df = pd.read_csv(val_rank_path)
    plot_rmse_mae(val_pred_df, viz_dir, split="val")
    plot_ranking_metrics(val_rank_df, viz_dir, split="val")

    test_pred_path = out / "test_prediction_metrics.csv"
    test_rank_path = out / "test_ranking_metrics.csv"
    if test_pred_path.exists():
        test_pred_df = pd.read_csv(test_pred_path)
        test_rank_df = pd.read_csv(test_rank_path)
        plot_rmse_mae(test_pred_df, viz_dir, split="test")
        plot_ranking_metrics(test_rank_df, viz_dir, split="test")

    print("\nDone \u2014 all plots saved to", viz_dir)


if __name__ == "__main__":
    main()
