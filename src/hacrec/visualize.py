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
    fig = go.Figure(
        data=[
            go.Bar(name="RMSE", x=pred_df["strategy"], y=pred_df["rmse"]),
            go.Bar(name="MAE", x=pred_df["strategy"], y=pred_df["mae"]),
        ]
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
# 2. ALS loss curve
# ------------------------------------------------------------------


def plot_als_loss(out_dir: path.Path, viz_dir: path.Path) -> None:
    """Line plot of ALS training loss per iteration."""
    loss_path = out_dir / "als_loss_history.json"
    if not loss_path.exists():
        print("  Skipping ALS loss plot (als_loss_history.json not found)")
        return

    loss_history: list[float] = _load_json(loss_path)
    iterations = list(range(1, len(loss_history) + 1))

    fig = go.Figure(
        data=go.Scatter(x=iterations, y=loss_history, mode="lines+markers")
    )
    fig.update_layout(
        title="ALS Training Loss per Iteration",
        xaxis_title="Iteration",
        yaxis_title="Loss (MSE + L2 reg)",
    )
    fig.write_html(viz_dir / "als_loss_curve.html")
    print("  Saved als_loss_curve.html")


# ------------------------------------------------------------------
# 3. Ranking metrics comparison
# ------------------------------------------------------------------


def plot_ranking_metrics(rank_df: pd.DataFrame, viz_dir: path.Path) -> None:
    """Grouped bar chart of Precision@K, Recall@K, NDCG@K."""
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
# 4. Recommended movies comparison
# ------------------------------------------------------------------


def plot_recommendation_bars(rec_df: pd.DataFrame, viz_dir: path.Path) -> None:
    """Per-strategy grouped bar chart of top-N movie scores (all sampled users combined)."""
    strategies = rec_df["strategy"].unique().tolist()

    fig = go.Figure()
    for s in strategies:
        strat = rec_df[rec_df["strategy"] == s].sort_values("user_id")
        labels = [f"U{row.user_id}: {row.title}" for row in strat.itertuples()]
        fig.add_trace(go.Bar(name=s, x=labels, y=strat["score"].tolist()))

    fig.update_layout(
        barmode="group",
        title="Recommended Movie Scores by Strategy",
        xaxis_title="User: Movie",
        yaxis_title="Predicted Score",
        xaxis_tickangle=-60,
        margin=dict(b=220),
    )
    fig.write_html(viz_dir / "recs_scores_comparison.html")
    print("  Saved recs_scores_comparison.html")


# ------------------------------------------------------------------
# 5. Dashboard index
# ------------------------------------------------------------------


def build_dashboard(viz_dir: path.Path) -> None:
    """Generate an index.html dashboard that embeds all plot iframes."""
    als_section = ""
    if (viz_dir / "als_loss_curve.html").exists():
        als_section = """
    <section id="training-loss">
      <h2>Training Loss</h2>
      <div class="grid">
        <div class="card full">
          <h3>ALS Loss Curve</h3>
          <iframe src="als_loss_curve.html"></iframe>
        </div>
      </div>
    </section>"""

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Dashboard</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #161b22;
    --card-bg: #1c2128;
    --border: #30363d;
    --text: #e6edf3;
    --text-secondary: #8b949e;
    --accent: #58a6ff;
    --accent-soft: rgba(88, 166, 255, .12);
    --gradient-start: #58a6ff;
    --gradient-end: #bc8cff;
    --radius: 12px;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    min-height: 100vh;
  }}

  .wrapper {{
    margin: 0 auto;
    padding: 2.5rem 2rem 4rem;
  }}

  /* ---- header ---- */
  header {{
    margin-bottom: 2rem;
  }}
  header h1 {{
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: .35rem;
  }}
  header .subtitle {{
    color: var(--text-secondary);
    font-size: .9rem;
  }}
  header .subtitle code {{
    background: var(--accent-soft);
    padding: .15em .45em;
    border-radius: 4px;
    font-size: .85em;
    color: var(--accent);
  }}

  /* ---- nav ---- */
  nav {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    display: flex;
    gap: .25rem;
    padding: .35rem;
    margin-bottom: 2.5rem;
  }}
  nav a {{
    text-decoration: none;
    color: var(--text-secondary);
    padding: .5rem 1rem;
    border-radius: 8px;
    font-size: .85rem;
    font-weight: 500;
    transition: background .15s, color .15s;
  }}
  nav a:hover {{
    background: var(--accent-soft);
    color: var(--accent);
  }}

  /* ---- sections ---- */
  section {{
    margin-bottom: 3rem;
  }}
  section h2 {{
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 1rem;
    padding-left: .75rem;
    border-left: 3px solid var(--accent);
  }}

  /* ---- grid / cards ---- */
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(540px, 1fr));
    gap: 1.25rem;
  }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    transition: border-color .2s, box-shadow .2s;
  }}
  .card:hover {{
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent), 0 8px 24px rgba(0,0,0,.35);
  }}
  .card.full {{
    grid-column: 1 / -1;
  }}
  .card h3 {{
    font-size: .85rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: .04em;
    margin-bottom: .75rem;
  }}

  iframe {{
    width: 100%;
    height: 70vh;
    min-height: 400px;
    border: none;
    border-radius: 8px;
    background: #fff;
  }}

  /* ---- responsive ---- */
  @media (max-width: 640px) {{
    .wrapper {{ padding: 1.25rem 1rem; }}
    .grid {{ grid-template-columns: 1fr; }}
    nav {{ flex-wrap: wrap; }}
  }}
</style>
</head>
<body>
<div class="wrapper">

<header>
  <h1>Recommender Evaluation Dashboard</h1>
  <p class="subtitle">Generated by <code>hacrecviz</code></p>
</header>

<nav>
  <a href="#prediction-metrics">Prediction Metrics</a>
  <a href="#training-loss">Training Loss</a>
  <a href="#ranking-metrics">Ranking Metrics</a>
  <a href="#recommendations">Recommendations</a>
</nav>

<section id="prediction-metrics">
  <h2>Prediction Metrics</h2>
  <div class="grid">
    <div class="card full">
      <h3>RMSE &amp; MAE Comparison</h3>
      <iframe src="rmse_mae_comparison.html"></iframe>
    </div>
  </div>
</section>
{als_section}
<section id="ranking-metrics">
  <h2>Ranking Metrics</h2>
  <div class="grid">
    <div class="card full">
      <h3>Precision / Recall / NDCG</h3>
      <iframe src="ranking_metrics_comparison.html"></iframe>
    </div>
  </div>
</section>

<section id="recommendations">
  <h2>Recommendations</h2>
  <div class="grid">
    <div class="card full">
      <h3>Score Comparison (All Users)</h3>
      <iframe src="recs_scores_comparison.html"></iframe>
    </div>
  </div>
</section>

</div>
</body>
</html>
"""
    dashboard_path = viz_dir / "index.html"
    dashboard_path.write_text(html, encoding="utf-8")
    print("  Saved index.html")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    out = path.Path(OUT_DIR)
    viz_dir = ensure_dir(out, "viz")

    pred_path = out / "eval_prediction_metrics.csv"
    rank_path = out / "eval_ranking_metrics.csv"
    recs_path = out / "eval_recommendations.csv"

    if not pred_path.exists():
        print(f"No evaluation results found at {pred_path}.")
        print("Run `hacreceval` first to generate evaluation data.")
        return

    pred_df = pd.read_csv(pred_path)
    rank_df = pd.read_csv(rank_path)

    print("Generating visualisations \u2026")

    plot_rmse_mae(pred_df, viz_dir)
    plot_als_loss(out, viz_dir)
    plot_ranking_metrics(rank_df, viz_dir)

    if recs_path.exists():
        rec_df = pd.read_csv(recs_path)
        plot_recommendation_bars(rec_df, viz_dir)
    else:
        print("  Skipping recommendation plots (eval_recommendations.csv not found)")

    build_dashboard(viz_dir)

    print("\nDone \u2014 all plots saved to", viz_dir)


if __name__ == "__main__":
    main()
