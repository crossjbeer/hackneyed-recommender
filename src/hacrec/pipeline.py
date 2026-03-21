"""End-to-end pipeline: prepare → transform → fit → eval → visualize."""

import argparse
import sys
import time


# ------------------------------------------------------------------
# Step runner
# ------------------------------------------------------------------

def _step(name: str, fn, *args, **kwargs) -> None:
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    print(f"{'=' * 60}")
    t_start = time.perf_counter()
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        print(f"\n[ERROR] Step '{name}' failed: {exc}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - t_start
    print(f"[OK] {name} completed in {elapsed:.1f}s")


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full hackneyed-recommender pipeline:\n"
            "  prepare → transform → fit → eval → visualize"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip the data download/extraction step.",
    )
    parser.add_argument(
        "--skip-transform",
        action="store_true",
        help="Skip the data transformation step.",
    )
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        help="Skip the model fitting step.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the evaluation step.",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip the visualisation step.",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Re-fit models even if checkpoints already exist.",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Use the full MovieLens dataset instead of the small one (prepare step).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download the dataset even if it already exists locally (prepare step).",
    )
    return parser


# ------------------------------------------------------------------
# Step implementations
# ------------------------------------------------------------------

def _run_prepare(large: bool, overwrite: bool) -> None:
    from .prepare import prepare
    prepare(large=large, overwrite=overwrite)


def _run_transform() -> None:
    from .transform import main as transform_main

    transform_main()


def _run_fit(force_refit: bool) -> None:
    import pathlib as path

    from .fit import fit_recommenders
    from .load import load_user_item_matrix
    from .transform import OUT_DIR

    urm = load_user_item_matrix(path.Path(OUT_DIR))
    fit_recommenders(urm=urm, force_refit=force_refit)


def _run_eval(force_refit: bool) -> None:
    from .eval import _print_results, run_evaluation

    results = run_evaluation(force_refit=force_refit)
    _print_results(results)


def _run_viz() -> None:
    from .visualize import main as viz_main

    viz_main()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    args = make_parser().parse_args()
    t_total = time.perf_counter()

    if not args.skip_prepare:
        _step("prepare", _run_prepare, large=args.large, overwrite=args.overwrite)

    if not args.skip_transform:
        _step("transform", _run_transform)

    if not args.skip_fit:
        _step("fit", _run_fit, force_refit=args.force_refit)

    if not args.skip_eval:
        _step("eval", _run_eval, force_refit=args.force_refit)

    if not args.skip_viz:
        _step("visualize", _run_viz)

    total = time.perf_counter() - t_total
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
