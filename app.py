"""
app.py
------
CLI entry point for the Smart Investment Decision Fusion System.

Usage:
    python app.py --symbol TCS.NS
    python app.py --symbol TSLA --period 1y
    python app.py --symbol INFY.NS --no-plots
"""

import argparse
import sys

from pipeline import run_analysis
from src.utils.printing import print_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Investment Decision Fusion System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --symbol TCS.NS
  python app.py --symbol TSLA --period 1y
  python app.py --symbol INFY.NS --no-plots
        """,
    )
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="Stock ticker (e.g. TCS.NS, TSLA, AAPL).",
    )
    parser.add_argument(
        "--period", "-p",
        default="2y",
        choices=["1y", "2y", "3y", "4y", "5y"],
        help="Historical data period (default: 2y).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip chart generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = run_analysis(
            symbol         = args.symbol,
            period         = args.period,
            generate_plots = not args.no_plots,
        )
        print_result(result)

        if not args.no_plots and "chart_paths" in result:
            print("\n  📊 Charts saved:")
            for name, path in result["chart_paths"].items():
                if path:
                    print(f"     • {name}: {path}")

    except ValueError as exc:
        print(f"\n[ERROR] Invalid input: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n[ERROR] Runtime failure: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
