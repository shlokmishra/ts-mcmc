"""
Deprecated benchmark entry point.

Use benchmark_recorder.py as the authoritative benchmark workflow.
This wrapper exists only to preserve the old command shape.
"""

from benchmark_recorder import main


if __name__ == "__main__":
    print("benchmarks.py is now a thin wrapper over benchmark_recorder.py.")
    print("For authoritative runs, use: python benchmark_recorder.py ...")
    main()
