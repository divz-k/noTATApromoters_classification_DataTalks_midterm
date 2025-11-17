#!/usr/bin/env python
"""Lightweight runner that provides the requested main() scaffold.

This file imports `code.py` (which contains the pipeline) and then
prints a confirmation message. Importing `code.py` will execute its
notebook-style top-level code; this runner simply gives an explicit
entrypoint without modifying `code.py` itself.
"""

import sys


def main():
    print("Running ML pipeline...")

    # Try to detect trained model objects defined in `code.py` namespace
    try:
        import code as pipeline
    except Exception as e:
        print("Could not import code.py:", e)
        sys.exit(1)

    for name in ("xgb_model", "forest_model", "tree_model", "regression_model"):
        if hasattr(pipeline, name):
            print(f"Found trained model in code.py: {name}")


if __name__ == "__main__":
    main()
