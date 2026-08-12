"""
run_pipeline.py

Interactive runner for the MSCTHESIS pipeline. Lets you pick a single stage,
a range, a comma-separated list, or the whole thing — rather than having to
remember run order and exact script paths every time.

Usage:
    python run_pipeline.py

Then follow the prompt. Examples of valid input:
    5          -> run stage 5 only
    3,4,5      -> run stages 3, 4, 5
    1-6        -> run stages 1 through 6 inclusive
    all        -> run everything in order
    q          -> quit
"""

import subprocess
import sys
import os

# ---------------------------------------------------------------------
# Pipeline stages, in dependency order.
# Edit `path` here if a script gets renamed/moved.
# ---------------------------------------------------------------------
STAGES = [
    {
        "id": 1,
        "name": "TDA features — landscape (raw)",
        "path": "src/topology/tda_pipeline.py",
    },
    {
        "id": 2,
        "name": "TDA features — L^p-norm summary (depends on 1)",
        "path": "src/topology/lp_norm_features.py",
    },
    {
        "id": 3,
        "name": "GARCH residuals per asset",
        "path": "src/volatility/Garch.py",
    },
    {
        "id": 4,
        "name": "DCC baseline (fixed a, b) (depends on 3)",
        "path": "src/models/dcc_baseline.py",
    },
    {
        "id": 5,
        "name": "DCC topology model — lpnorm (depends on 2, 3)",
        "path": "src/models/dcc_topo.py",
    },
    {
        "id": 6,
        "name": "Permutation test — lpnorm (depends on 5)",
        "path": "src/models/permutation_test_lpnorm.py",
    },
    {
        "id": 7,
        "name": "Stats tests (LR test, R^2, etc.)",
        "path": "src/evaluation/stats_tests.py",
    },
    {
        "id": 8,
        "name": "Diagnostic: L^p-norm vs known crashes",
        "path": "scripts/diagnostic_lpnorm_vs_crashes.py",
    },
    {
        "id": 9,
        "name": "Check: lpnorm model vs realized correlation",
        "path": "scripts/check_lpnorm_vs_realized_corr.py",
    },
]


def print_menu():
    print()
    print("MSCTHESIS pipeline — available stages")
    print()
    for stage in STAGES:
        exists = os.path.exists(stage["path"])
        flag = "" if exists else "  [MISSING FILE]"
        print(f"  {stage['id']:>2}. {stage['name']}{flag}")
    print()
    print("Enter a stage number, a comma list (e.g. 3,4,5), a range (e.g. 1-6),")
    print("'all' to run everything in order, or 'q' to quit.")
    print()


def parse_selection(raw, max_id):
    raw = raw.strip().lower()
    if raw == "q":
        return None
    if raw == "all":
        return list(range(1, max_id + 1))

    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-")
                start, end = int(start), int(end)
                ids.update(range(start, end + 1))
            except ValueError:
                print(f"  Skipping unrecognized range: '{part}'")
        elif part.isdigit():
            ids.add(int(part))
        elif part:
            print(f"  Skipping unrecognized entry: '{part}'")

    return sorted(i for i in ids if 1 <= i <= max_id)


def run_stage(stage):
    path = stage["path"]
    if not os.path.exists(path):
        print(f"  SKIPPED — file not found: {path}")
        return False

    print(f"\n{'=' * 60}")
    print(f"Running stage {stage['id']}: {stage['name']}")
    print(f"  -> {path}")
    print(f"{'=' * 60}\n")

    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"\n  FAILED — stage {stage['id']} exited with code {result.returncode}")
        return False

    print(f"\n  OK — stage {stage['id']} completed")
    return True


def main():
    max_id = STAGES[-1]["id"]

    while True:
        print_menu()
        raw = input("Selection: ")
        selected_ids = parse_selection(raw, max_id)

        if selected_ids is None:
            print("Exiting.")
            return

        if not selected_ids:
            print("  No valid stages selected — try again.\n")
            continue

        stages_to_run = [s for s in STAGES if s["id"] in selected_ids]

        print(f"\nWill run {len(stages_to_run)} stage(s): "
              f"{', '.join(str(s['id']) for s in stages_to_run)}")
        confirm = input("Proceed? [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Cancelled.\n")
            continue

        for stage in stages_to_run:
            ok = run_stage(stage)
            if not ok:
                cont = input("\n  Continue with remaining stages anyway? [y/N]: ").strip().lower()
                if cont != "y":
                    print("  Stopping.\n")
                    break

        print("\nDone with this run. Back to menu.\n")


if __name__ == "__main__":
    main()