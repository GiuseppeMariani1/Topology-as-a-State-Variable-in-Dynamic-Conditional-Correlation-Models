#!/usr/bin/env python3
"""
Master overnight run: exhaustive feature and window sensitivity analysis.

Part 1: Default window (250), all 4 feature variants
  - lpnorm
  - landscape
  - landscape + PCA-10
  - landscape + PCA-5

Part 2: All windows [100, 150, 200, 300, 400], all 4 feature variants
  (20 more OOS evaluations total)

Shutdown when complete.
"""

import subprocess
import sys
import time
import platform

def run_cmd(cmd, label):
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"[{time.strftime('%H:%M:%S')}] {label}")
    print('='*70)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {label} failed with code {result.returncode}")
        return False
    return True

def main():
    print(f"Starting overnight run at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # PART 1: Default window, all features
    part1_cmds = [
        ('python -m src.evaluation.oos_evaluation --features lpnorm', 
         'OOS: lpnorm (default window)'),
        ('python -m src.evaluation.oos_evaluation --features landscape', 
         'OOS: landscape (default window)'),
        ('python -m src.evaluation.oos_evaluation --features landscape --pca 10', 
         'OOS: landscape PCA-10 (default window)'),
        ('python -m src.evaluation.oos_evaluation --features landscape --pca 5', 
         'OOS: landscape PCA-5 (default window)'),
    ]

    print(f"\n{'#'*70}")
    print("PART 1: Default window (250), all feature variants")
    print(f"{'#'*70}")

    for cmd, label in part1_cmds:
        if not run_cmd(cmd, label):
            print("Aborting at Part 1")
            return False

    # PART 2: All windows, all features
    windows = [100, 150, 200, 300, 400]
    features_list = [
        ('lpnorm', None),
        ('landscape', None),
        ('landscape', 10),
        ('landscape', 5),
    ]

    print(f"\n{'#'*70}")
    print(f"PART 2: Windows {windows}, all feature variants")
    print(f"{'#'*70}")

    for window in windows:
        print(f"\n\n{'*'*70}")
        print(f"WINDOW = {window}")
        print(f"{'*'*70}")

        for feat, pca in features_list:
            if pca is None:
                cmd = f'python -m src.evaluation.oos_evaluation --features {feat} --window {window}'
                label = f'OOS: {feat} (window={window})'
            else:
                cmd = f'python -m src.evaluation.oos_evaluation --features {feat} --window {window} --pca {pca}'
                label = f'OOS: {feat} PCA-{pca} (window={window})'

            if not run_cmd(cmd, label):
                print(f"Aborting at window={window}")
                return False

    print(f"\n{'='*70}")
    print(f"All runs complete at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Shutdown
    print("\nShutting down in 30 seconds...")
    time.sleep(30)

    system = platform.system()
    if system == 'Windows':
        subprocess.run('shutdown /s /t 0')
    elif system == 'Darwin':  # macOS
        subprocess.run('osascript -e "tell app \\"System Events\\" to shut down"')
    else:  # Linux
        subprocess.run('sudo shutdown -h now')

if __name__ == "__main__":
    main()
