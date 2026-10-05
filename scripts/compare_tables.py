#!/usr/bin/env python3
"""Build the tables from the metrics of a new run and compare them, cell by cell, with the paper's.

The tables under tables/from_metrics/ come from the paper's metrics (results/raw/Evaluations).
This script runs the same table scripts on other metrics, by default yours (results/generated/Evaluations),
writes the result to tables/generated/ and prints, for each table, how many cells are identical and the
largest relative difference. A dataset that was not evaluated leaves its cells empty.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"-?\d+\.\d+|-?\d+")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluations", default="results/generated/Evaluations", help="metrics to build the tables from")
    args = parser.parse_args()
    output = ROOT / "tables/generated"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, IMPACTSYNTH_EVALUATIONS=args.evaluations, IMPACTSYNTH_TABLES="tables/generated")
    scripts = sorted(path for path in (ROOT / "scripts/tables").glob("table_*.py") if "leaderboards" not in path.name)
    for script in scripts:
        done = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, capture_output=True, text=True)
        if done.returncode:
            print(f"{script.name} failed: {(done.stderr or done.stdout).strip().splitlines()[-1]}")
    for reference in sorted((ROOT / "tables/from_metrics").glob("*.tex")):
        mine = output / reference.name
        if not mine.is_file():
            print(f"{reference.name:38s} not generated")
            continue
        rows = [[line for line in path.read_text().splitlines() if "&" in line] for path in (reference, mine)]
        if len(rows[0]) != len(rows[1]):
            print(f"{reference.name:38s} {len(rows[0])} rows in the paper's, {len(rows[1])} here")
            continue
        cells = same = empty = 0
        worst = (0.0, "")
        for paper_row, my_row in zip(*rows):
            for paper_cell, my_cell in zip(paper_row.split("&")[1:], my_row.split("&")[1:]):
                paper_numbers, my_numbers = NUMBER.findall(paper_cell), NUMBER.findall(my_cell)
                if not paper_numbers:
                    continue
                cells += 1
                if len(paper_numbers) != len(my_numbers):
                    empty += 1
                    continue
                same += paper_cell.strip() == my_cell.strip()
                paper_value, my_value = float(paper_numbers[0]), float(my_numbers[0])
                if paper_value and abs(my_value - paper_value) / abs(paper_value) > worst[0]:
                    worst = (abs(my_value - paper_value) / abs(paper_value),
                             f"{paper_value:g} -> {my_value:g} ({paper_row.split('&')[0].strip()[:24]})")
        print(f"{reference.name:38s} identical {same:3d}/{cells:3d}, empty {empty:3d}, "
              f"largest difference {100 * worst[0]:6.2f} %  {worst[1]}")


if __name__ == "__main__":
    main()
