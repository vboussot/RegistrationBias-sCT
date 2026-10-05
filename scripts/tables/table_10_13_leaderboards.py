#!/usr/bin/env python3
"""Fetch the public SynthRAD2025 test leaderboards behind Tables 10-13 and check the tables.

Sources (grand-challenge.org, final test phases):
  https://synthrad2025.grand-challenge.org/evaluation/test-task-1-mri/leaderboard/   Tables 10, 11
  https://synthrad2025.grand-challenge.org/evaluation/test-task-1-cbct/leaderboard/  Tables 12, 13

Writes results/leaderboards/{phase}.csv (rank, user, team and the mean of each metric), then
compares every row of tables/table_1{0,1,2,3}_*.tex with it at the printed precision.
--offline checks against the saved CSV only. Standard library only.
"""

import argparse
import csv
import html
import http.cookiejar
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/leaderboards"
PHASES = {"test-task-1-mri": ("10", "11"), "test-task-1-cbct": ("12", "13")}
METRICS = ["MAE", "PSNR", "MS-SSIM", "Dice", "HD95", "Dose MAE photon", "Dose MAE proton",
           "DVH photon", "DVH proton", "GPR photon", "GPR proton"]


def text(cell: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", cell))).strip()


def fetch(phase: str) -> list[dict]:
    # The page fills its table with a DataTables POST to itself, guarded by a CSRF cookie.
    url = f"https://synthrad2025.grand-challenge.org/evaluation/{phase}/leaderboard/"
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.open(url).read()
    token = next(cookie.value for cookie in jar if cookie.name == "_csrftoken")
    data = urllib.parse.urlencode({"draw": 1, "start": 0, "length": 100,
                                   "order[0][column]": 0, "order[0][dir]": "asc"}).encode()
    request = urllib.request.Request(url, data, {"X-CSRFToken": token, "Referer": url,
                                                 "X-Requested-With": "XMLHttpRequest"})
    rows = []
    for cells in json.load(opener.open(request))["data"]:
        cells = [text(cell) for cell in cells]
        user, _, team = cells[1].partition(" (")
        rows.append({"rank": cells[0].rstrip("stndrh"), "user": user.strip(), "team": team.rstrip(" )").strip(),
                     **{m: cells[5 + i].split(" ±")[0] for i, m in enumerate(METRICS)}})
    return rows


def check(table: str, rows: list[dict], metrics: list[str]) -> int:
    tex = next((ROOT / "tables").glob(f"table_{table}_*.tex")).read_text().replace("\n", " ")
    errors = 0
    for line in re.findall(r"\\textbf\{\d+\} & (.*?)\\\\", tex):
        row = next(r for r in rows if r["user"] in line)
        printed = re.findall(r"(?<![\w.])\d+\.\d+", line)
        for value, metric in zip(printed, metrics):
            decimals = len(value.split(".")[1])
            if abs(float(value) - float(row[metric])) > 0.51 * 10 ** -decimals:
                errors += 1
                print(f"table {table} {row['user']} {metric}: printed {value}, leaderboard {row[metric]}")
    print(f"table {table}: {'OK' if not errors else f'{errors} mismatch(es)'}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="check against results/leaderboards only")
    args = parser.parse_args()
    errors = 0
    for phase, (image_table, dose_table) in PHASES.items():
        path = OUTPUT / f"{phase}.csv"
        if not args.offline:
            rows = fetch(phase)
            OUTPUT.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        rows = list(csv.DictReader(path.open()))
        errors += check(image_table, rows, METRICS[:5]) + check(dose_table, rows, METRICS[5:])
    raise SystemExit(errors > 0)


if __name__ == "__main__":
    main()
