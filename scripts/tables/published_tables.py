#!/usr/bin/env python3
"""Write the 19 tables of the paper from its source (paper/arxiv_v1.tex)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = (ROOT / "paper/arxiv_v1.tex").read_text(encoding="utf-8")
    tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", source, re.DOTALL)
    if len(tables) != 19:
        raise SystemExit(f"Expected 19 tables, found {len(tables)}")
    for number, table in enumerate(tables, start=1):
        label = re.search(r"\\label\{([^}]+)\}", table)
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label.group(1) if label else f"table_{number}")
        output = ROOT / "tables" / f"table_{number:02d}_{name}.tex"
        output.write_text(table.strip() + "\n", encoding="utf-8")
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
