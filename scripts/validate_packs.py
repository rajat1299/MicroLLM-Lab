#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_packs import PACK_ORDER, TARGET_LINES, generate_all_packs


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _check_arithmetic(lines: list[str]) -> None:
    pattern = re.compile(r"^(\d)([+-])(\d)=(\d{1,2})$")
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            _fail(f"arithmetic: invalid template at line {idx + 1}: {line}")
        a = int(match.group(1))
        op = match.group(2)
        b = int(match.group(3))
        c = int(match.group(4))

        if idx < 20:
            if op != "+" or a + b >= 10:
                _fail(f"arithmetic: first 20 lines must be no-carry additions (line {idx + 1})")
            if c != a + b:
                _fail(f"arithmetic: incorrect equation at line {idx + 1}")
        elif idx < 40:
            if op != "+" or a + b < 10:
                _fail(f"arithmetic: lines 21-40 must be carry additions (line {idx + 1})")
            if c != a + b:
                _fail(f"arithmetic: incorrect equation at line {idx + 1}")
        else:
            if op != "-" or a < b:
                _fail(f"arithmetic: lines 41-60 must be subtraction with a>=b (line {idx + 1})")
            if c != a - b:
                _fail(f"arithmetic: incorrect equation at line {idx + 1}")


def _check_chess(lines: list[str]) -> None:
    pattern = re.compile(r"^1\. (\S+) (\S+) 2\. (\S+) (\S+) 3\. (\S+) (\S+)$")
    move_pattern = re.compile(r"^[A-Za-z0-9+#=x]+$")
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 9 or parts[0] != "1." or parts[3] != "2." or parts[6] != "3.":
            _fail(f"chess_pgn: invalid move-number token layout at line {idx + 1}")
        match = pattern.match(line)
        if not match:
            _fail(f"chess_pgn: invalid template at line {idx + 1}: {line}")
        moves = [match.group(i) for i in range(1, 7)]
        if not all(move_pattern.match(move) for move in moves):
            _fail(f"chess_pgn: invalid move token at line {idx + 1}")

        w1, b1 = moves[0], moves[1]
        if idx < 20 and (w1 != "e4" or b1 != "e5"):
            _fail(f"chess_pgn: lines 1-20 must start with '1. e4 e5' (line {idx + 1})")
        if 20 <= idx < 40 and (w1 != "d4" or b1 != "d5"):
            _fail(f"chess_pgn: lines 21-40 must start with '1. d4 d5' (line {idx + 1})")
        if 40 <= idx < 50 and (w1 != "e4" or b1 != "c5"):
            _fail(f"chess_pgn: lines 41-50 must start with '1. e4 c5' (line {idx + 1})")
        if 50 <= idx < 60 and (w1 != "e4" or b1 != "c6"):
            _fail(f"chess_pgn: lines 51-60 must start with '1. e4 c6' (line {idx + 1})")


def _check_abc(lines: list[str]) -> None:
    scalar = {"CDEF", "DEFG", "EFGA", "FGAB", "GABC", "ABCD", "BCDE", "DCBA", "EDCB", "FEDC"}
    arpeggio = {"CEGC", "DFAF", "EGBE", "FACE", "GBDG", "ACEA", "BDFB", "CEAC", "DFAD", "EGBD"}
    alternating = {"CDCD", "EFEF", "GAGA", "BGBG", "ACAC", "DFDF", "EAEA", "FGFG", "ABAB", "BCBC"}
    pattern = re.compile(r"^X:(\d+) K:([CGDFA]) \|([A-G]{4})\|([A-G]{4})\|([A-G]{4})\|([A-G]{4})\|$")

    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            _fail(f"abc_music: invalid template at line {idx + 1}: {line}")
        n = int(match.group(1))
        b1, b2, b3, b4 = match.group(3), match.group(4), match.group(5), match.group(6)
        if n != idx + 1:
            _fail(f"abc_music: X index must be sequential (line {idx + 1})")

        if idx < 20:
            if not (b1 == b3 and b2 == b4 and b1 in scalar and b2 in scalar):
                _fail(f"abc_music: scalar block invalid at line {idx + 1}")
        elif idx < 40:
            if not (b1 == b2 and b3 == b4 and b1 in arpeggio and b3 in arpeggio):
                _fail(f"abc_music: arpeggio block invalid at line {idx + 1}")
        else:
            if not (b1 == b3 and b2 == b4 and b1 in alternating and b2 in alternating):
                _fail(f"abc_music: alternating block invalid at line {idx + 1}")


def _check_sql(lines: list[str]) -> None:
    allowed_tables = {"users", "orders", "products", "sessions"}
    allowed_cols = {"id", "name", "email", "total", "stock", "status", "age", "score"}
    allowed_fields = {"active", "status", "stock", "age", "score", "total", "valid"}
    pattern = re.compile(
        r"^SELECT ([a-z]+) FROM (users|orders|products|sessions) WHERE ([a-z]+)(>=|<=|=|>|<)(\d{1,3});$"
    )
    table_by_block = ["users", "orders", "products", "sessions"]

    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            _fail(f"sql_snippets: invalid template at line {idx + 1}: {line}")

        col = match.group(1)
        table = match.group(2)
        field = match.group(3)
        value = int(match.group(5))

        if col not in allowed_cols:
            _fail(f"sql_snippets: invalid column at line {idx + 1}")
        if table not in allowed_tables:
            _fail(f"sql_snippets: invalid table at line {idx + 1}")
        if field not in allowed_fields:
            _fail(f"sql_snippets: invalid condition field at line {idx + 1}")
        if not (0 <= value <= 200):
            _fail(f"sql_snippets: condition value out of range at line {idx + 1}")

        expected_table = table_by_block[idx // 15]
        if table != expected_table:
            _fail(f"sql_snippets: table grouping mismatch at line {idx + 1}")


def _check_regex(lines: list[str]) -> None:
    allowed_classes = {"[a-z]", "[a-z0-9]", "[A-Z]", "[A-Za-z]", "[A-Za-z0-9]"}
    allowed_domains = {"gmail.com", "yahoo.com", "outlook.com", "company.org", "school.edu", "proton.me"}
    pattern = re.compile(r"^\^(\[[A-Za-z0-9-]+\])\+@(.+)\$$")

    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            _fail(f"regex: invalid template at line {idx + 1}: {line}")
        charclass = match.group(1)
        domain = match.group(2)
        normalized_domain = domain.replace(r"\.", ".")

        if charclass not in allowed_classes:
            _fail(f"regex: invalid charclass at line {idx + 1}")
        if normalized_domain not in allowed_domains:
            _fail(f"regex: invalid domain at line {idx + 1}")


def _check_json(lines: list[str]) -> None:
    allowed_pairs = {
        ("name", "age"),
        ("city", "pop"),
        ("item", "price"),
        ("team", "rank"),
        ("model", "score"),
    }
    for idx, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"json: invalid JSON at line {idx + 1}: {exc}")  # pragma: no cover
        keys = list(obj.keys())
        if len(keys) != 2:
            _fail(f"json: line {idx + 1} must have exactly 2 keys")
        pair = (keys[0], keys[1])
        if pair not in allowed_pairs:
            _fail(f"json: invalid key pair at line {idx + 1}")
        if not isinstance(obj[keys[0]], str):
            _fail(f"json: first value must be string at line {idx + 1}")
        if not isinstance(obj[keys[1]], int):
            _fail(f"json: second value must be integer at line {idx + 1}")
        if not (0 <= obj[keys[1]] <= 999):
            _fail(f"json: integer out of range at line {idx + 1}")
        if " " in line:
            _fail(f"json: spaces are not allowed at line {idx + 1}")


PACK_CHECKERS = {
    "arithmetic": _check_arithmetic,
    "chess_pgn": _check_chess,
    "abc_music": _check_abc,
    "sql_snippets": _check_sql,
    "regex": _check_regex,
    "json": _check_json,
}


def main() -> int:
    pack_dir = ROOT / "packs"
    expected = generate_all_packs()
    required = set(PACK_ORDER)

    for pack_id in required:
        path = pack_dir / f"{pack_id}.txt"
        if not path.exists():
            _fail(f"missing pack file: {path}")

        lines = _read_lines(path)
        if len(lines) != TARGET_LINES:
            _fail(f"{pack_id}: expected {TARGET_LINES} lines, got {len(lines)}")
        if len(set(lines)) != len(lines):
            _fail(f"{pack_id}: duplicate lines found")

        joined = "".join(lines)
        unique_chars = len(set(joined))
        if unique_chars > 60:
            _fail(f"{pack_id}: unique character count too high ({unique_chars} > 60)")

        PACK_CHECKERS[pack_id](lines)

        expected_lines = expected[pack_id]
        if lines != expected_lines:
            for idx, (current, generated) in enumerate(zip(lines, expected_lines), start=1):
                if current != generated:
                    _fail(
                        f"{pack_id}: deterministic order/content mismatch at line {idx}\n"
                        f"current:   {current}\n"
                        f"generated: {generated}"
                    )
            _fail(f"{pack_id}: content mismatch with deterministic generator")

        print(f"{pack_id}: ok (lines={len(lines)}, unique_chars={unique_chars})")

    print("All packs validated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
