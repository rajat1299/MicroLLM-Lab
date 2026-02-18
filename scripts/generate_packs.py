#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

TARGET_LINES = 60

PACK_ORDER = [
    "regex",
    "abc_music",
    "chess_pgn",
    "sql_snippets",
    "arithmetic",
    "json",
]


def _ensure_size(name: str, rows: list[str]) -> list[str]:
    unique_rows = list(dict.fromkeys(rows))
    if len(unique_rows) < TARGET_LINES:
        raise ValueError(f"{name}: expected at least {TARGET_LINES} rows, got {len(unique_rows)}")
    return unique_rows[:TARGET_LINES]


def generate_arithmetic() -> list[str]:
    add_no_carry = [
        f"{a}+{b}={a+b}"
        for a in range(10)
        for b in range(10)
        if a + b < 10
    ][:20]
    add_carry = [
        f"{a}+{b}={a+b}"
        for a in range(10)
        for b in range(10)
        if a + b >= 10
    ][:20]
    sub_rows = [
        f"{a}-{b}={a-b}"
        for a in range(10)
        for b in range(a + 1)
    ][:20]
    return _ensure_size("arithmetic", add_no_carry + add_carry + sub_rows)


def generate_chess_pgn() -> list[str]:
    rows: list[str] = []

    family_e4_e5 = {
        ("Nf3", "Nc6"): [("Bb5", "a6"), ("Bc4", "Bc5"), ("d4", "exd4"), ("c3", "Nf6")],
        ("Nf3", "Nf6"): [("Nxe5", "d6"), ("d4", "d6"), ("Nc3", "Nc6"), ("g3", "g6")],
        ("Nc3", "Nc6"): [("Nf3", "Nf6"), ("Bc4", "Bc5"), ("Bb5", "Nd4"), ("f4", "exf4")],
        ("Bc4", "Bc5"): [("Nf3", "Nc6"), ("c3", "Nf6"), ("d3", "d6"), ("Qh5", "Qe7")],
        ("d4", "exd4"): [("Nf3", "Nc6"), ("c3", "d5"), ("Bc4", "Bb4+"), ("Qxd4", "Nc6")],
    }
    for (w2, b2), third_moves in family_e4_e5.items():
        for w3, b3 in third_moves:
            rows.append(f"1. e4 e5 2. {w2} {b2} 3. {w3} {b3}")

    family_d4_d5 = {
        ("c4", "e6"): [("Nc3", "Nf6"), ("Nf3", "Nf6"), ("g3", "Nf6"), ("cxd5", "exd5")],
        ("c4", "c6"): [("Nc3", "Nf6"), ("Nf3", "Nf6"), ("e3", "e6"), ("Bf4", "Nf6")],
        ("Nf3", "Nf6"): [("c4", "e6"), ("g3", "g6"), ("Bf4", "e6"), ("e3", "e6")],
        ("Bf4", "Nf6"): [("e3", "e6"), ("Nf3", "c5"), ("c3", "e6"), ("h3", "c5")],
        ("g3", "Nf6"): [("Bg2", "e6"), ("Nf3", "c6"), ("c4", "dxc4"), ("Bg2", "g6")],
    }
    for (w2, b2), third_moves in family_d4_d5.items():
        for w3, b3 in third_moves:
            rows.append(f"1. d4 d5 2. {w2} {b2} 3. {w3} {b3}")

    family_e4_c5 = {
        ("Nf3", "d6"): [("d4", "cxd4"), ("Bb5+", "Nd7")],
        ("Nf3", "Nc6"): [("d4", "cxd4"), ("Bb5", "g6")],
        ("Nc3", "Nc6"): [("f4", "g6"), ("Nf3", "g6")],
        ("c3", "d5"): [("exd5", "Qxd5"), ("e5", "Nc6")],
        ("d4", "cxd4"): [("Nf3", "d6"), ("c3", "Nf6")],
    }
    for (w2, b2), third_moves in family_e4_c5.items():
        for w3, b3 in third_moves:
            rows.append(f"1. e4 c5 2. {w2} {b2} 3. {w3} {b3}")

    family_e4_c6 = {
        ("d4", "d5"): [("Nc3", "dxe4"), ("Nd2", "dxe4")],
        ("Nc3", "d5"): [("Nf3", "dxe4"), ("d4", "dxe4")],
        ("Nf3", "d5"): [("Nc3", "dxe4"), ("e5", "Bf5")],
        ("d3", "d5"): [("Nd2", "dxe4"), ("Nf3", "Bg4")],
        ("c4", "d5"): [("exd5", "cxd5"), ("cxd5", "Nf6")],
    }
    for (w2, b2), third_moves in family_e4_c6.items():
        for w3, b3 in third_moves:
            rows.append(f"1. e4 c6 2. {w2} {b2} 3. {w3} {b3}")

    return _ensure_size("chess_pgn", rows)


def generate_abc_music() -> list[str]:
    keys = ["C", "G", "D", "F", "A"]
    scalar = ["CDEF", "DEFG", "EFGA", "FGAB", "GABC", "ABCD", "BCDE", "DCBA", "EDCB", "FEDC"]
    arpeggio = ["CEGC", "DFAF", "EGBE", "FACE", "GBDG", "ACEA", "BDFB", "CEAC", "DFAD", "EGBD"]
    alternating = ["CDCD", "EFEF", "GAGA", "BGBG", "ACAC", "DFDF", "EAEA", "FGFG", "ABAB", "BCBC"]

    rows: list[str] = []

    for i in range(20):
        m1 = scalar[i % len(scalar)]
        m2 = scalar[(i + 1) % len(scalar)]
        rows.append(f"X:{i+1} K:{keys[i % len(keys)]} |{m1}|{m2}|{m1}|{m2}|")

    for i in range(20):
        n = i + 21
        m1 = arpeggio[i % len(arpeggio)]
        m2 = arpeggio[(i + 2) % len(arpeggio)]
        rows.append(f"X:{n} K:{keys[i % len(keys)]} |{m1}|{m1}|{m2}|{m2}|")

    for i in range(20):
        n = i + 41
        m1 = alternating[i % len(alternating)]
        m2 = alternating[(i + 3) % len(alternating)]
        rows.append(f"X:{n} K:{keys[i % len(keys)]} |{m1}|{m2}|{m1}|{m2}|")

    return _ensure_size("abc_music", rows)


def generate_sql_snippets() -> list[str]:
    tables = ["users", "orders", "products", "sessions"]
    cols = ["id", "name", "email", "total", "stock", "status", "age", "score"]
    fields = ["active", "status", "stock", "age", "score", "total", "valid"]
    ops = ["=", ">", ">=", "<", "<="]

    rows: list[str] = []
    for t_idx, table in enumerate(tables):
        for i in range(15):
            col = cols[(i + t_idx) % len(cols)]
            field = fields[(i + 2 * t_idx) % len(fields)]
            op = ops[i % len(ops)]
            value = (t_idx * 53 + i * 17 + 9) % 201
            rows.append(f"SELECT {col} FROM {table} WHERE {field}{op}{value};")

    return _ensure_size("sql_snippets", rows)


def generate_regex() -> list[str]:
    charclasses = ["[a-z]", "[a-z0-9]", "[A-Z]", "[A-Za-z]", "[A-Za-z0-9]"]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.org", "school.edu", "proton.me"]

    rows: list[str] = []
    for charclass in charclasses:
        for domain in domains:
            rows.append(f"^{charclass}+@{domain}$")
            rows.append("^{}+@{}$".format(charclass, domain.replace(".", r"\.")))

    return _ensure_size("regex", rows)


def generate_json() -> list[str]:
    key_pairs = [
        ("name", "age"),
        ("city", "pop"),
        ("item", "price"),
        ("team", "rank"),
        ("model", "score"),
    ]
    values = [
        "alice",
        "bob",
        "carol",
        "dave",
        "eve",
        "rome",
        "oslo",
        "delhi",
        "book",
        "pen",
        "lamp",
        "chair",
        "falcon",
        "otter",
        "nova",
        "zen",
        "atlas",
        "pixel",
        "orbit",
        "delta",
    ]

    rows: list[str] = []
    for p_idx, (k1, k2) in enumerate(key_pairs):
        for i in range(12):
            value = values[(p_idx * 7 + i) % len(values)]
            number = (p_idx * 137 + i * 17 + 23) % 1000
            rows.append(f'{{"{k1}":"{value}","{k2}":{number}}}')

    return _ensure_size("json", rows)


def generate_all_packs() -> dict[str, list[str]]:
    packs = {
        "regex": generate_regex(),
        "abc_music": generate_abc_music(),
        "chess_pgn": generate_chess_pgn(),
        "sql_snippets": generate_sql_snippets(),
        "arithmetic": generate_arithmetic(),
        "json": generate_json(),
    }
    if sorted(packs.keys()) != sorted(PACK_ORDER):
        raise ValueError("Pack definitions do not match PACK_ORDER")
    return packs


def _pack_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "packs"


def write_packs() -> None:
    pack_dir = _pack_dir()
    packs = generate_all_packs()
    for pack_id in PACK_ORDER:
        path = pack_dir / f"{pack_id}.txt"
        path.write_text("\n".join(packs[pack_id]) + "\n", encoding="utf-8")


def check_packs() -> int:
    pack_dir = _pack_dir()
    packs = generate_all_packs()
    mismatches: list[str] = []
    for pack_id in PACK_ORDER:
        expected = "\n".join(packs[pack_id]) + "\n"
        path = pack_dir / f"{pack_id}.txt"
        if not path.exists():
            mismatches.append(f"{pack_id}: missing file {path}")
            continue
        current = path.read_text(encoding="utf-8")
        if current != expected:
            mismatches.append(f"{pack_id}: file content differs from generated output")
    if mismatches:
        for mismatch in mismatches:
            print(mismatch)
        return 1
    print("All pack files match generated output.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic MicroLLM data packs.")
    parser.add_argument("--check", action="store_true", help="Check existing files match generator output.")
    args = parser.parse_args()

    if args.check:
        return check_packs()
    write_packs()
    print("Generated pack files in packs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
