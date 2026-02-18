# Pack Curation Notes

All built-in packs are deterministic, fixed-template corpora designed for tiny-model learnability.

## Global constraints
- Exactly 60 non-empty lines per pack
- One strict template per pack
- Deterministic ordering
- No duplicate lines

## Pack templates

- `arithmetic`
  - Template: `A{op}B=C`
  - Sections: 20 no-carry addition, 20 carry addition, 20 subtraction (`A>=B`)

- `chess_pgn`
  - Template: `1. W B 2. W B 3. W B`
  - Sections: 20 `e4 e5`, 20 `d4 d5`, 10 `e4 c5`, 10 `e4 c6`

- `abc_music`
  - Template: `X:N K:KEY |B1|B2|B3|B4|`
  - Sections: scalar motifs, arpeggio motifs, alternating motifs

- `sql_snippets`
  - Template: `SELECT col FROM table WHERE field{op}int;`
  - Grouped by table in fixed 15-line blocks

- `regex`
  - Template: `^{charclass}+@{domain}$`
  - Uses normalized domain variants with deterministic escaped-dot alternates

- `json`
  - Template: `{"k1":"value","k2":number}`
  - Five fixed key-pair families in deterministic blocks

## Tooling
- Generate/rewrite packs: `python scripts/generate_packs.py`
- Verify deterministic output only: `python scripts/generate_packs.py --check`
- Validate templates/constraints: `python scripts/validate_packs.py`
- Run learnability smoke gate: `python scripts/pack_smoke.py`
