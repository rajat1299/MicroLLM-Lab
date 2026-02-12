# Limits

## Upload limits
- File type: `.txt` (UTF-8)
- Size: 200KB max
- Unique characters: 256 max
- Corpus characters: 200,000 max
- Strict blocklist checks

## Run limits
- `num_steps <= 2000`
- `block_size <= 64`
- `n_embd <= 64`
- `n_layer <= 2`
- `n_head <= 8`
- `n_embd % n_head == 0`
- Concurrent runs: 3

## Retention
- Run and upload data TTL: 24 hours
