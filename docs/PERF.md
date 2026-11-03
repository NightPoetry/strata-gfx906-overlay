# PERF — measurement methodology and results

All numbers measured on MI50 16G, ROCm 6.3.3, production config
(`configs/occamy-main-q4.json`), greedy decoding, fixed seeds, 85-task
four-room open-source probe (GSM8K-30 / HumanEval-20 / IFEval-15 /
JSON-20) and a scripted speed block (scripts/quant-probe/).

## Quantization sweep

| Quant | Quality | Short decode | Hot-expert residency |
|---|---|---|---|
| Q2_K_XL | 82/85 | 78.5 tok/s | 10145/10496 (96%) |
| Q4_K_M | 84/85 | 65.5 tok/s | 5172/10496 (49%) |
| Q6_K | 83/85 | 20.9 tok/s (no draft) | 3842/10240 (37.5%) |

Quality deltas sit inside the sampling noise band; decode speed is the
selector. Q4_K_M is the production choice.

## Prefill staging ladder

| Configuration | Prefill @14.7K |
|---|---|
| baseline (defaults, post kv-whole) | 704.3 tok/s |
| + `--stage-ahead 2048` | 736.5 tok/s |
| + `--prompt-batch 1536` | 814.6 tok/s |

`--prompt-batch 2048` was rejected: the workspace costs ≈1.4 MiB/token and
would have displaced expert residency on a 16G card.

## GPU duty cycle

- Full-fill (67179-token prompt, fresh): 83% busy, 49 compute bursts / 48
  staging gaps (0.2-0.4 s each, ≈8.5% wall time), 481 tok/s.
- Real traffic (231 s window): 26% busy, 150 short bursts — small-request
  pacing, no anomalous stalls.
