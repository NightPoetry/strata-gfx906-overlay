# strata-gfx906-overlay — AMD MI50 (gfx906) port overlay for Strata

An overlay distribution that brings the PolyStrata rich server (the full-featured
qwen35moe path of [Strata](https://github.com/Niko1221/Strata), MIT licensed) to
AMD Instinct MI50 accelerators under ROCm 6.3, where HIP support had never been
built upstream and ROCm 6.4 removed the gfx906 target entirely.

Upstream: Strata (MIT, Copyright Niko1221 and the Strata contributors).
This overlay adds an HIP/CUDA compatibility layer and build/route switches;
all additions are MIT licensed by FasFox. See `LICENSE` and `NOTICE`.

## What is included

```
overlay/    complete replacement copies of every upstream file touched by the
            port, plus the new include/strata/{hip_compat,platform/hip_compat}
            headers; apply by copying over a pristine Strata tree
diffs/      unified diffs against the pre-port state of the two files whose
            pristine copies were retained (CMakeLists.txt, generate.cpp)
configs/    production config for MI50 16G (UD-Q4_K_M + independent MTP draft,
            262K context, KV q8_0, pinned expert residency via --kv-whole)
scripts/    launch wrapper and the quantization benchmark suite
docs/       port notes and performance measurements
MANIFEST.sha256
```

## Why an overlay exists at all

Two upstream decisions kept gfx906 out:

1. `CMakeLists.txt` excludes the rich server target when `STRATA_HIP` is set
   (`AND NOT STRATA_HIP_GFX906`, three blocks near lines 2050/2073/2081);
2. the qwen35moe family router (`generate.cpp`) dispatches qwen35moe to a
   reduced side engine.

The port opens the switches, supplies an HIP/CUDA compatibility layer
(driver VMM shim, cublas mappings, fp16/intrinsics/math constants), rebuilds
the 18 MMQ kernels through `LANGUAGE HIP`, and excludes MXFP4 (Blackwell FP4
MMA-only path, absent on gfx906). Details in `docs/PORT-NOTES-gfx906.md`.

## Applying the overlay

```bash
# upstream tree, e.g. git clone https://github.com/Niko1221/Strata
cp -r overlay/* /path/to/strata/
# verify against MANIFEST.sha256, then:
cmake -B build -DSTRATA_HIP=ON -DSTRATA_HIP_GFX906=ON <your usual flags>
cmake --build build -j$(nproc)
```

ROCm 6.3 is required (6.4 dropped gfx906). Built and validated with
`strata-poly-rich` on MI50 16G; production config in `configs/`.

## Measured results (MI50 16G, 262K context)

| Metric | Value |
|---|---|
| Decode (UD-Q4_K_M + MTP draft, short prompt) | 65.5 tok/s |
| Prefill (14.7K prompt, after staging tuning) | 814.6 tok/s |
| Hot-expert residency (pinned, `--kv-whole`) | 49% |
| Quantization sweep (Q2/Q4/Q6 quality, 85 tasks) | 82/84/83 — noise band |
| Full-window prefill (253K tokens) | 265 tok/s |

See `docs/PERF.md` for methodology and the failure modes that shaped the
current configuration (kv_grow cache squeeze, Q6 draft famine, staging gaps).
