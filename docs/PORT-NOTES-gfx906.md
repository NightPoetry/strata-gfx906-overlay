# PORT-NOTES — gfx906 port of the Strata rich server

## Scope

First successful build of the full-featured qwen35moe rich server
(`strata-poly-rich`) on AMD MI50 (gfx906) under ROCm 6.3.3. Before this port
the target was excluded at build time (`AND NOT STRATA_HIP_GFX906`) and the
qwen35moe family router dispatched the model to a reduced side engine.

## Port composition

1. Build switches. Three CMakeLists blocks (near lines 2050/2073/2081) changed
   from `AND NOT STRATA_HIP_GFX906` to include the HIP-GFX906 combination; a
   `STRATA_HIP_GFX906` option routes MXFP4 out (Blackwell FP4 MMA-only path,
   no gfx906 equivalent).
2. HIP/CUDA compatibility headers. `include/strata/hip_compat/` and
   `include/strata/platform/hip_compat/` provide: a driver VMM shim
   (`cuMem*` → `hipMem*`, integer handles with wrapper casts), cublas
   mappings (`cublasGetStream`, `SgemmStridedBatched` etc.), fp16 and
   intrinsics shims, math constants. CUDA headers are shadowed through
   include-path remapping, so unmodified upstream files resolve unchanged.
3. Sources with explicit port references (complete replacement copies in
   `overlay/`): `src/program/generate.cpp` (family router), `src/poly/mmq.cu`
   (18 MMQ kernels under `LANGUAGE HIP`, MXFP4 guarded out),
   `src/prefill/prefill.cpp`, `src/core/{vmm.cpp,device.cu,verify.cpp,mtp.cpp}`,
   `include/strata/kernels/{mrope.hpp,dp4a.hpp}`.
4. Model surgery, not code: community GGUF conversions silently drop the MTP
   draft layer (verified via GGUF metadata, `block_count=40`, no nextn keys).
   Production therefore uses a surgically stripped UD-Q4_K_M main file plus an
   independently packaged MTP draft (`--draft-model`), measured draft
   acceptance 84%.

## Failure modes encountered (chronological)

- Missing vendored CUDA headers → compat layer (see §2).
- `pair<uint64_t>` vs HIP handle type mismatches → integer handles with
  wrapper casts.
- rvalue address-of errors on `hipMemAllocationProp` → local lvalue staging.
- MXFP4 undefined symbol (`mul_mat_q_case<39>`) → guarded out under
  `STRATA_HIP_GFX906`.
- A decoy: a renamed harness binary was mistaken for the rich server; the
  real target is the separate CMake target built from `src/poly/main.cpp`.
- 16G VRAM cannot hold Q6 + draft: staging experiments recovered prefill but
  killed decode (Peter-Paul trade), so Q6 ships without draft.

## Production hardening (on top of the port)

- kv_grow cache squeeze: default adaptive KV growth evicted hot experts
  (residency 49% → 20-25% after a full-window test); idle-then-first requests
  paid a 14-19 s cold-staging penalty. Fix: `--kv-whole` allocates the whole
  262K KV (≈5.5 GiB at q8_0) at load; expert residency is pinned at 49%.
- Checkpoints (24×4096) and conversation parking (8 slots) live in host
  memory; session restore no longer touches VRAM layout.
- Prefill staging tuning (14.7K prompt, fixed protocol):
  564 → 736.5 (`--stage-ahead 2048`) → 814.6 tok/s (`--prompt-batch 1536`).
  GPU duty cycle measured: 83% during full-fill, 26% during typical small
  requests; the idle slices are cold-expert DMA between prompt batches
  (structural at 49% residency).
