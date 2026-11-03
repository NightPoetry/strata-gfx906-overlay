#!/bin/sh
# Main (rich) PolyStrata server: full-feature engine path for qwen35moe (adaptive expert cache,
# KV q8_0, speculation window). ROCm runtime libs from the kvmem-hip install.
export LD_LIBRARY_PATH=/opt/kvmem-hip/rocm-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec /data/ai/poly/engine-poly/strata-poly-rich "$@"
