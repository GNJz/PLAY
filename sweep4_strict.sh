#!/usr/bin/env bash
set -euo pipefail
ROOT="runs/sweep4_strict"
mkdir -p "$ROOT"
for ad in 1.75 1.85 1.95 2.05 2.10; do
  for tt in 1.40 1.50 1.60 1.70; do
    for kc in 0.12 0.16; do
      OUT="$ROOT/ad${ad}_tt${tt}_kc${kc}"
      python three_body_3d_ns_dtg_v3_5_full.py \
        --ic exp2 --tmax 1 --dt 0.0015 \
        --use_symplectic --dtg 1 --no_plots \
        --k_planar 0.6 \
        --alpha_dtg "$ad" --tau_theta "$tt" \
        --k_ctrl "$kc" --ctrl_signed \
        --dtheta_max 0.12 \
        --out "$OUT"
    done
  done
done
