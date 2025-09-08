#!/usr/bin/env bash
set -euo pipefail
ROOT="runs/sweep3_hot"
mkdir -p "$ROOT"
for ad in 1.3 1.5 1.7; do
  for tt in 1.6 1.8 2.0; do
    for kc in 0.08 0.12; do
      OUT="$ROOT/ad${ad}_tt${tt}_kc${kc}"
      python three_body_3d_ns_dtg_v3_5_full.py \
        --ic exp2 --tmax 1 --dt 0.0015 \
        --use_symplectic --dtg 1 --no_plots \
        --k_planar 0.6 \
        --alpha_dtg "$ad" --tau_theta "$tt" \
        --k_ctrl "$kc" --ctrl_signed \
        --dtheta_max 0.10 \
        --out "$OUT"
    done
  done
done
