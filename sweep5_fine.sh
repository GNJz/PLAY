#!/usr/bin/env bash
set -e
ROOT="runs/sweep5_fine"
mkdir -p "$ROOT"
for ad in 2.05 2.08 2.10 2.12 2.15; do
  for tt in 1.35 1.40 1.45 1.50 1.55; do
    for kc in 0.10 0.12 0.14 0.16 0.18; do
      OUT="$ROOT/ad${ad}_tt${tt}_kc${kc}"
      if [ -d "$OUT" ]; then echo "[SKIP] $OUT (exists)"; continue; fi
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
