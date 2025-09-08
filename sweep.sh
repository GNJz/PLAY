#!/usr/bin/env bash
set -euo pipefail

ROOT="runs/sweep1"
mkdir -p "$ROOT"

for kp in 0.6; do
  for ad in 0.8 1.0; do
    for tt in 1.6 1.8 2.0; do
      for kc in 0.06 0.08 0.12; do
        OUT="$ROOT/kp${kp}_ad${ad}_tt${tt}_kc${kc}"
        # 이미 완료된 경우 스킵 (메타 json 존재 체크)
        if ls "$OUT"/meta/meta_*_dtg1.json >/dev/null 2>&1; then
          echo "[SKIP] $OUT (exists)"
          continue
        fi
        mkdir -p "$OUT"
        python three_body_3d_ns_dtg_v3_5_full.py \
          --ic exp2 --tmax 1 --dt 0.0015 \
          --use_symplectic --dtg 1 --no_plots \
          --k_planar "$kp" \
          --alpha_dtg "$ad" --tau_theta "$tt" \
          --k_ctrl "$kc" --ctrl_signed \
          --out "$OUT"
      done
    done
  done
done
