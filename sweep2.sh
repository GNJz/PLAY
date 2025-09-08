#!/usr/bin/env bash
set -euo pipefail

ROOT="runs/sweep2"
mkdir -p "$ROOT"

# 고정/범위
kps="0.6"                 # 필요 시 0.5 0.6 0.7 로 늘릴 수 있음
ads="0.8 1.0 1.2"         # alpha_dtg 확장
tts="2.0 2.2 2.4 2.6"     # tau_theta 상향
kcs="0.06 0.08 0.12"

for kp in $kps; do
  for ad in $ads; do
    for tt in $tts; do
      for kc in $kcs; do
        OUT="$ROOT/kp${kp}_ad${ad}_tt${tt}_kc${kc}"
        # 존재하면 스킵
        if [ -f "$OUT/data/diagnostics_exp2_a1.0_dtg1.csv" ]; then
          echo "[SKIP] $OUT (exists)"; continue
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
